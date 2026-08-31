"""OPT — 행 값을 forward-only 로 최적화한다.

선행연구의 OPT 는 gold logprob 에 대해 행에 Adam 15스텝을 돌린다.  이 체크포인트는
W4A16 Marlin 커널에 backward 가 없어 경사를 못 얻으므로, **SPSA**(동시 섭동
확률근사)로 대체한다 — 차원과 무관하게 반복당 forward 2회면 된다.

2560차원 전체를 직접 흔들면 수렴이 느리므로 저차원 부분공간에서 탐색한다:

    e(θ) = e0 · (1 + θ_0) + Σ_i θ_i · b_i      b_i = 정규직교 랜덤 방향

θ 는 (1 + n_basis) 차원.  θ_0 는 UNEMBED_P 방향의 이득, 나머지는 보정항이다.
"""
import numpy as np
from . import live
from .core import consts


def make_basis(n_basis, dim, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n_basis, dim)).astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B


class RowOptimizer:
    """한 편집 단계(16행)의 값을 SPSA 로 최적화."""

    def __init__(self, addrs, e0, scorer, n_basis=48, seed=0,
                 a=0.05, c=0.06, gamma=0.101, alpha=0.602, heads=None,
                 basis=None):
        """heads: 편집할 헤드 인덱스.  None 이면 16개 전부.

        2-gram 헤드(0~7)는 마지막 2토큰만 보므로, 그 중 하나가 숫자가 되는
        순간 문맥이 일반적이 된다 (예: ('3',' ') 를 코드가 3으로 시작하는
        285개 종목이 공유).  3-gram 헤드(8~15)만 쓰면 부수 피해가 급감한다.
        """
        C = consts()
        all_addrs = np.asarray(addrs, dtype=np.int64)
        self.heads = list(range(C.n_heads_total)) if heads is None else list(heads)
        self.addrs = all_addrs[self.heads]
        e0_full = np.asarray(e0, dtype=np.float32).reshape(C.n_heads_total, C.row_dim)
        self.e0 = e0_full[self.heads].reshape(-1)
        self.dim = self.e0.size
        self.scorer = scorer
        if basis is not None:
            # 전체 2560차원 기저를 선택한 헤드 부분만 잘라 쓴다
            Bf = np.asarray(basis, dtype=np.float32).reshape(-1, C.n_heads_total, C.row_dim)
            Bsel = Bf[:, self.heads, :].reshape(len(Bf), -1)
            Bsel = Bsel / (np.linalg.norm(Bsel, axis=1, keepdims=True) + 1e-12)
            extra = make_basis(max(0, n_basis - len(Bsel)), self.dim, seed)
            self.B = np.concatenate([Bsel, extra]) if len(extra) else Bsel
        else:
            self.B = make_basis(n_basis, self.dim, seed)
        self.theta = np.zeros(1 + len(self.B), dtype=np.float32)
        self.a, self.c, self.gamma, self.alpha = a, c, gamma, alpha
        self.rng = np.random.default_rng(seed + 1)
        self.original = None
        self.history = []
        self.probe_every = 2

    # -- e(theta) ---------------------------------------------------------
    def _e(self, theta):
        return self.e0 * (1.0 + theta[0]) + theta[1:] @ self.B

    def _rows(self, theta):
        C = consts()
        return self._e(theta).reshape(len(self.heads), C.row_dim)

    # -- one objective evaluation ----------------------------------------
    def evaluate(self, theta):
        prev = live.write(self.addrs, self._rows(theta))
        if self.original is None:
            self.original = prev
        return self.scorer()

    def baseline(self):
        """편집 전 점수 (원본 행 그대로)."""
        return self.scorer()

    # -- SPSA -------------------------------------------------------------
    def step(self, k):
        """SPSA 한 스텝.  theta 를 갱신하고 그 지점의 실제 점수를 돌려준다."""
        ak = self.a / (k + 1 + 10) ** self.alpha
        ck = self.c / (k + 1) ** self.gamma
        delta = self.rng.choice([-1.0, 1.0], size=self.theta.shape).astype(np.float32)
        lp = self.evaluate(self.theta + ck * delta)
        lm = self.evaluate(self.theta - ck * delta)
        ghat = (lp - lm) / (2.0 * ck) * delta      # maximize -> ascend
        self.theta = self.theta + ak * ghat
        # 갱신된 theta 에서 실제로 재본다.  섭동 두 점의 평균을 쓰면 best 추적이
        # theta 와 어긋난다.  매 스텝 재면 평가 3회이므로 격번으로 잰다.
        if k % self.probe_every == 0:
            return self.evaluate(self.theta)
        return None

    def run(self, n_steps=200, log_every=20, verbose=True):
        """SPSA 로 행 값을 찾는다.

        주의: theta=0 은 "원본 행"이 아니라 이미 UNEMBED_P 가 쓰인 상태다.
        따라서 baseline(원본) 을 이기지 못하면 theta 를 남기면 안 되고
        **원본 행을 복원**해야 한다.
        """
        base = self.baseline()          # 원본 행 상태에서 측정
        if verbose:
            print(f"  baseline logprob {base:+.4f}")

        init = self.evaluate(self.theta)   # UNEMBED_P 자체의 점수
        best, best_theta = init, self.theta.copy()
        if verbose:
            print(f"  UNEMBED_P logprob {init:+.4f}")

        for k in range(n_steps):
            cur = self.step(k)
            if cur is None:
                continue
            self.history.append(cur)
            if cur > best:
                best, best_theta = cur, self.theta.copy()
            if verbose and (k + 1) % log_every == 0:
                print(f"  step {k+1:4d}  logprob {cur:+.4f}   best {best:+.4f}")

        improved = best > base
        if improved:
            self.theta = best_theta
            final = self.evaluate(self.theta)
        else:
            self.restore()              # 원본으로 되돌린다
            final = base
        if verbose:
            print(f"  final logprob {final:+.4f}  (baseline {base:+.4f}, "
                  f"Δ {final-base:+.4f})" + ("" if improved else "  [개선 없음 — 원본 복원]"))
        return {"baseline": base, "unembed_p": init, "final": final,
                "improved": bool(improved), "theta": self.theta,
                "rows": self._rows(self.theta), "history": self.history}

    def restore(self):
        if self.original is not None:
            live.write(self.addrs, self.original)
            return True
        return False


# 단계별 헤드 정책 -------------------------------------------------------
# 2-gram 헤드(0~7)는 마지막 2토큰만 본다.  그 2칸이 회사명 토큰이면 문맥이
# 고유하지만(step 0: 평균 1.4개 종목 공유), 숫자가 끼면 일반적이 된다
# (step 2: 평균 200개, 최대 643개).  따라서 앞 단계에서만 16헤드를 다 쓴다.
#
#   step 0  전 헤드   문맥 (이름끝, 이름끝-1)      공유 1.4
#   step 1  전 헤드   문맥 (' ', 이름끝)           공유 3.4
#   step 2+ 3-gram    문맥 (숫자, ' ')             공유 200  <- 2-gram 배제

ALL_HEADS = list(range(16))
TRIGRAM_HEADS = list(range(8, 16))
BIGRAM_HEADS = list(range(0, 8))


def heads_for_step(step, policy="adaptive"):
    if policy == "all":
        return ALL_HEADS
    if policy == "3gram":
        return TRIGRAM_HEADS
    if policy == "2gram":
        return BIGRAM_HEADS
    if policy == "adaptive":
        return ALL_HEADS if step <= 1 else TRIGRAM_HEADS
    raise ValueError(policy)


def run_with_restarts(addrs, e0, scorer, n_restarts=3, n_steps=40, n_basis=32,
                      heads=None, seed0=0, verbose=False):
    """같은 단계를 서로 다른 시드로 여러 번 돌려 최선만 남긴다.

    SPSA 는 분산이 커서 같은 사례가 실행마다 2/7 ~ 5/7 을 오간다.  한 번에
    반복을 늘리는 것보다 **짧게 여러 번 시작**하는 편이 같은 예산에서 낫다 —
    나쁜 초기 섭동에 갇히지 않기 때문이다.

    각 재시작은 이전 최선 값 위에서가 아니라 **원본 행에서** 다시 출발한다.
    """
    best = None
    for r in range(n_restarts):
        o = RowOptimizer(addrs, e0, scorer, n_basis=n_basis,
                         seed=seed0 + 977 * r, heads=heads)
        res = o.run(n_steps=n_steps, verbose=False)
        if best is None or res["final"] > best[0]["final"]:
            if best is not None:
                best[1].restore()          # 진 쪽은 되돌린다
            best = (res, o)
        else:
            o.restore()
        if verbose:
            print(f"    restart {r}: {res['final']:+.3f} "
                  f"(best {best[0]['final']:+.3f})", flush=True)
    return best[0], best[1]


class JointOptimizer:
    """모든 편집 지점을 **동시에** 최적화한다.

    단계별로 따로 돌리면 뒤 단계를 고치면서 앞 단계를 깨뜨린다 — 자연 질문
    형식에서 접두 일치가 3/7 → 2/7 로 줄어든 것이 그 예다.  각 단계가 자기
    목적함수를 독립적으로 올려도 전체는 나빠질 수 있다.

    여기서는 모든 지점의 theta 를 이어붙여 하나의 벡터로 두고 SPSA 를 돌린다.
    평가 1회가 전체 구성을 한꺼번에 재므로 조정 실패가 원리적으로 없다.
    SPSA 는 차원과 무관하게 반복당 2회 평가면 되므로 비용도 같다.

    선행연구의 Joint OPT 에 대응한다 (fact 가 많아지면 그쪽도 이 방식이 기본).
    """

    def __init__(self, sites, scorer, n_basis=24, seed=0,
                 a=0.05, c=0.06, gamma=0.101, alpha=0.602):
        """sites: [(addrs, e0, heads), ...]  편집 지점 목록."""
        C = consts()
        self.scorer = scorer
        self.parts, self.slices = [], []
        off = 0
        for k, (addrs, e0, heads) in enumerate(sites):
            hs = list(range(C.n_heads_total)) if heads is None else list(heads)
            aa = np.asarray(addrs, dtype=np.int64)[hs]
            e0 = np.asarray(e0, dtype=np.float32).reshape(-1)
            # e0 는 16헤드 전체(2560)로 올 수도, 선택 헤드분만(len(hs)*row_dim)
            # 으로 올 수도 있다.  전체면 잘라 쓰고, 이미 잘려 있으면 그대로 쓴다.
            # 잘못 자르면 theta=0 이 원본을 재현하지 못해 시작점부터 망가진다.
            if e0.size == C.n_heads_total * C.row_dim:
                ee = e0.reshape(C.n_heads_total, C.row_dim)[hs].reshape(-1)
            elif e0.size == len(hs) * C.row_dim:
                ee = e0
            else:
                raise ValueError(f"e0 크기 {e0.size} 가 헤드 구성과 맞지 않음")
            B = make_basis(n_basis, ee.size, seed + 131 * k)
            self.parts.append({"addrs": aa, "e0": ee, "B": B, "heads": hs,
                               "orig": None})
            self.slices.append(slice(off, off + 1 + n_basis))
            off += 1 + n_basis
        self.theta = np.zeros(off, dtype=np.float32)
        self.a, self.c, self.gamma, self.alpha = a, c, gamma, alpha
        self.rng = np.random.default_rng(seed + 7)
        self.history = []

    def _write(self, theta):
        C = consts()
        for p, sl in zip(self.parts, self.slices):
            th = theta[sl]
            e = p["e0"] * (1.0 + th[0]) + th[1:] @ p["B"]
            prev = live.write(p["addrs"], e.reshape(len(p["heads"]), C.row_dim))
            if p["orig"] is None:
                p["orig"] = prev

    def evaluate(self, theta):
        self._write(theta)
        return self.scorer()

    def restore(self):
        n = 0
        for p in self.parts:
            if p["orig"] is not None:
                live.write(p["addrs"], p["orig"])
                n += len(p["addrs"])
        return n

    def check_identity(self, tol=1e-6):
        """theta=0 이 원본 행을 그대로 재현하는지 확인.

        재현하지 못하면 시작점부터 망가진 상태로 탐색이 시작되어
        (실측 init -10.19 vs baseline +0.06) SPSA 가 구덩이에서 못 나온다.
        """
        cur = [live.read(p["addrs"]).copy() for p in self.parts]
        self._write(np.zeros_like(self.theta))
        after = [live.read(p["addrs"]) for p in self.parts]
        ok = all(np.allclose(a, b, atol=tol) for a, b in zip(cur, after))
        for p, c in zip(self.parts, cur):
            live.write(p["addrs"], c)
        return ok

    def run(self, n_steps=150, log_every=25, verbose=True):
        base = self.scorer()                      # 원본 상태
        init = self.evaluate(self.theta)          # 초기 구성
        best, best_theta = max(base, init), self.theta.copy()
        if verbose:
            print(f"  baseline {base:+.3f}  init {init:+.3f}  "
                  f"({len(self.parts)}개 지점, theta {self.theta.size}차원)", flush=True)
        for k in range(n_steps):
            ak = self.a / (k + 11) ** self.alpha
            ck = self.c / (k + 1) ** self.gamma
            d = self.rng.choice([-1.0, 1.0], size=self.theta.shape).astype(np.float32)
            lp = self.evaluate(self.theta + ck * d)
            lm = self.evaluate(self.theta - ck * d)
            self.theta = self.theta + ak * (lp - lm) / (2.0 * ck) * d
            if k % 2 == 0:
                cur = self.evaluate(self.theta)
                self.history.append(cur)
                if cur > best:
                    best, best_theta = cur, self.theta.copy()
                if verbose and (k + 1) % log_every == 0:
                    print(f"  step {k+1:4d}  {cur:+.3f}  best {best:+.3f}", flush=True)
        if best > base:
            self.theta = best_theta
            final = self.evaluate(self.theta)
        else:
            self.restore()
            final = base
        if verbose:
            print(f"  final {final:+.3f}  (baseline {base:+.3f}, Δ {final-base:+.3f})"
                  + ("" if best > base else "  [개선 없음 — 복원]"), flush=True)
        return {"baseline": base, "final": final, "improved": bool(best > base)}
