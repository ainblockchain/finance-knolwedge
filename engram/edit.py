"""Plan, solve and apply a single-fact pin-point edit to the Engram table.

A fact is (trigger, answer).  The Engram row read at position p contributes to
predicting token p+1, so writing the fact means editing the rows addressed at
every position whose *next* token belongs to the answer.

    "삼성전자 종목코드는" + " 005930"
      pos of '드는'  -> steers ' '
      pos of ' '     -> steers '0'
      ...

Each such position has its own 16 addresses.  Late answer positions have
generic trigrams ('5','9','3') whose rows are shared with every number in the
corpus, so `plan()` reports a genericness score per position and you choose
which ones to actually write.
"""
import json, os, datetime
import numpy as np
from . import core
from .core import consts, addresses, read_rows, write_rows, encode, locate

BACKUP_DIR = os.environ.get("ENGRAM_BACKUP_DIR",
                            "/mnt/newdata/qwen3.8/engram/backups")


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan(trigger: str, answer: str, pre_steps: int = 0):
    """-> dict with one entry per edit position."""
    C = consts()
    t_ids, a_ids = encode(trigger), encode(answer)
    ids = encode(trigger + answer)
    if ids != t_ids + a_ids:
        raise ValueError(
            "토크나이저 경계에서 trigger/answer 가 재분절된다. "
            f"concat={len(ids)} vs t+a={len(t_ids)+len(a_ids)}. "
            "answer 앞의 공백 위치를 바꾸거나 trigger 를 조정하라."
        )
    tok = core.tokenizer()

    steps = []
    # 음수 step = trigger 내부 위치.  그 자리의 행도 어텐션을 통해 답 위치의
    # h_t 를 바꾸므로 추가 채널이 된다.  문맥이 회사명 토큰이라 부수 피해도 낮다.
    for k in range(1, pre_steps + 1):
        p = len(t_ids) - 1 - k
        if p < 2:
            break
        ctx = [ids[p - j] if p - j >= 0 else 0 for j in range(C.ngram_size)]
        steps.append({
            "step": -k, "pos": p, "target_id": ids[p + 1],
            "target": tok.decode([ids[p + 1]]),
            "ctx_ids": ctx, "ctx": [tok.decode([c]) for c in ctx],
            "addrs": addresses(ids, p).tolist(),
        })
    steps.reverse()
    for i in range(len(a_ids)):
        p = len(t_ids) - 1 + i          # position whose row steers a_ids[i]
        ctx = [ids[p - j] if p - j >= 0 else 0 for j in range(C.ngram_size)]
        steps.append({
            "step": i,
            "pos": p,
            "target_id": a_ids[i],
            "target": tok.decode([a_ids[i]]),
            "ctx_ids": ctx,                                  # [cur, -1, -2]
            "ctx": [tok.decode([c]) for c in ctx],
            "addrs": addresses(ids, p).tolist(),
        })
    return {"trigger": trigger, "answer": answer,
            "trigger_ids": t_ids, "answer_ids": a_ids, "steps": steps}


def genericness(pl, corpus_ids, sample_positions=None):
    """How often does each step's context n-gram occur in a reference corpus?

    Returns per-step hit counts for the 2-gram and 3-gram contexts.  High counts
    mean the rows are shared with a lot of ordinary text: writing them is loud.
    """
    C = consts()
    ids = np.asarray(corpus_ids, dtype=np.int64)
    bi = {}
    tri = {}
    for p in range(2, len(ids)):
        bi[(int(ids[p-1]), int(ids[p]))] = bi.get((int(ids[p-1]), int(ids[p])), 0) + 1
        key = (int(ids[p-2]), int(ids[p-1]), int(ids[p]))
        tri[key] = tri.get(key, 0) + 1
    out = []
    for s in pl["steps"]:
        c = s["ctx_ids"]
        out.append({
            "step": s["step"],
            "bigram_hits": bi.get((c[1], c[0]), 0),
            "trigram_hits": tri.get((c[2], c[1], c[0]), 0),
        })
    return out


# --------------------------------------------------------------------------
# UNEMBED_P : closed-form row value
# --------------------------------------------------------------------------

def _read_partial(name, row_indices):
    path, dtype, shape, start = core.index()[name]
    assert dtype == "BF16"
    width = shape[1]
    out = np.empty((len(row_indices), width), dtype=np.float32)
    with open(path, "rb") as fh:
        for i, r in enumerate(row_indices):
            fh.seek(start + r * width * 2)
            out[i] = core._bf16_to_f32(fh.read(width * 2))
    return out


_WV = None

def value_proj():
    global _WV
    if _WV is None:
        _WV = core.read_tensor(core.PLE + "value_proj.weight")   # [2560, 2560]
    return _WV


def unembed_p(target_id: int, gain: float = 1.0):
    """Closed-form e* whose value-path projection points at the target token.

    Solve  W_V e = u_y   for e, then scale so that ||e|| = gain * (typical row
    norm * sqrt(n_heads)), keeping the write in the same magnitude regime as
    the rows it replaces.
    """
    C = consts()
    u = _read_partial("lm_head.weight", [target_id])[0]          # [2560]
    e = np.linalg.solve(value_proj(), u).astype(np.float32)      # [2560]
    typical = 0.10 * np.sqrt(C.n_heads_total)                    # measured base
    e *= gain * typical / (np.linalg.norm(e) + 1e-12)
    return e


def split_heads(e):
    """[2560] -> [16, 160], the per-head rows in table layout."""
    C = consts()
    return np.asarray(e, dtype=np.float32).reshape(C.n_heads_total, C.row_dim)


# --------------------------------------------------------------------------
# apply / backup / restore
# --------------------------------------------------------------------------

def backup(addrs, tag):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    addrs = np.asarray(addrs, dtype=np.int64)
    orig = read_rows(addrs)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(BACKUP_DIR, f"{tag}-{stamp}.npz")
    np.savez_compressed(path, addrs=addrs, rows=orig)
    return path


def restore(path):
    z = np.load(path)
    write_rows(z["addrs"], z["rows"], dry_run=False)
    return len(z["addrs"])


def apply_step(step, e, dry_run=True):
    """Write the 16 head-rows of one step.  Returns the write plan."""
    return write_rows(step["addrs"], split_heads(e), dry_run=dry_run)


def unembed_basis(target_id, competitor_ids):
    """W_V^-1 u_t 방향들로 이루어진 기저.

    무작위 방향 대신 **실제로 다투는 로짓들**의 방향을 준다.  목표를 올리고
    경쟁 토큰을 내리는 조합이 이 부분공간 안에 정확히 들어 있으므로, SPSA 가
    1280차원을 헤매지 않고 몇 십 차원 안에서 답을 찾는다.
    반환: [m, 2560] 정규화된 행벡터들.
    """
    ids = [int(target_id)] + [int(c) for c in competitor_ids]
    U = _read_partial("lm_head.weight", ids)                      # [m, 2560]
    E = np.linalg.solve(value_proj(), U.T).T.astype(np.float32)    # [m, 2560]
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    diffs = [E[0] - E[j] for j in range(1, len(E))]   # margin 을 직접 움직이는 방향
    B = np.stack(list(E) + diffs) if diffs else E
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B.astype(np.float32)
