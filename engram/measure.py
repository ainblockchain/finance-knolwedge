"""서빙 중인 모델에서 목표 답의 확률을 잰다.

편집의 성패는 "trigger 다음에 answer 가 나오는가"이므로, 생성이 아니라
**answer 토큰들의 logprob 합**을 직접 재는 것이 정확하고 빠르다.
vLLM completions 의 echo+logprobs 를 쓴다 (max_tokens=0).
"""
import json, os, time, urllib.request

BASE = os.environ.get("ENGRAM_API", "http://localhost:8000")
MODEL = os.environ.get("ENGRAM_MODEL", "Qwen3.8-Flash-Next")


def _post(path, body, timeout=300, retries=40):
    """엔진이 죽으면 500 을 반환하고 컨테이너가 자동 재시작된다 (약 5분).

    실측: PLE 테이블을 추론 도중 병렬로 쓰면 MTP 드래프트/검증 불일치로
    GPU 워커가 응답을 멈추고 EngineCore 가 타임아웃으로 죽는다.
    긴 실험이 그것 때문에 통째로 날아가지 않도록 복구를 기다렸다 재시도한다.
    """
    data = json.dumps(body).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + path, data,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(15)   # 재기동 대기


def answer_logprob(trigger: str, answer: str, timeout=300):
    """log p(answer | trigger).  answer 토큰들의 logprob 합."""
    d = _post("/v1/completions", {
        "model": MODEL, "prompt": trigger + answer,
        "max_tokens": 0, "echo": True, "logprobs": 0, "temperature": 0,
    }, timeout)
    lp = d["choices"][0]["logprobs"]
    n_ans = len(_post("/v1/completions", {  # answer 토큰 수
        "model": MODEL, "prompt": answer, "max_tokens": 0,
        "echo": True, "logprobs": 0, "temperature": 0,
    }, timeout)["choices"][0]["logprobs"]["tokens"])
    vals = [v for v in lp["token_logprobs"][-n_ans:] if v is not None]
    return sum(vals)


class Scorer:
    """answer 토큰 수를 한 번만 재고 재사용하는 버전 (최적화 루프용)."""

    def __init__(self, trigger, answer):
        from .core import encode
        self.trigger, self.answer = trigger, answer
        self.answer_tokens = encode(answer)
        self.n_ans = len(self.answer_tokens)
        if encode(trigger + answer) != encode(trigger) + self.answer_tokens:
            raise ValueError("토크나이저 경계에서 재분절됨 — plan() 과 동일 제약")

    def __call__(self):
        d = _post("/v1/completions", {
            "model": MODEL, "prompt": self.trigger + self.answer,
            "max_tokens": 0, "echo": True, "logprobs": 0, "temperature": 0,
        })
        tl = d["choices"][0]["logprobs"]["token_logprobs"][-self.n_ans:]
        return sum(v for v in tl if v is not None)


def generate(prompt, max_tokens=12, logprobs=1):
    d = _post("/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "logprobs": logprobs,
    })
    return d["choices"][0]["text"]


class AnswerScorer:
    """전체 프롬프트 한 번으로 answer 토큰별 logprob 벡터를 얻는다.

    주의: 이 모델은 짧은 프롬프트와 긴 프롬프트가 **같은 토큰에 다른 logprob**
    을 준다 (GDN 순환 상태 + prefix caching + chunked prefill).  따라서 단계별로
    프롬프트를 잘라 재면 안 되고, 반드시 생성 시점과 같은 전체 프롬프트로
    한 번에 재서 분해해야 한다.  API 호출도 1회로 줄어든다.
    """

    def __init__(self, trigger, answer):
        from .core import encode
        self.trigger, self.answer = trigger, answer
        self.answer_tokens = encode(answer)
        self.n_ans = len(self.answer_tokens)
        if encode(trigger + answer) != encode(trigger) + self.answer_tokens:
            raise ValueError("토크나이저 경계에서 재분절됨 — plan() 과 동일 제약")
        self.prompt = trigger + answer

    def vector(self):
        """[n_ans] answer 토큰별 logprob."""
        d = _post("/v1/completions", {
            "model": MODEL, "prompt": self.prompt,
            "max_tokens": 0, "echo": True, "logprobs": 0, "temperature": 0,
        })
        tl = d["choices"][0]["logprobs"]["token_logprobs"][-self.n_ans:]
        return [0.0 if v is None else v for v in tl]

    def __call__(self):
        return sum(self.vector())

    def margins(self):
        """[n_ans] 각 answer 토큰의 margin = logprob(목표) - max logprob(그 외).

        greedy 로 정답이 나오려면 **모든 margin 이 양수**여야 한다.  logprob 합이
        아무리 좋아져도 margin 이 음수인 자리가 하나 있으면 생성은 틀린다.
        """
        d = _post("/v1/completions", {
            "model": MODEL, "prompt": self.prompt,
            "max_tokens": 0, "echo": True, "logprobs": 5, "temperature": 0,
        })
        lp = d["choices"][0]["logprobs"]
        out = []
        for tok, tl in list(zip(lp["tokens"], lp["top_logprobs"]))[-self.n_ans:]:
            if not tl:
                out.append(0.0); continue
            mine = tl.get(tok)
            other = max((v for k, v in tl.items() if k != tok), default=None)
            out.append(0.0 if mine is None or other is None else mine - other)
        return out

    def competitors(self, i, k=4):
        """step i 에서 목표 토큰과 다투는 상위 토큰들의 문자열."""
        d = _post("/v1/completions", {
            "model": MODEL, "prompt": self.prompt,
            "max_tokens": 0, "echo": True, "logprobs": 5, "temperature": 0,
        })
        lp = d["choices"][0]["logprobs"]
        pairs = list(zip(lp["tokens"], lp["top_logprobs"]))[-self.n_ans:]
        tok, tl = pairs[i]
        if not tl:
            return []
        others = sorted(((v, kk) for kk, v in tl.items() if kk != tok), reverse=True)
        return [kk for _, kk in others[:k]]

    def margin_objective(self, i, cap=1.0):
        """step i 부터의 margin 을 cap 까지만 올리는 목적함수.

        한 자리를 과하게 밀어붙이는 대신 모든 자리를 양수로 만드는 쪽으로
        압력을 준다.  cap 을 넘는 여유분은 보상하지 않는다.
        """
        def f():
            m = self.margins()[i:]
            return sum(min(x, cap) for x in m)
        return f

    def component(self, i):
        """step i 토큰만을 목적함수로 쓰는 호출자."""
        return lambda: self.vector()[i]

    def tail(self, i):
        """step i 이후 전부를 목적함수로."""
        return lambda: sum(self.vector()[i:])


def warmup():
    """콜드 요청 이상치를 피하기 위한 버리는 호출."""
    try:
        _post("/v1/completions", {"model": MODEL, "prompt": "warm",
                                  "max_tokens": 1, "temperature": 0})
    except Exception:
        pass


class GenScorer:
    """**생성 경로에서** 직접 재는 목적함수.

    AnswerScorer.margins() 는 정답 prefix 를 강제한 상태(teacher forcing)로 잰다.
    이 모델은 프롬프트 구성에 따라 같은 토큰의 logprob 이 최대 1.7 nats 달라지므로
    (GDN 순환 상태 + prefix caching + chunked prefill), TF margin 이 전부 양수인데
    생성은 틀리는 일이 실제로 일어난다 (레이언스: TF 7/7 양수, 생성 오답).

    이 클래스는 greedy 생성을 그대로 돌려 각 위치의 top_logprobs 를 읽는다.
    생성이 갈라지면 그 자리 margin 이 음수이므로, 목적함수를 올리는 것이
    곧 갈림을 없애는 것이 된다 — 앞에서부터 차례로 고쳐지는 커리큘럼이 된다.
    """

    def __init__(self, trigger, answer, cap=1.0):
        from .core import encode, tokenizer
        self.trigger = trigger
        self.answer_tokens = encode(answer)
        self.n_ans = len(self.answer_tokens)
        self.targets = [tokenizer().decode([t]) for t in self.answer_tokens]
        self.cap = cap

    def _raw(self):
        d = _post("/v1/completions", {
            "model": MODEL, "prompt": self.trigger,
            "max_tokens": self.n_ans, "temperature": 0, "logprobs": 5,
        })
        c = d["choices"][0]
        return c["text"], c["logprobs"]["top_logprobs"]

    def margins(self):
        """[n_ans] 생성 경로에서의 자리별 margin (목표 − 그 자리 2등).

        목표가 top-5 밖이면 (top1 − 5등) 만큼 아래에 있다고 보수적으로 친다.
        """
        _, tops = self._raw()
        out = []
        for i in range(self.n_ans):
            tl = tops[i] if i < len(tops) else None
            if not tl:
                out.append(-10.0); continue
            tgt = self.targets[i]
            mine = tl.get(tgt)
            other = max((v for k, v in tl.items() if k != tgt), default=None)
            if mine is None:
                lo = min(tl.values()); hi = max(tl.values())
                out.append(lo - hi)          # top-5 밖 — 보수적 하한
            elif other is None:
                out.append(self.cap)
            else:
                out.append(mine - other)
        return out

    def matched(self):
        """생성이 목표와 일치하는 접두 길이."""
        m = self.margins()
        n = 0
        for x in m:
            if x <= 0:
                break
            n += 1
        return n

    def objective(self, i=0):
        """step i 부터의 capped margin 합.  갈림 이후 자리도 세므로 잡음이 섞인다."""
        def f():
            return sum(min(x, self.cap) for x in self.margins()[i:])
        return f

    def staged_objective(self, cap=1.0):
        """**갈림 지점만** 겨냥하는 단계형 목적함수.

        생성이 갈라진 뒤의 자리들은 틀린 prefix 위에서 계산되므로 그 margin 은
        의미가 없다 — 그것까지 합에 넣으면 최적화 용량을 잡음에 쓴다.

            점수 = (일치한 접두 길이) + min(첫 실패 자리의 margin, cap)

        접두를 한 칸 늘리는 것이 +1 이므로 항상 그쪽이 우선되고, 같은 단계
        안에서는 첫 실패 자리의 margin 을 끌어올린다.
        """
        def f():
            m = self.margins()
            k = 0
            while k < len(m) and m[k] > 0:
                k += 1
            if k >= len(m):
                return float(len(m)) + cap      # 전부 통과
            return float(k) + min(m[k], cap)
        return f

    def hit(self):
        text, _ = self._raw()
        from .core import tokenizer
        return text.startswith(tokenizer().decode(self.answer_tokens))
