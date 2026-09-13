"""데모 프론트엔드 서버 (표준 라이브러리만).

  python3 web/server.py --port 8080
정적 파일(web/) 을 서빙하고, /api/chat/<before|after> 를 각 vLLM 인스턴스의
/v1/chat/completions 로 스트리밍 프록시한다.  /api/bench, /api/train 은 결과 JSON.
"""
import argparse, json, os, re, sys, threading, time, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import privy_pay  # /api/fact (402 결제 요구) · /api/privy/* — Privy 서버 지갑으로 사실을 사는 에이전트

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "scripts"))
BACKENDS = {"before": os.environ.get("BEFORE_API", "http://localhost:8000"),
            "after":  os.environ.get("AFTER_API",  "http://localhost:8001")}
MODEL = "Qwen3.8-Flash-Next"
# "학습 후" 서버가 실제로 무엇을 서빙하고 있는지. 서버가 떠 있다는 것과 패치가 테이블에
# 올라가 있다는 것은 서로 다른 사실인데, 이 둘을 뭉뚱그린 탓에 2026-09-12 에는 원본 모델이
# 22시간 동안 "학습 후" 로 표시됐다. 이제 /api/status 가 둘을 따로 답한다.
AFTER_NPZ = os.environ.get("AFTER_NPZ", os.path.join(REPO, "results", "train-all", "rows-bidir-final.npz"))
AFTER_PATCH_DIR = os.environ.get("AFTER_PATCH_DIR", os.path.join(REPO, "ple_patch_after"))
PATCH_CHECK_TTL = float(os.environ.get("PATCH_CHECK_TTL", "20"))
# OpenAI 호환 API: 모델명 → 백엔드. 학습 후(패치 적용)가 기본.
API_MODELS = {"finance-knowledge": "after", "Qwen3.8-Flash-Next": "before"}
# API 키는 저장소에 두지 않는다. 환경변수 API_KEYS(쉼표구분) 또는 web/api_keys.txt(줄단위, gitignore).
def _load_keys():
    ks = set(k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip())
    kf = os.path.join(ROOT, "api_keys.txt")
    if os.path.exists(kf):
        ks |= set(l.strip() for l in open(kf) if l.strip() and not l.startswith("#"))
    return ks
API_KEYS = _load_keys()
try:
    _KRX = json.load(open(os.path.join(REPO, "data", "krx.json"), encoding="utf-8"))
except FileNotFoundError:
    _KRX = []
_BY_CODE = {r["종목코드"]: r for r in _KRX}
_NAMES = sorted({r["회사명"]: r for r in _KRX}.values(), key=lambda r: -len(r["회사명"]))


def lookup(text):
    """질문에 들어 있는 회사명/종목코드의 정답 목록."""
    hits, used = [], set()
    for r in _NAMES:
        n = r["회사명"]
        if len(n) >= 2 and n in text and not any(n in u for u in used):
            hits.append({"name": n, "code": r["종목코드"], "market": r.get("시장구분", "")}); used.add(n)
    for c in re.findall(r"(?<![0-9A-Z])([0-9][0-9A-Z]{5})(?![0-9A-Z])", text):
        if c in _BY_CODE and c not in {h["code"] for h in hits}:
            r = _BY_CODE[c]; hits.append({"name": r["회사명"], "code": c, "market": r.get("시장구분", "")})
    return hits


_patch_cache = {"at": 0.0, "val": None}
_patch_lock = threading.Lock()


def patch_state():
    """학습 후 서버의 PLE 테이블에 패치가 실제로 올라가 있는가.

    {"applied": True|False|None, "npz": ..., "rows": ..., "detail": ...}
    `applied: None` 은 "확인하지 못했다" 이지 "적용됐다" 가 아니다 — 화면도 그렇게 구분해서
    보여줘야 한다. 훅 왕복이 있으므로 PATCH_CHECK_TTL 초 동안 캐시한다.
    """
    with _patch_lock:
        if _patch_cache["val"] is not None and time.time() - _patch_cache["at"] < PATCH_CHECK_TTL:
            return _patch_cache["val"]
    if not os.path.exists(AFTER_NPZ):
        val = {"applied": None, "npz": os.path.basename(AFTER_NPZ), "rows": None,
               "detail": f"npz 를 찾을 수 없다: {AFTER_NPZ} (AFTER_NPZ 로 지정)"}
    else:
        try:
            from patch_fingerprint import check, load
            fp = load(AFTER_NPZ)
            state, detail = check(fp, AFTER_PATCH_DIR)
            val = {"applied": {"applied": True, "stale": False}.get(state),
                   "npz": fp["npz"], "rows": fp["rows"], "detail": detail}
        except Exception as e:
            val = {"applied": None, "npz": os.path.basename(AFTER_NPZ), "rows": None,
                   "detail": f"{type(e).__name__}: {e}"}
    with _patch_lock:
        _patch_cache.update(at=time.time(), val=val)
    return val


privy_pay.LOOKUP = lookup  # 결제 후 답의 정답 판정에 같은 사전을 쓴다

def train_table():
    rows = []
    try:
        for line in open(os.path.join(REPO, "train", "train_all.log"), encoding="utf-8"):
            m = re.match(r"학습 전: 템플릿 정답 (\d+)/(\d+).*답 logprob ([-\d.]+)", line)
            if m: rows.append({"epoch": "학습 전", "loss": None, "correct": int(m[1]), "n": int(m[2]), "logprob": float(m[3])})
            m = re.match(r"에포크 (\d+): loss ([\d.]+)\s+템플릿 정답 (\d+)/(\d+).*답 logprob ([-\d.]+)\s+(\d+)s", line)
            if m: rows.append({"epoch": f"1단계 에포크 {m[1]}", "loss": float(m[2]), "correct": int(m[3]), "n": int(m[4]), "logprob": float(m[5]), "sec": int(m[6])})
        for line in open(os.path.join(REPO, "train", "train_p3.log"), encoding="utf-8"):
            m = re.match(r"에포크 (\d+): loss ([\d.]+)\s+템플릿 정답 (\d+)/(\d+).*답 logprob ([-\d.]+)\s+(\d+)s", line)
            if m: rows.append({"epoch": f"3단계(채팅 형식 추가) 에포크 {m[1]}", "loss": float(m[2]), "correct": int(m[3]), "n": int(m[4]), "logprob": float(m[5]), "sec": int(m[6])})
        rows.append({"epoch": "핀포인트 (고유 행 376개, 1스텝)", "loss": 0.081, "correct": 2761, "n": 2761, "logprob": -0.002, "sec": 60})
    except FileNotFoundError:
        pass
    return rows


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if privy_pay.handle_get(self): return
        if self.path == "/api/bench":
            p = os.path.join(REPO, "results", "verify-all.json")
            return self._json(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"error": "아직 없음"})
        if self.path == "/api/chat-eval":
            p = os.path.join(REPO, "results", "chat-eval.json")
            return self._json(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"error": "아직 없음"})
        if self.path == "/api/trace":
            p = os.path.join(REPO, "results", "trace-demo.json")
            return self._json(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"error": "아직 없음"})
        if self.path == "/api/train":
            return self._json({"epochs": train_table()})
        if self.path.startswith("/api/lookup"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            return self._json({"hits": lookup(q)})
        if self.path == "/v1/models":
            return self._json({"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "comcom"} for m in API_MODELS]})
        if self.path == "/api/status":
            st = {}
            for k, u in BACKENDS.items():
                try:
                    urllib.request.urlopen(u + "/v1/models", timeout=3); st[k] = True
                except Exception:
                    st[k] = False
            # 서버가 떠 있는 것과 학습 내용이 들어 있는 것은 다른 사실이다. 후자를 따로 답한다.
            st["patch"] = patch_state() if st.get("after") else {
                "applied": None, "npz": os.path.basename(AFTER_NPZ), "rows": None,
                "detail": "학습 후 서버가 응답하지 않는다"}
            return self._json(st)
        if self.path == "/": self.path = "/index.html"
        return super().do_GET()

    def _auth_ok(self):
        h = self.headers.get("Authorization", "")
        key = h[7:].strip() if h.startswith("Bearer ") else ""
        return key in API_KEYS if API_KEYS else False

    def do_POST(self):
        if privy_pay.handle_post(self): return
        if self.path == "/v1/chat/completions":
            return self._openai_chat()
        m = re.match(r"/api/chat/(before|after)$", self.path)
        if not m: return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        payload = {"model": MODEL, "messages": body.get("messages", []), "stream": True,
                   "temperature": body.get("temperature", 0.0), "max_tokens": body.get("max_tokens", 512),
                   "chat_template_kwargs": {"enable_thinking": bool(body.get("thinking", False))}}
        req = urllib.request.Request(BACKENDS[m[1]] + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            up = urllib.request.urlopen(req, timeout=600)
        except Exception as e:
            return self._json({"error": f"{m[1]} 서버 응답 없음: {e}"}, 502)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        try:
            for line in up:
                self.wfile.write(line); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _openai_chat(self):
        if not self._auth_ok():
            return self._json({"error": {"message": "invalid or missing API key. request one at kimminhyun@comcom.ai",
                                          "type": "invalid_request_error", "code": "invalid_api_key"}}, 401)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": {"message": "invalid JSON body", "type": "invalid_request_error"}}, 400)
        side = API_MODELS.get(body.get("model", "finance-knowledge"))
        if side is None:
            return self._json({"error": {"message": f"unknown model. available: {list(API_MODELS)}",
                                          "type": "invalid_request_error"}}, 404)
        stream = bool(body.get("stream", False))
        payload = {"model": MODEL, "messages": body.get("messages", []), "stream": stream,
                   "temperature": body.get("temperature", 0.0), "max_tokens": body.get("max_tokens", 512),
                   "chat_template_kwargs": {"enable_thinking": bool(body.get("enable_thinking", False))}}
        req = urllib.request.Request(BACKENDS[side] + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            up = urllib.request.urlopen(req, timeout=600)
        except Exception as e:
            return self._json({"error": {"message": f"upstream error: {e}", "type": "server_error"}}, 502)
        if stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                for line in up:
                    self.wfile.write(line); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            data = up.read()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8080); a = ap.parse_args()
    print(f"demo: http://0.0.0.0:{a.port}  backends {BACKENDS}")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
