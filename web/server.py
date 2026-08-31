"""데모 프론트엔드 서버 (표준 라이브러리만).

  python3 web/server.py --port 8080
정적 파일(web/) 을 서빙하고, /api/chat/<before|after> 를 각 vLLM 인스턴스의
/v1/chat/completions 로 스트리밍 프록시한다.  /api/bench, /api/train 은 결과 JSON.
"""
import argparse, json, os, re, sys, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
BACKENDS = {"before": os.environ.get("BEFORE_API", "http://localhost:8000"),
            "after":  os.environ.get("AFTER_API",  "http://localhost:8001")}
MODEL = "Qwen3.8-Flash-Next"
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
        if self.path == "/api/status":
            st = {}
            for k, u in BACKENDS.items():
                try:
                    urllib.request.urlopen(u + "/v1/models", timeout=3); st[k] = True
                except Exception:
                    st[k] = False
            return self._json(st)
        if self.path == "/": self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8080); a = ap.parse_args()
    print(f"demo: http://0.0.0.0:{a.port}  backends {BACKENDS}")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
