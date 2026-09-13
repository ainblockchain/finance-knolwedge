"""Privy 서버 지갑 + 정책 엔진으로 "사실을 사는 에이전트" 결제 루프 (표준 라이브러리만).

  에이전트 ── GET /api/fact?q=… ─────────────────────────────► 402 {price_wei, payTo, chainId, nonce}
  에이전트 ── POST api.privy.io/v1/wallets/{id}/rpc eth_sendTransaction ─► tx hash   (정책이 ALLOW 한 경우만)
  에이전트 ── GET /api/fact?q=…  X-Payment-Receipt: <hash> ──► 체인에서 검증 후 학습된 모델의 종목코드 답

왜 Privy 인가: 이 데모의 원본 모델은 종목코드의 82% 를 환각한다. 그런 모델 위에 세운 에이전트는 예산도
환각한다. Privy 정책(to == FACT_PROVIDER_ADDRESS, value <= FACT_PRICE_WEI, 그 외 전부 DENY)은 TEE 안에서
강제되므로, 에이전트 프로세스가 오작동하거나 프롬프트 인젝션을 당해도 트레저리를 털 수 없다. 이 파일은
그 정책이 실제로 거부하는 장면과 정상 결제가 통과하는 장면을 둘 다 만들어 보이는 데 필요한 것만 담는다.

설정: 환경변수, 또는 아래 순서로 처음 발견되는 파일 (KEY=VALUE 한 줄씩).
  1) $PRIVY_ENV_FILE           2) web/privy.env (이 파일 옆, gitignore)
  3) ~/.privy/finance-demo.env  ← 기본. 체크아웃·워크트리와 무관하게 한 곳에 두면 어떤 배포에서도 같은 값을 읽는다.
  PRIVY_APP_ID, PRIVY_APP_SECRET        dashboard.privy.io → App settings → Basics / API keys
  PRIVY_WALLET_ID, PRIVY_WALLET_ADDRESS `python3 web/privy_pay.py setup` 이 만들어 준다
  PRIVY_POLICY_ID                       `setup` 이 만들어 준다
  FACT_PROVIDER_ADDRESS                 사실 공급자(수취인) 주소 — 정책의 to. 이 주소만 받을 수 있다
  FACT_PRICE_WEI                        건당 가격. 기본 10000000000000 (= 0.00001 ETH)
  CHAIN_CAIP2                           기본 eip155:84532 (Base Sepolia). Hedera 테스트넷은 eip155:296
  RPC_URL                               체인 JSON-RPC. 기본 https://sepolia.base.org
PRIVY_APP_ID 가 없으면 **mock 모드**: 정책 판정을 이 프로세스가 흉내 내고 화면·응답에 MOCK 이라고 적는다.
mock 은 화면을 만들고 흐름을 점검하는 용도이며 제출 증거가 될 수 없다.

CLI:
  python3 web/privy_pay.py setup --provider 0x…  [--write]   정책 → 지갑 순서로 만들고 env 줄을 출력
  python3 web/privy_pay.py spike --caip2 eip155:296          Privy 지갑이 그 체인으로 tx 를 내보내는지 확인
  python3 web/privy_pay.py status
  python3 web/privy_pay.py agent "픽셀플러스 종목코드 알려줘. 숫자만."
  python3 web/privy_pay.py try --to 0x… --value-wei 1000000000000000
"""
import argparse, base64, hashlib, json, os, re, secrets, sys, threading, time, urllib.error, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
PRIVY_API = "https://api.privy.io"


CANONICAL_ENV = os.path.expanduser("~/.privy/finance-demo.env")
ENV_CANDIDATES = [os.environ.get("PRIVY_ENV_FILE", ""), os.path.join(ROOT, "privy.env"), CANONICAL_ENV]
ENV_FILE = None  # 실제로 읽은 파일. status 와 시작 로그에 나오므로 "어느 env 를 읽었나" 를 추측하지 않아도 된다.


def _load_env_file():
    global ENV_FILE
    for p in ENV_CANDIDATES:
        if p and os.path.exists(p):
            ENV_FILE = p
            break
    if not ENV_FILE:
        return
    for line in open(ENV_FILE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


_load_env_file()

CFG = {
    "app_id": os.environ.get("PRIVY_APP_ID", ""),
    "app_secret": os.environ.get("PRIVY_APP_SECRET", "") or os.environ.get("PRIVY_API_KEY", ""),
    "wallet_id": os.environ.get("PRIVY_WALLET_ID", ""),
    "wallet_address": os.environ.get("PRIVY_WALLET_ADDRESS", "").lower(),
    "policy_id": os.environ.get("PRIVY_POLICY_ID", ""),
    "provider": os.environ.get("FACT_PROVIDER_ADDRESS", "").lower(),
    "price_wei": int(os.environ.get("FACT_PRICE_WEI", "10000000000000")),
    "caip2": os.environ.get("CHAIN_CAIP2", "eip155:84532"),
    "rpc_url": os.environ.get("RPC_URL", "https://sepolia.base.org"),
    "before_api": os.environ.get("BEFORE_API", "http://localhost:8000"),
    "after_api": os.environ.get("AFTER_API", "http://localhost:8001"),
}
MODE = "privy" if CFG["app_id"] and CFG["app_secret"] else "mock"
print(f"privy: mode={MODE} env={ENV_FILE or '(없음)'}" + ("" if MODE == "privy" else
      f"  ← MOCK. PRIVY_APP_ID/PRIVY_APP_SECRET 을 {CANONICAL_ENV} 에 넣으면 LIVE"), file=sys.stderr, flush=True)
CHAIN_ID = int(CFG["caip2"].split(":")[1])
EXPLORER = {84532: "https://sepolia.basescan.org/tx/", 8453: "https://basescan.org/tx/",
            11155111: "https://sepolia.etherscan.io/tx/", 296: "https://hashscan.io/testnet/transaction/"}
MODEL = "Qwen3.8-Flash-Next"
NONCE_TTL = 600  # 초. 402 로 내준 nonce 가 유효한 시간
CODE_RE = re.compile(r"(?<![0-9A-Z])([0-9][0-9A-Z]{5})(?![0-9A-Z])")
# mock 모드에서만 쓰는 가짜 주소. 실제 체인 어디에도 없는 값이라는 것이 눈에 보이게 한다.
MOCK_PROVIDER = "0x00000000000000000000000000000000000fac70"
MOCK_WALLET = "0x000000000000000000000000000000000a9e4700"
if MODE == "mock":
    CFG["provider"] = CFG["provider"] or MOCK_PROVIDER
    CFG["wallet_address"] = CFG["wallet_address"] or MOCK_WALLET
    CFG["wallet_id"] = CFG["wallet_id"] or "mock-wallet"
    CFG["policy_id"] = CFG["policy_id"] or "mock-policy"

LOOKUP = None  # server.py 가 자신의 lookup(text) 을 꽂아 준다. 없으면 krx.json 을 직접 읽는다.


def lookup(text):
    global LOOKUP
    if LOOKUP is None:
        try:
            krx = json.load(open(os.path.join(REPO, "data", "krx.json"), encoding="utf-8"))
        except FileNotFoundError:
            krx = []
        by_code = {r["종목코드"]: r for r in krx}
        names = sorted({r["회사명"]: r for r in krx}.values(), key=lambda r: -len(r["회사명"]))

        def _lk(t):
            hits, used = [], set()
            for r in names:
                n = r["회사명"]
                if len(n) >= 2 and n in t and not any(n in u for u in used):
                    hits.append({"name": n, "code": r["종목코드"], "market": r.get("시장구분", "")}); used.add(n)
            for c in CODE_RE.findall(t):
                if c in by_code and c not in {h["code"] for h in hits}:
                    r = by_code[c]; hits.append({"name": r["회사명"], "code": c, "market": r.get("시장구분", "")})
            return hits
        LOOKUP = _lk
    return LOOKUP(text)


# ───────────────────────────── 상태 (nonce · 영수증 · 이벤트) ─────────────────────────────
STATE_PATH = os.environ.get("PRIVY_STATE", os.path.join(ROOT, "privy_state.json"))
_lock = threading.Lock()
try:
    _state = json.load(open(STATE_PATH, encoding="utf-8"))
except Exception:
    _state = {}
_state.setdefault("nonces", {}); _state.setdefault("receipts", []); _state.setdefault("events", [])
_state.setdefault("denials", [])


def _save():
    tmp = STATE_PATH + ".tmp"
    json.dump(_state, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def event(kind, **kw):
    e = {"t": time.time(), "kind": kind, "mode": MODE, **kw}
    with _lock:
        _state["events"].append(e); _state["events"] = _state["events"][-200:]; _save()
    return e


def issue_nonce(q):
    n = secrets.token_hex(16)
    with _lock:
        now = time.time()
        for k in [k for k, v in _state["nonces"].items() if now - v["issued"] > NONCE_TTL and not v.get("used")]:
            del _state["nonces"][k]
        _state["nonces"][n] = {"q": q, "issued": now, "used": None}; _save()
    return n


# ───────────────────────────── Privy REST ─────────────────────────────
def privy(method, path, body=None):
    """(http status, json). Basic 인증은 app_id:app_secret 이고 privy-app-id 헤더도 함께 보낸다."""
    hdr = {"Authorization": "Basic " + base64.b64encode(f"{CFG['app_id']}:{CFG['app_secret']}".encode()).decode(),
           "privy-app-id": CFG["app_id"], "Content-Type": "application/json",
           # Cloudflare 가 urllib 기본 UA 를 막는다 (403, error code 1010). 우리가 누구인지 밝힌다.
           "User-Agent": "finance-demo-agent/1.0 (+https://finance-demo.ainetwork.ai)"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(PRIVY_API + path, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": raw.decode(errors="replace")}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def policy_body(provider, price_wei):
    """허용 룰 하나. Privy 정책은 명시적으로 허용하지 않은 것은 전부 거부한다."""
    return {"version": "1.0", "name": "Fact-purchase budget guardrail", "chain_type": "ethereum",
            "rules": [{"name": "Allow micropayments to fact provider only", "method": "eth_sendTransaction",
                       "action": "ALLOW", "conditions": [
                           {"field_source": "ethereum_transaction", "field": "to", "operator": "eq", "value": provider},
                           {"field_source": "ethereum_transaction", "field": "value", "operator": "lte", "value": str(price_wei)}]}]}


def create_policy(provider, price_wei):
    return privy("POST", "/v1/policies", policy_body(provider, price_wei))


def create_wallet(policy_ids):
    return privy("POST", "/v1/wallets", {"chain_type": "ethereum", "policy_ids": list(policy_ids)})


def _mock_policy(to, value_wei):
    """mock 모드의 정책 판정. 실제 Privy 와 같은 규칙(to eq, value lte, 그 외 DENY)을 이 프로세스가 흉내 낸다."""
    if (to or "").lower() != CFG["provider"]:
        return False, f"MOCK policy: DENY — rule 'Allow micropayments to fact provider only' requires to == {CFG['provider']}, got {to}"
    if int(value_wei) > CFG["price_wei"]:
        return False, f"MOCK policy: DENY — value {value_wei} wei exceeds cap {CFG['price_wei']} wei"
    return True, "MOCK policy: ALLOW"


def send_tx(to, value_wei, data="0x", caip2=None):
    """Privy 지갑으로 eth_sendTransaction. 반환 {allowed, hash, status, request, response, ms, mode}.
    정책이 거부하면 allowed=False 이고 response 에 Privy 의 거부 응답 원문이 들어 있다."""
    caip2 = caip2 or CFG["caip2"]
    chain_id = int(caip2.split(":")[1])
    req = {"method": "eth_sendTransaction", "caip2": caip2,
           "params": {"transaction": {"to": to, "value": hex(int(value_wei)), "data": data, "chain_id": chain_id}}}
    t0 = time.time()
    if MODE == "mock":
        ok, msg = _mock_policy(to, value_wei)
        h = "0x" + hashlib.sha256(f"{to}{value_wei}{data}{time.time()}".encode()).hexdigest() if ok else None
        resp = {"method": "eth_sendTransaction", "data": {"hash": h, "caip2": caip2}} if ok else {"error": msg, "code": 403}
        out = {"allowed": ok, "hash": h, "status": 200 if ok else 403, "request": req, "response": resp,
               "ms": int((time.time() - t0) * 1000), "mode": MODE}
    else:
        st, resp = privy("POST", f"/v1/wallets/{CFG['wallet_id']}/rpc", req)
        h = (resp.get("data") or {}).get("hash") if st == 200 else None
        out = {"allowed": st == 200 and bool(h), "hash": h, "status": st, "request": req, "response": resp,
               "ms": int((time.time() - t0) * 1000), "mode": MODE}
    # 거부에는 두 종류가 있고 화면은 이를 구분해야 한다: 정책 엔진이 서명 자체를 거부한 것(code policy_violation)과,
    # 정책은 통과했지만 브로드캐스트·시뮬레이션에서 실패한 것(잔액 부족 등). 후자는 가드레일의 증거가 아니다.
    code = (out["response"] or {}).get("code", "")
    out["denied_by_policy"] = (not out["allowed"]) and (code == "policy_violation" or (MODE == "mock" and code == 403))
    kind = "tx_allowed" if out["allowed"] else ("tx_denied" if out["denied_by_policy"] else "tx_failed")
    event(kind, to=to, value_wei=int(value_wei), hash=out["hash"], status=out["status"],
          detail=json.dumps(out["response"], ensure_ascii=False)[:400])
    if out["denied_by_policy"]:
        with _lock:
            _state["denials"].append({"t": time.time(), "to": to, "value_wei": int(value_wei), "method": "eth_sendTransaction",
                                      "response": out["response"]})
            _state["denials"] = _state["denials"][-50:]; _save()
    return out


def try_method(method, params):
    """정책이 허용하지 않은 메서드(예: personal_sign)를 같은 지갑으로 호출한다. 정책은 eth_sendTransaction 하나만 허용하므로
    거부되어야 하고, 브로드캐스트가 없어 잔액과 무관하게 정책 엔진의 판정만 남는다."""
    req = {"method": method, "params": params}
    t0 = time.time()
    if MODE == "mock":
        st, resp = 400, {"error": "MOCK policy: RPC request denied due to policy violation", "code": "policy_violation"}
    else:
        st, resp = privy("POST", f"/v1/wallets/{CFG['wallet_id']}/rpc", req)
    out = {"allowed": st == 200, "hash": None, "status": st, "request": req, "response": resp, "ms": int((time.time() - t0) * 1000),
           "mode": MODE, "denied_by_policy": st != 200 and (resp or {}).get("code") == "policy_violation"}
    event("tx_allowed" if out["allowed"] else ("tx_denied" if out["denied_by_policy"] else "tx_failed"), method=method,
          status=st, detail=json.dumps(resp, ensure_ascii=False)[:400])
    if out["denied_by_policy"]:
        with _lock:
            _state["denials"].append({"t": time.time(), "to": None, "value_wei": 0, "method": method, "response": resp})
            _state["denials"] = _state["denials"][-50:]; _save()
    return out


# ───────────────────────────── 체인 JSON-RPC ─────────────────────────────
def rpc(method, params):
    req = urllib.request.Request(CFG["rpc_url"], data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "finance-demo-agent/1.0"})  # 공개 RPC 도 urllib UA 를 403 으로 막는다
    with urllib.request.urlopen(req, timeout=20) as r:
        j = json.loads(r.read())
    if "error" in j:
        raise RuntimeError(f"rpc {method}: {j['error']}")
    return j["result"]


def balance_wei(addr):
    try:
        return int(rpc("eth_getBalance", [addr, "latest"]), 16)
    except Exception:
        return None


def wait_receipt(h, timeout=90):
    """영수증이 채굴될 때까지 기다린다. mock 은 즉시 성공을 흉내 낸다."""
    if MODE == "mock":
        return {"status": "0x1", "blockNumber": hex(int(time.time())), "mock": True}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = rpc("eth_getTransactionReceipt", [h])
        except Exception:
            r = None
        if r:
            return r
        time.sleep(2)
    return None


def verify_receipt(h):
    """tx hash 하나가 '이 공급자에게, 가격 이상을, 우리가 내준 nonce 를 data 에 담아, 성공적으로' 보낸 결제인지.
    반환 (ok, nonce, detail). 같은 hash 는 한 번만 인정한다."""
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}", h or ""):
        return False, None, "영수증 형식이 아니다 (0x + 64 hex)"
    with _lock:
        if any(r["hash"] == h for r in _state["receipts"]):
            return False, None, "이미 사용된 영수증이다 (한 결제 = 한 조회)"
    if MODE == "mock":
        with _lock:
            for n, v in _state["nonces"].items():
                if v.get("mock_hash") == h and not v.get("used"):
                    return True, n, "MOCK: 이 프로세스가 발행한 가짜 영수증과 일치"
        return False, None, "MOCK: 이 프로세스가 발행한 영수증이 아니다"
    try:
        tx = rpc("eth_getTransactionByHash", [h])
    except Exception as e:
        return False, None, f"체인 조회 실패: {e}"
    if not tx:
        return False, None, "체인에 그런 tx 가 없다"
    if (tx.get("to") or "").lower() != CFG["provider"]:
        return False, None, f"수취인이 사실 공급자가 아니다: {tx.get('to')}"
    if int(tx.get("value", "0x0"), 16) < CFG["price_wei"]:
        return False, None, f"금액 부족: {int(tx.get('value', '0x0'), 16)} < {CFG['price_wei']} wei"
    if CFG["wallet_address"] and (tx.get("from") or "").lower() != CFG["wallet_address"]:
        return False, None, f"보낸 지갑이 에이전트의 Privy 지갑이 아니다: {tx.get('from')}"
    nonce = (tx.get("input") or "0x")[2:].lower()
    with _lock:
        rec = _state["nonces"].get(nonce)
    if not rec:
        return False, None, "tx data 에 이 서버가 발행한 nonce 가 없다 (영수증 재사용·위조 방지)"
    if rec.get("used"):
        return False, None, "이 nonce 는 이미 사용됐다"
    rcpt = wait_receipt(h, timeout=60)
    if not rcpt:
        return False, None, "영수증이 아직 채굴되지 않았다 (잠시 후 다시)"
    if rcpt.get("status") != "0x1":
        return False, None, f"tx 실패 (status {rcpt.get('status')})"
    return True, nonce, f"검증 완료: to·value·from·nonce·status 모두 일치, block {int(rcpt['blockNumber'], 16)}"


# ───────────────────────────── 모델에 묻기 ─────────────────────────────
def ask_model(side, q, max_tokens=48):
    url = CFG["before_api" if side == "before" else "after_api"] + "/v1/chat/completions"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": q}], "stream": False, "temperature": 0.0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        m = json.load(r)["choices"][0]["message"]
    txt = (m.get("content") or "").strip()
    # 답은 한 줄이다. 그 뒤에 이어지는 것(툴콜 파서 잔여물 등)은 답이 아니므로 첫 줄만 남긴다.
    return next((l.strip() for l in txt.splitlines() if l.strip()), txt)


def grade(answer, truth):
    """답에서 종목코드를 뽑아 정답과 비교. truth 가 없으면 None."""
    m = CODE_RE.search(answer or "")
    code = m.group(1) if m else None
    if not truth:
        return code, None
    return code, code in {t["code"] for t in truth}


# ───────────────────────────── /api/fact ─────────────────────────────
def fact_response(q, receipt):
    """(http status, body). 영수증이 없거나 틀리면 402 와 결제 조건, 맞으면 학습된 모델의 답."""
    if not q:
        return 400, {"error": "q 가 비었다"}
    if not receipt:
        n = issue_nonce(q)
        event("fact_402", q=q, nonce=n)
        return 402, {"error": "payment required", "price_wei": CFG["price_wei"], "price_eth": CFG["price_wei"] / 1e18,
                     "payTo": CFG["provider"], "chainId": CHAIN_ID, "caip2": CFG["caip2"], "nonce": n,
                     "expires_in": NONCE_TTL, "mode": MODE,
                     "how": "eth_sendTransaction {to: payTo, value: price_wei, data: '0x'+nonce} 를 보내고, "
                            "같은 요청에 X-Payment-Receipt: <tx hash> 를 붙여 다시 부르면 된다."}
    ok, nonce, detail = verify_receipt(receipt)
    if not ok:
        event("fact_rejected", q=q, receipt=receipt, detail=detail)
        return 402, {"error": "payment not accepted", "detail": detail, "receipt": receipt, "mode": MODE}
    try:
        answer = ask_model("after", q)
    except Exception as e:
        return 502, {"error": f"학습 후 모델 응답 없음: {e}"}
    truth = lookup(q)
    code, correct = grade(answer, truth)
    with _lock:
        _state["nonces"][nonce]["used"] = receipt
        _state["receipts"].append({"hash": receipt, "q": q, "code": code, "correct": correct, "t": time.time(), "mode": MODE,
                                   "value_wei": CFG["price_wei"]})
        _state["receipts"] = _state["receipts"][-200:]; _save()
    event("fact_paid", q=q, receipt=receipt, code=code, correct=correct)
    return 200, {"answer": answer, "code": code, "correct": correct, "truth": truth, "model": "finance-knowledge (학습 후)",
                 "receipt": {"hash": receipt, "explorer": EXPLORER.get(CHAIN_ID, "") + receipt if MODE == "privy" else None,
                             "verified": detail}, "mode": MODE}


# ───────────────────────────── 에이전트 ─────────────────────────────
def run_agent(q, fact_url=None):
    """사실을 사는 에이전트 한 바퀴. 402 → Privy 결제 → 영수증 → 답. 각 단계를 steps 에 남겨 화면이 그대로 보여준다.
    fact_url 이 없으면 서버 안에서 fact_response 를 직접 부른다(같은 코드 경로)."""
    steps, t0 = [], time.time()

    def step(name, **kw):
        s = {"step": name, "t": round(time.time() - t0, 3), **kw}; steps.append(s); return s

    truth = lookup(q)
    try:
        free = ask_model("before", q)
        code, correct = grade(free, truth)
        step("free_answer", side="before", answer=free, code=code, correct=correct, truth=truth,
             note="원본 모델 — 결제 없이 누구나 받을 수 있는 답. 이 모델은 종목코드의 82% 를 환각한다.")
    except Exception as e:
        step("free_answer", side="before", error=str(e), truth=truth)

    st, body = fact_response(q, None)
    step("fact_request", status=st, body=body, note="영수증 없이 물었다 → 402 Payment Required 와 결제 조건")
    if st != 402:
        return {"steps": steps, "mode": MODE, "ok": False}

    # 에이전트의 자기 예산 확인 — 하지만 이건 애플리케이션 코드라 오작동할 수 있다. 진짜 상한은 다음 단계의 정책이다.
    if body["price_wei"] > CFG["price_wei"]:
        step("agent_budget", ok=False, note=f"에이전트 자체 예산 초과: {body['price_wei']} > {CFG['price_wei']}")
        return {"steps": steps, "mode": MODE, "ok": False}

    tx = send_tx(body["payTo"], body["price_wei"], data="0x" + body["nonce"], caip2=body["caip2"])
    step("privy_pay", **tx, note="Privy 서버 지갑으로 eth_sendTransaction. 정책(to eq 공급자, value lte 상한)이 TEE 안에서 평가된다.")
    if not tx["allowed"]:
        return {"steps": steps, "mode": MODE, "ok": False}
    if MODE == "mock":
        with _lock:
            _state["nonces"][body["nonce"]]["mock_hash"] = tx["hash"]; _save()

    rcpt = wait_receipt(tx["hash"])
    step("wait_receipt", ok=bool(rcpt) and rcpt.get("status") == "0x1",
         block=int(rcpt["blockNumber"], 16) if rcpt and rcpt.get("blockNumber") else None,
         explorer=(EXPLORER.get(CHAIN_ID, "") + tx["hash"]) if MODE == "privy" else None,
         note="체인에서 영수증이 채굴되길 기다린다")

    st, body2 = fact_response(q, tx["hash"])
    step("fact_request", status=st, body=body2, receipt=tx["hash"],
         note="X-Payment-Receipt 를 붙여 다시 물었다 → 서버가 체인에서 to·value·nonce·status 를 검증한 뒤 학습된 모델이 답한다")
    return {"steps": steps, "mode": MODE, "ok": st == 200, "elapsed": round(time.time() - t0, 2)}


def status():
    bal = balance_wei(CFG["wallet_address"]) if MODE == "privy" and CFG["wallet_address"] else None
    with _lock:
        receipts = list(_state["receipts"][-20:]); denials = list(_state["denials"][-20:]); events = list(_state["events"][-40:])
    return {"mode": MODE, "env_file": ENV_FILE, "wallet_id": CFG["wallet_id"], "wallet_address": CFG["wallet_address"], "policy_id": CFG["policy_id"],
            "provider": CFG["provider"], "price_wei": CFG["price_wei"], "price_eth": CFG["price_wei"] / 1e18,
            "caip2": CFG["caip2"], "chain_id": CHAIN_ID, "rpc_url": CFG["rpc_url"], "explorer": EXPLORER.get(CHAIN_ID, ""),
            "balance_wei": bal, "balance_eth": (bal / 1e18) if bal is not None else None,
            "policy": policy_body(CFG["provider"], CFG["price_wei"]),
            "receipts": receipts, "denials": denials, "events": events,
            "checklist": {"wallet": bool(CFG["wallet_id"]) and MODE == "privy",
                          "policy_attached": bool(CFG["policy_id"]) and MODE == "privy",
                          "paid_flow": any(r.get("mode") == "privy" for r in receipts),
                          "denial_observed": any(d for d in denials) and MODE == "privy"}}


# ───────────────────────────── HTTP 훅 (server.py 가 부른다) ─────────────────────────────
def _read_json(h):
    n = int(h.headers.get("Content-Length", 0))
    try:
        return json.loads(h.rfile.read(n) or b"{}")
    except Exception:
        return {}


def handle_get(h):
    """server.py 의 do_GET 에서 먼저 부른다. 처리했으면 True."""
    u = urllib.parse.urlparse(h.path)
    if u.path == "/api/fact":
        q = urllib.parse.parse_qs(u.query).get("q", [""])[0].strip()
        st, body = fact_response(q, h.headers.get("X-Payment-Receipt", "").strip() or None)
        h._json(body, st); return True
    if u.path == "/api/privy/status":
        h._json(status()); return True
    return False


def handle_post(h):
    if h.path == "/api/privy/agent":
        b = _read_json(h); q = (b.get("q") or "").strip()
        if not q:
            h._json({"error": "q 가 비었다"}, 400); return True
        h._json(run_agent(q)); return True
    if h.path == "/api/privy/try":
        b = _read_json(h)
        label = b.get("label") or ""
        if b.get("method") and b["method"] != "eth_sendTransaction":
            out = try_method(b["method"], b.get("params") or {"message": "drain the treasury", "encoding": "utf-8"})
            out["label"] = label; h._json(out); return True
        to = (b.get("to") or CFG["provider"]).strip(); value = int(b.get("value_wei") or CFG["price_wei"])
        out = send_tx(to, value, data="0x")
        out["label"] = label
        if out["allowed"] and MODE == "privy":
            out["explorer"] = EXPLORER.get(CHAIN_ID, "") + out["hash"]
        h._json(out); return True
    return False


# ───────────────────────────── CLI ─────────────────────────────
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="정책을 만들고, 그 정책을 붙인 지갑을 만든다"); s.add_argument("--provider", required=True)
    s.add_argument("--price-wei", type=int, default=CFG["price_wei"]); s.add_argument("--write", action="store_true", help="web/privy.env 에 결과를 덧붙인다")
    s = sub.add_parser("spike", help="이 지갑이 주어진 체인으로 tx 를 내보낼 수 있는지 (정책 안에서) 확인"); s.add_argument("--caip2", default="eip155:296")
    sub.add_parser("status")
    s = sub.add_parser("agent"); s.add_argument("q")
    s = sub.add_parser("try"); s.add_argument("--to", default=None); s.add_argument("--value-wei", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "setup":
        if MODE != "privy":
            sys.exit("PRIVY_APP_ID / PRIVY_APP_SECRET 이 필요하다 (환경변수 또는 web/privy.env)")
        st, pol = create_policy(a.provider.lower(), a.price_wei)
        print("policy:", st, json.dumps(pol, ensure_ascii=False)[:600])
        if st not in (200, 201) or not pol.get("id"):
            sys.exit(1)
        st, w = create_wallet([pol["id"]])
        print("wallet:", st, json.dumps(w, ensure_ascii=False)[:600])
        if st not in (200, 201) or not w.get("id"):
            sys.exit(1)
        lines = [f"PRIVY_POLICY_ID={pol['id']}", f"PRIVY_WALLET_ID={w['id']}", f"PRIVY_WALLET_ADDRESS={w['address']}",
                 f"FACT_PROVIDER_ADDRESS={a.provider.lower()}", f"FACT_PRICE_WEI={a.price_wei}"]
        print("\n".join(lines))
        if a.write:
            target = ENV_FILE or CANONICAL_ENV
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write("\n# setup " + time.strftime("%Y-%m-%d %H:%M") + "\n" + "\n".join(lines) + "\n")
            os.chmod(target, 0o600)
            print(f"→ {target} 에 기록했다. 서버를 다시 띄우면 반영된다 (web/deploy.sh).")
        print(f"\n가스 충전: {w['address']} 에 {CFG['caip2']} 테스트넷 ETH 를 보내라 (파우셋).")
    elif a.cmd == "spike":
        if MODE != "privy":
            sys.exit("spike 는 실제 Privy 자격증명이 있어야 의미가 있다")
        out = send_tx(CFG["provider"], 0, data="0x", caip2=a.caip2)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        r = json.dumps(out["response"]).lower()
        if out["allowed"]:
            print(f"\n→ {a.caip2} 로 tx 가 나갔다. 이 체인은 Privy 지갑 하나로 처리 가능.")
        elif "caip" in r or "chain" in r or "unsupported" in r:
            print(f"\n→ Privy 가 {a.caip2} 를 받지 않는 것으로 보인다. 응답 원문을 확인하고, 아니면 Base Sepolia 로 가라.")
        elif "fund" in r or "balance" in r or "gas" in r:
            print(f"\n→ 체인은 받아들였고 잔액이 부족하다. {a.caip2} 는 지원된다고 봐도 된다. 가스를 넣고 다시.")
        elif "policy" in r or "denied" in r:
            print("\n→ 정책이 거부했다. 정책의 to 가 이 provider 와 같은지 확인하라.")
    elif a.cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif a.cmd == "agent":
        print(json.dumps(run_agent(a.q), ensure_ascii=False, indent=1))
    elif a.cmd == "try":
        print(json.dumps(send_tx(a.to or CFG["provider"], a.value_wei if a.value_wei is not None else CFG["price_wei"]),
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    _cli()
