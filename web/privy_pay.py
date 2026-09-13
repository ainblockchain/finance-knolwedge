"""이름으로 라우팅되는 지급 — 종목코드가 결제 라우팅 키이고, Privy 정책 엔진이 검증되지 않은 수취인에게는 서명을 거부한다.

  POST /settle {instruction: "태양기계에 대금 지급", model: "before"|"after"}
    1. 지시문에서 회사명을 찾고, 해당 모델(:8000 학습 전 / :8001 학습 후)에 "{회사명} 종목코드 알려줘. 숫자만." 을 묻는다
    2. 답에서 6자리 코드를 뽑는다
    3. 레지스트리(web/registry.json)에서 그 코드의 수취 주소를 찾는다. 없으면 같은 규칙으로 주소를 파생한다
       (address = 0x + keccak256("ticker:" + code)[12:]) — 미검증 주소라도 **여기서 막지 않는다**
    4. Privy 서버 지갑으로 eth_sendTransaction 을 보낸다. 정책(to ∈ 검증 주소 목록, value ≤ 상한, 그 외 DENY)은 TEE 안에서 평가된다
    5. verdict 는 Privy 응답에서만 파생한다: hash 가 오면 exec, code == policy_violation 이면 deny, 그 외 error

에이전틱 금융의 실제 사고는 "너무 많이 씀" 이 아니라 "식별자를 환각해 엉뚱한 상대에게 보냄" 이다. 원본 모델은 태양기계에
033240 처럼 형식은 완벽하고 값만 틀린 코드를 답한다. 방어는 두 겹이고 서로 독립이다: 지식 교정(19%→100%) 은 우리 연구가,
권한 강제(검증된 주소가 아니면 서명 거부) 는 Privy 가 한다. 같은 지갑·같은 정책·같은 코드에서 모델만 바꾸면 결과가 갈린다.

설정 파일 (KEY=VALUE): $PRIVY_ENV_FILE → web/privy.env → ~/.privy/finance-demo.env (기본, 체크아웃과 무관)
  PRIVY_APP_ID, PRIVY_APP_SECRET, PRIVY_WALLET_ID, PRIVY_WALLET_ADDRESS, PRIVY_POLICY_ID
  SETTLE_AMOUNT_WEI (기본 10000000000000 = 0.00001 ETH; 정책 상한과 같다)   CHAIN_CAIP2 (eip155:11155111)   RPC_URL
PRIVY_APP_ID 가 없으면 mock 모드 — 정책 판정을 이 프로세스가 흉내 내고 응답에 MOCK 이라 적는다. 제출 증거가 될 수 없다.

CLI:
  python3 web/privy_pay.py registry [--n 16]      results/chat-eval.json 에서 학습 전 오답·학습 후 정답 종목으로 web/registry.json 생성
  python3 web/privy_pay.py policy                 레지스트리 주소 전체를 allowlist 로 하는 정책을 만들어(또는 갱신해) 지갑에 붙인다
  python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before|after
  python3 web/privy_pay.py try --method personal_sign     정책이 허용하지 않은 메서드 → 가스 없이 policy_violation
  python3 web/privy_pay.py status
"""
import argparse, base64, hashlib, json, os, re, sys, threading, time, urllib.error, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
PRIVY_API = "https://api.privy.io"
UA = "finance-demo-agent/1.0 (+https://finance-demo.ainetwork.ai)"  # Cloudflare 와 공개 RPC 가 urllib 기본 UA 를 403 으로 막는다

# ───────────────────────────── 설정 ─────────────────────────────
CANONICAL_ENV = os.path.expanduser("~/.privy/finance-demo.env")
ENV_CANDIDATES = [os.environ.get("PRIVY_ENV_FILE", ""), os.path.join(ROOT, "privy.env"), CANONICAL_ENV]
ENV_FILE = None


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
    "app_secret": os.environ.get("PRIVY_APP_SECRET", ""),
    "wallet_id": os.environ.get("PRIVY_WALLET_ID", ""),
    "wallet_address": os.environ.get("PRIVY_WALLET_ADDRESS", "").lower(),
    "policy_id": os.environ.get("PRIVY_POLICY_ID", ""),
    "amount_wei": int(os.environ.get("SETTLE_AMOUNT_WEI", os.environ.get("FACT_PRICE_WEI", "10000000000000"))),
    "caip2": os.environ.get("CHAIN_CAIP2", "eip155:11155111"),
    "rpc_url": os.environ.get("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com"),
    "before_api": os.environ.get("BEFORE_API", "http://localhost:8000"),
    "after_api": os.environ.get("AFTER_API", "http://localhost:8001"),
}
MODE = "privy" if CFG["app_id"] and CFG["app_secret"] else "mock"
CHAIN_ID = int(CFG["caip2"].split(":")[1])
EXPLORER = {84532: "https://sepolia.basescan.org/tx/", 8453: "https://basescan.org/tx/",
            11155111: "https://sepolia.etherscan.io/tx/", 296: "https://hashscan.io/testnet/transaction/"}
MODEL = "Qwen3.8-Flash-Next"
PROMPT = "{name} 종목코드 알려줘. 숫자만."   # results/chat-eval.json 의 q 와 같다 — 벤치마크와 같은 질문이어야 한다
CODE_RE = re.compile(r"([0-9][0-9A-Z]{5})")  # scripts/chat_eval.py 와 같은 파서 — 첫 6자리 코드. 벤치마크와 같은 눈으로 읽어야 한다
REGISTRY_PATH = os.path.join(ROOT, "registry.json")
STATE_PATH = os.environ.get("PRIVY_STATE", os.path.join(ROOT, "privy_state.json"))
if MODE == "mock":
    CFG["wallet_id"] = CFG["wallet_id"] or "mock-wallet"
    CFG["wallet_address"] = CFG["wallet_address"] or "0x000000000000000000000000000000000a9e4700"
    CFG["policy_id"] = CFG["policy_id"] or "mock-policy"
print(f"privy: mode={MODE} env={ENV_FILE or '(없음)'} chain={CFG['caip2']}" +
      ("" if MODE == "privy" else f"  ← MOCK. PRIVY_APP_ID/PRIVY_APP_SECRET 을 {CANONICAL_ENV} 에 넣으면 LIVE"), file=sys.stderr, flush=True)


# ───────────────────────────── keccak256 (표준 라이브러리에 없어 직접 구현) ─────────────────────────────
_M64 = (1 << 64) - 1
_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
       0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
       0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
       0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]


def _rol(a, n):
    return ((a << n) | (a >> (64 - n))) & _M64 if n else a


def _keccak_f(A):
    for rc in _RC:
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _ROT[x][y])
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        A[0][0] ^= rc
    return A


def keccak256(data: bytes) -> bytes:
    """이더리움의 keccak256 (SHA3 표준화 전 패딩 0x01). 검증 벡터: keccak256(b"") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"""
    rate = 136
    msg = bytearray(data) + b"\x01"
    while len(msg) % rate:
        msg.append(0)
    msg[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(msg[off + 8 * i: off + 8 * i + 8], "little")
        A = _keccak_f(A)
    return b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
assert keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"


# ───────────────────────────── 레지스트리: 종목코드 → 수취 주소 ─────────────────────────────
def derive_address(code: str) -> str:
    """코드에서 결정론적으로 파생한 주소. 같은 코드 → 항상 같은 주소, 다른 코드 → 다른 주소.
    실제 소유자는 없다 — 중요한 것은 코드가 곧 수취인이라는 점이고, 레지스트리에 없는 코드의 주소는 정책 밖이라 Privy 가 거부한다."""
    return "0x" + keccak256(("ticker:" + code).encode()).hex()[24:]


def _krx():
    try:
        return json.load(open(os.path.join(REPO, "data", "krx.json"), encoding="utf-8"))
    except FileNotFoundError:
        return []


def load_registry():
    try:
        return json.load(open(REGISTRY_PATH, encoding="utf-8"))
    except FileNotFoundError:
        return []


REGISTRY = load_registry()
BY_CODE = {r["code"]: r for r in REGISTRY}
BY_NAME = sorted(REGISTRY, key=lambda r: -len(r["name"]))  # 긴 이름 먼저 — "삼성전자우" 가 "삼성전자" 에 먹히지 않게


def find_company(text):
    """지시문에서 레지스트리의 회사명을 찾는다. 여러 개면 전부 (등장 순서)."""
    hits = []
    for r in BY_NAME:
        i = text.find(r["name"])
        if i >= 0 and not any(r["name"] in h["name"] for h in hits):
            hits.append({**r, "pos": i})
    return sorted(hits, key=lambda h: h["pos"])


def resolve(name):
    """회사명 → {code, address}. 완료 기준: resolve("태양기계") → 116100 → 0x…"""
    for r in BY_NAME:
        if r["name"] == name:
            return r
    return None


def build_registry(n=16, extra=("태양기계", "엑시온그룹", "성원에너텍", "닷밀", "KR모터스", "픽셀플러스")):
    """results/chat-eval.json 에서 학습 전 오답·학습 후 정답이었던 종목을 고른다 (전후 대비가 나와야 한다)."""
    krx = {r["회사명"]: r for r in _krx()}
    chosen, seen = [], set()

    def add(name, before_gen=None):
        r = krx.get(name)
        if not r or r["종목코드"] in seen:
            return
        seen.add(r["종목코드"])
        row = {"name": name, "code": r["종목코드"], "address": derive_address(r["종목코드"]), "market": r.get("시장구분", "")}
        if before_gen is not None:
            row["before_generation"] = before_gen
        chosen.append(row)

    ev = None
    try:
        ev = json.load(open(os.path.join(REPO, "results", "chat-eval.json"), encoding="utf-8"))
    except FileNotFoundError:
        pass
    gens = {}
    if ev:
        for s, bg, bok, aok in zip(ev["sample"], ev["before"]["gens"], ev["before"]["ok"], ev["after"]["ok"]):
            if not bok and aok:
                gens[s["name"]] = (CODE_RE.search(bg or "") or [None, None])[1] if CODE_RE.search(bg or "") else (bg or "")[:12]
    for name in extra:
        add(name, gens.get(name))
    if ev:
        for s in ev["sample"]:
            if len(chosen) >= n:
                break
            if s["name"] in gens:
                add(s["name"], gens[s["name"]])
    json.dump(chosen, open(REGISTRY_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return chosen


# ───────────────────────────── 상태 (원장) ─────────────────────────────
_lock = threading.Lock()
try:
    _state = json.load(open(STATE_PATH, encoding="utf-8"))
except Exception:
    _state = {}
_state.setdefault("ledger", [])


def _save():
    tmp = STATE_PATH + ".tmp"
    json.dump(_state, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def ledger_add(row):
    with _lock:
        _state["ledger"].append(row); _state["ledger"] = _state["ledger"][-300:]; _save()


# ───────────────────────────── Privy REST ─────────────────────────────
def privy(method, path, body=None):
    """(http status, json). Basic 인증은 app_id:app_secret, privy-app-id 헤더 동반."""
    hdr = {"Authorization": "Basic " + base64.b64encode(f"{CFG['app_id']}:{CFG['app_secret']}".encode()).decode(),
           "privy-app-id": CFG["app_id"], "Content-Type": "application/json", "User-Agent": UA}
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


def policy_body(addresses, cap_wei):
    """룰 하나: to ∈ 검증 주소 목록 AND value ≤ 상한 → ALLOW. Privy 는 명시적으로 허용하지 않은 것을 전부 거부하므로 룰을 더 넣지 않는다."""
    return {"version": "1.0", "name": "Verified identifier settlement", "chain_type": "ethereum",
            "rules": [{"name": "Allow settlement to verified tickers only", "method": "eth_sendTransaction", "action": "ALLOW",
                       "conditions": [
                           {"field_source": "ethereum_transaction", "field": "to", "operator": "in", "value": list(addresses)},
                           {"field_source": "ethereum_transaction", "field": "value", "operator": "lte", "value": str(cap_wei)}]}]}


def get_policy(pid):
    return privy("GET", f"/v1/policies/{pid}")


def apply_allowlist_policy():
    """레지스트리 주소 전체를 allowlist 로 하는 정책을 지갑에 건다.
    1순위: 기존 정책의 룰을 PATCH (id·지갑 그대로). 2순위: 새 정책 POST 후 지갑에 PATCH 로 policy_ids 교체.
    반환 {ok, how, policy_id, wallet_policy_ids, log}."""
    body = policy_body([r["address"] for r in REGISTRY], CFG["amount_wei"])
    log = []
    if MODE != "privy":
        return {"ok": False, "how": "mock", "log": ["mock 모드에서는 정책을 만들 수 없다"]}
    if CFG["policy_id"]:
        st, r = privy("PATCH", f"/v1/policies/{CFG['policy_id']}", {"rules": body["rules"], "name": body["name"]})
        log.append(f"PATCH /v1/policies/{CFG['policy_id']} → {st} {json.dumps(r, ensure_ascii=False)[:200]}")
        if st == 200:
            st2, w = privy("GET", f"/v1/wallets/{CFG['wallet_id']}")
            return {"ok": True, "how": "patched-existing-policy", "policy_id": CFG["policy_id"],
                    "wallet_policy_ids": w.get("policy_ids"), "log": log}
    st, pol = privy("POST", "/v1/policies", body)
    log.append(f"POST /v1/policies → {st} {json.dumps(pol, ensure_ascii=False)[:200]}")
    if st not in (200, 201) or not pol.get("id"):
        return {"ok": False, "how": "create-failed", "log": log}
    st, w = privy("PATCH", f"/v1/wallets/{CFG['wallet_id']}", {"policy_ids": [pol["id"]]})
    log.append(f"PATCH /v1/wallets/{CFG['wallet_id']} policy_ids → {st} {json.dumps(w, ensure_ascii=False)[:200]}")
    if st != 200:
        return {"ok": False, "how": "attach-failed", "policy_id": pol["id"], "log": log,
                "hint": "기존 지갑에 정책을 못 붙인다. 새 지갑을 만들고 가스를 옮겨야 한다 — 사람에게 물을 것"}
    return {"ok": True, "how": "new-policy-attached", "policy_id": pol["id"], "wallet_policy_ids": w.get("policy_ids"), "log": log}


def _mock_policy(to, value_wei):
    if (to or "").lower() not in {r["address"].lower() for r in REGISTRY}:
        return False, "MOCK policy: RPC request denied due to policy violation (to not in allowlist)"
    if int(value_wei) > CFG["amount_wei"]:
        return False, "MOCK policy: RPC request denied due to policy violation (value over cap)"
    return True, "MOCK policy: ALLOW"


def send_tx(to, value_wei, data="0x", caip2=None):
    """Privy 지갑으로 eth_sendTransaction. 반환 {allowed, denied_by_policy, hash, status, request, response, ms}."""
    caip2 = caip2 or CFG["caip2"]
    req = {"method": "eth_sendTransaction", "caip2": caip2,
           "params": {"transaction": {"to": to, "value": hex(int(value_wei)), "data": data, "chain_id": int(caip2.split(":")[1])}}}
    t0 = time.time()
    if MODE == "mock":
        ok, msg = _mock_policy(to, value_wei)
        h = "0x" + hashlib.sha256(f"{to}{value_wei}{data}{time.time()}".encode()).hexdigest() if ok else None
        st, resp = (200, {"method": "eth_sendTransaction", "data": {"hash": h, "caip2": caip2}}) if ok else (400, {"error": msg, "code": "policy_violation"})
    else:
        st, resp = privy("POST", f"/v1/wallets/{CFG['wallet_id']}/rpc", req)
        h = (resp.get("data") or {}).get("hash") if st == 200 else None
    allowed = st == 200 and bool(h)
    # 거부 두 종류를 구분한다: 정책 엔진이 서명을 거부(policy_violation) vs 정책은 통과했으나 브로드캐스트·시뮬레이션 실패(잔액 부족 등).
    return {"allowed": allowed, "denied_by_policy": (not allowed) and (resp or {}).get("code") == "policy_violation",
            "hash": h, "status": st, "request": req, "response": resp, "ms": int((time.time() - t0) * 1000), "mode": MODE}


def try_method(method, params):
    """정책이 허용하지 않은 메서드를 같은 지갑으로 부른다 → 브로드캐스트가 없어 잔액과 무관하게 정책 판정만 남는다."""
    req = {"method": method, "params": params}
    t0 = time.time()
    if MODE == "mock":
        st, resp = 400, {"error": "MOCK policy: RPC request denied due to policy violation", "code": "policy_violation"}
    else:
        st, resp = privy("POST", f"/v1/wallets/{CFG['wallet_id']}/rpc", req)
    return {"allowed": st == 200, "denied_by_policy": st != 200 and (resp or {}).get("code") == "policy_violation", "hash": None,
            "status": st, "request": req, "response": resp, "ms": int((time.time() - t0) * 1000), "mode": MODE}


# ───────────────────────────── 체인 RPC ─────────────────────────────
def rpc(method, params):
    req = urllib.request.Request(CFG["rpc_url"], data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA})
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


# ───────────────────────────── 모델 ─────────────────────────────
def ask_model(side, q, max_tokens=40):  # chat_eval.py 와 같은 max_tokens
    url = CFG["before_api" if side == "before" else "after_api"] + "/v1/chat/completions"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": q}], "stream": False, "temperature": 0.0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        m = json.load(r)["choices"][0]["message"]
    txt = (m.get("content") or "").strip()
    return next((l.strip() for l in txt.splitlines() if l.strip()), txt)  # 답은 한 줄. 뒤에 붙는 파서 잔여물은 답이 아니다


# ───────────────────────────── 정산 ─────────────────────────────
def settle_one(company, model, instruction=""):
    """회사 하나에 대한 지급 한 건. company 는 레지스트리 항목."""
    t0 = time.time()
    q = PROMPT.format(name=company["name"])
    out = {"instruction": instruction, "model": model, "company": company["name"], "prompt": q,
           "expected_code": company["code"], "amount_wei": CFG["amount_wei"]}
    try:
        answer = ask_model(model, q)
    except Exception as e:
        out.update(model_answer=None, code=None, correct=None, to=None, verified_recipient=False, verdict="error",
                   privy_status=None, privy_raw={"error": f"모델 서버 응답 없음 ({model}): {e}"}, tx_hash=None,
                   latency_ms=int((time.time() - t0) * 1000))
        ledger_add(out); return out
    m = CODE_RE.search(answer)
    code = m.group(1) if m else None
    out.update(model_answer=answer, code=code, correct=(code == company["code"]))
    if code is None:
        out.update(to=None, verified_recipient=False, verdict="error", privy_status=None,
                   privy_raw={"error": "답에서 6자리 종목코드를 찾지 못했다 — 보낼 상대가 없다"}, tx_hash=None,
                   latency_ms=int((time.time() - t0) * 1000))
        ledger_add(out); return out
    entry = BY_CODE.get(code)
    to = entry["address"] if entry else derive_address(code)   # 미검증 코드도 같은 규칙으로 주소를 만든다. 막는 것은 Privy 의 일이다.
    out.update(to=to, verified_recipient=bool(entry), resolved_name=entry["name"] if entry else None)
    tx = send_tx(to, CFG["amount_wei"], data="0x" + code.encode().hex())   # tx data 에 종목코드를 메모로 남긴다
    verdict = "exec" if tx["allowed"] else ("deny" if tx["denied_by_policy"] else "error")
    out.update(verdict=verdict, privy_status=tx["status"], privy_raw=tx["response"], privy_request=tx["request"],
               tx_hash=tx["hash"], explorer=(EXPLORER.get(CHAIN_ID, "") + tx["hash"]) if tx["hash"] and MODE == "privy" else None,
               latency_ms=int((time.time() - t0) * 1000), mode=MODE, t=time.time())
    ledger_add(out)
    return out


def settle(instruction, model):
    """POST /settle — 지시문에서 첫 회사명을 찾아 한 건 지급."""
    hits = find_company(instruction)
    if not hits:
        return 400, {"error": "지시문에서 레지스트리의 회사명을 찾지 못했다", "instruction": instruction,
                     "known": [r["name"] for r in REGISTRY]}
    return 200, settle_one(hits[0], model, instruction)


def settle_batch(instruction, model):
    """POST /settle/batch — 지시문의 모든 회사에 지급. 한 건이 환각이면 그 건만 거부되고 나머지는 정산된다 (부분 실패 격리)."""
    hits = find_company(instruction)
    if not hits:
        return 400, {"error": "지시문에서 레지스트리의 회사명을 찾지 못했다", "instruction": instruction}
    legs = [settle_one(h, model, instruction) for h in hits]
    return 200, {"instruction": instruction, "model": model, "legs": legs,
                 "summary": {v: sum(1 for l in legs if l["verdict"] == v) for v in ("exec", "deny", "error")}}


def status():
    bal = balance_wei(CFG["wallet_address"]) if MODE == "privy" and CFG["wallet_address"] else None
    with _lock:
        ledger = list(_state["ledger"][-60:])
    return {"mode": MODE, "env_file": ENV_FILE, "wallet_id": CFG["wallet_id"], "wallet_address": CFG["wallet_address"],
            "policy_id": CFG["policy_id"], "caip2": CFG["caip2"], "chain_id": CHAIN_ID, "explorer": EXPLORER.get(CHAIN_ID, ""),
            "amount_wei": CFG["amount_wei"], "amount_eth": CFG["amount_wei"] / 1e18,
            "balance_wei": bal, "balance_eth": (bal / 1e18) if bal is not None else None,
            "registry_size": len(REGISTRY), "policy": policy_body([r["address"] for r in REGISTRY], CFG["amount_wei"]),
            "ledger": ledger,
            "counts": {v: sum(1 for l in ledger if l.get("verdict") == v) for v in ("exec", "deny", "error")}}


# ───────────────────────────── HTTP 훅 (server.py 가 부른다) ─────────────────────────────
def _read_json(h):
    n = int(h.headers.get("Content-Length", 0))
    try:
        return json.loads(h.rfile.read(n) or b"{}")
    except Exception:
        return {}


def handle_get(h):
    u = urllib.parse.urlparse(h.path)
    if u.path == "/api/privy/status":
        h._json(status()); return True
    if u.path == "/api/privy/registry":
        h._json({"registry": REGISTRY, "derive": 'address = "0x" + keccak256("ticker:" + code)[12:]'}); return True
    return False


def handle_post(h):
    if h.path in ("/settle", "/settle/batch"):
        b = _read_json(h)
        instr = (b.get("instruction") or "").strip(); model = b.get("model", "after")
        if model not in ("before", "after"):
            h._json({"error": "model 은 before 또는 after"}, 400); return True
        if not instr:
            h._json({"error": "instruction 이 비었다"}, 400); return True
        st, body = (settle_batch if h.path.endswith("/batch") else settle)(instr, model)
        h._json(body, st); return True
    if h.path == "/api/privy/try":
        b = _read_json(h)
        if b.get("method") and b["method"] != "eth_sendTransaction":
            out = try_method(b["method"], b.get("params") or {"message": "pay 1 ETH to 0xdead", "encoding": "utf-8"})
        else:
            out = send_tx((b.get("to") or "").strip() or derive_address("000000"), int(b.get("value_wei") or CFG["amount_wei"]))
        out["label"] = b.get("label") or ""
        h._json(out); return True
    return False


# ───────────────────────────── CLI ─────────────────────────────
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("registry"); s.add_argument("--n", type=int, default=16)
    sub.add_parser("policy")
    sub.add_parser("status")
    s = sub.add_parser("settle"); s.add_argument("instruction"); s.add_argument("--model", default="after", choices=["before", "after"]); s.add_argument("--batch", action="store_true")
    s = sub.add_parser("try"); s.add_argument("--method", default=None); s.add_argument("--to", default=None); s.add_argument("--value-wei", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "registry":
        rows = build_registry(a.n)
        for r in rows:
            print(f"{r['name']:<12s} {r['code']}  {r['address']}  학습 전 생성: {r.get('before_generation')}")
        print(f"→ {REGISTRY_PATH} ({len(rows)}개)")
    elif a.cmd == "policy":
        print(json.dumps(apply_allowlist_policy(), ensure_ascii=False, indent=1))
    elif a.cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif a.cmd == "settle":
        st, body = (settle_batch if a.batch else settle)(a.instruction, a.model)
        print(json.dumps(body, ensure_ascii=False, indent=1))
    elif a.cmd == "try":
        if a.method:
            print(json.dumps(try_method(a.method, {"message": "pay 1 ETH to 0xdead", "encoding": "utf-8"}), ensure_ascii=False, indent=1))
        else:
            print(json.dumps(send_tx(a.to or derive_address("000000"), a.value_wei if a.value_wei is not None else CFG["amount_wei"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    _cli()
