"""Settlement routed by name — the stock ticker is the payment routing key, and Privy's policy engine refuses to sign
for any recipient that is not a verified one.

  POST /settle {instruction: "태양기계에 대금 지급" (= "pay 태양기계"), model: "before"|"after"}
    1. Find the company name in the instruction and ask the chosen model (:8000 base / :8001 trained)
       "{company} 종목코드 알려줘. 숫자만." ("tell me the ticker of {company}, digits only")
    2. Take the first 6-character code from the answer
    3. Look the code up in the registry (web/registry.json). If absent, derive the address by the same rule
       (address = 0x + keccak256("ticker:" + code)[12:]) — an unverified address is **not blocked here**
    4. Send eth_sendTransaction from the Privy server wallet. The policy (to ∈ verified addresses, value ≤ cap, everything
       else DENY) is evaluated inside the TEE
    5. The verdict is derived only from Privy's response: a hash → exec, code == policy_violation → deny, anything else → error

The real accident in agentic finance is not "spent too much" but "hallucinated an identifier and paid the wrong party".
The base model answers 033240 for 태양기계: perfectly formed, simply wrong. The defense has two independent layers:
knowledge correction (19% → 100%) is our research; authority enforcement (no signature unless the address is verified) is
Privy. Same wallet, same policy, same code — swap the model and the outcome flips.

Config file (KEY=VALUE): $PRIVY_ENV_FILE → web/privy.env → ~/.privy/finance-demo.env (default; independent of the checkout)
  PRIVY_APP_ID, PRIVY_APP_SECRET, PRIVY_WALLET_ID, PRIVY_WALLET_ADDRESS, PRIVY_POLICY_ID
  SETTLE_AMOUNT_WEI (default 10000000000000 = 0.00001 ETH; equals the policy cap)   CHAIN_CAIP2 (eip155:11155111)   RPC_URL
Without PRIVY_APP_ID the module runs in mock mode — this process imitates the policy verdict and every response says MOCK.
Mock output is not submission evidence.

CLI:
  python3 web/privy_pay.py registry [--n 16]      build web/registry.json from results/chat-eval.json (wrong before, right after)
  python3 web/privy_pay.py policy                 create (or update) the allowlist policy for all registry addresses and attach it
  python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before|after
  python3 web/privy_pay.py try --method personal_sign     a method the policy does not allow → policy_violation, no gas needed
  python3 web/privy_pay.py status
"""
import argparse, base64, hashlib, json, os, re, sys, threading, time, urllib.error, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
PRIVY_API = "https://api.privy.io"
UA = "finance-demo-agent/1.0 (+https://finance-demo.ainetwork.ai)"  # Cloudflare and public RPCs reject urllib's default User-Agent with 403

# ───────────────────────────── config ─────────────────────────────
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
PROMPT = "{name} 종목코드 알려줘. 숫자만."   # "tell me the ticker of {name}, digits only" — identical to q in results/chat-eval.json; must match the benchmark
CODE_RE = re.compile(r"([0-9][0-9A-Z]{5})")  # same parser as scripts/chat_eval.py — first 6-character code; read answers the way the benchmark does
REGISTRY_PATH = os.path.join(ROOT, "registry.json")
STATE_PATH = os.environ.get("PRIVY_STATE", os.path.join(ROOT, "privy_state.json"))
if MODE == "mock":
    CFG["wallet_id"] = CFG["wallet_id"] or "mock-wallet"
    CFG["wallet_address"] = CFG["wallet_address"] or "0x000000000000000000000000000000000a9e4700"
    CFG["policy_id"] = CFG["policy_id"] or "mock-policy"
print(f"privy: mode={MODE} env={ENV_FILE or '(none)'} chain={CFG['caip2']}" +
      ("" if MODE == "privy" else f"  ← MOCK. Put PRIVY_APP_ID/PRIVY_APP_SECRET in {CANONICAL_ENV} to go LIVE"), file=sys.stderr, flush=True)


# ───────────────────────────── keccak256 (not in the standard library, so implemented here) ─────────────────────────────
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
    """Ethereum's keccak256 (pre-SHA3 padding byte 0x01). Test vector: keccak256(b"") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"""
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


# ───────────────────────────── registry: ticker → recipient address ─────────────────────────────
def derive_address(code: str) -> str:
    """Address derived deterministically from the code: same code → same address, different code → different address.
    Nobody owns these addresses — what matters is that the code IS the recipient, and the address of a code that is not
    in the registry falls outside the policy, so Privy refuses it."""
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
BY_NAME = sorted(REGISTRY, key=lambda r: -len(r["name"]))  # longest names first so "삼성전자우" is not swallowed by "삼성전자"


def find_company(text):
    """Find registry company names in the instruction — all of them, in order of appearance."""
    hits = []
    for r in BY_NAME:
        i = text.find(r["name"])
        if i >= 0 and not any(r["name"] in h["name"] for h in hits):
            hits.append({**r, "pos": i})
    return sorted(hits, key=lambda h: h["pos"])


def resolve(name):
    """Company name → {code, address}. Acceptance: resolve("태양기계") → 116100 → 0x…"""
    for r in BY_NAME:
        if r["name"] == name:
            return r
    return None


def build_registry(n=16, extra=("태양기계", "엑시온그룹", "성원에너텍", "닷밀", "KR모터스", "픽셀플러스")):
    """Pick tickers the base model got wrong and the trained model got right (results/chat-eval.json), so the A/B shows."""
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


# ───────────────────────────── state (ledger) ─────────────────────────────
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
    """(http status, json). Basic auth is app_id:app_secret, with the privy-app-id header alongside."""
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
    """One rule: to ∈ verified addresses AND value ≤ cap → ALLOW. Privy denies everything not explicitly allowed, so no more rules."""
    return {"version": "1.0", "name": "Verified identifier settlement", "chain_type": "ethereum",
            "rules": [{"name": "Allow settlement to verified tickers only", "method": "eth_sendTransaction", "action": "ALLOW",
                       "conditions": [
                           {"field_source": "ethereum_transaction", "field": "to", "operator": "in", "value": list(addresses)},
                           {"field_source": "ethereum_transaction", "field": "value", "operator": "lte", "value": str(cap_wei)}]}]}


def get_policy(pid):
    return privy("GET", f"/v1/policies/{pid}")


def apply_allowlist_policy():
    """Put an allowlist policy covering every registry address on the wallet.
    First choice: PATCH the existing policy's rules (same id, same wallet). Fallback: POST a new policy, then PATCH the
    wallet's policy_ids. Returns {ok, how, policy_id, wallet_policy_ids, log}."""
    body = policy_body([r["address"] for r in REGISTRY], CFG["amount_wei"])
    log = []
    if MODE != "privy":
        return {"ok": False, "how": "mock", "log": ["cannot create a policy in mock mode"]}
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
                "hint": "the policy cannot be attached to the existing wallet; a new wallet plus a gas transfer is needed — ask a human"}
    return {"ok": True, "how": "new-policy-attached", "policy_id": pol["id"], "wallet_policy_ids": w.get("policy_ids"), "log": log}


def _mock_policy(to, value_wei):
    if (to or "").lower() not in {r["address"].lower() for r in REGISTRY}:
        return False, "MOCK policy: RPC request denied due to policy violation (to not in allowlist)"
    if int(value_wei) > CFG["amount_wei"]:
        return False, "MOCK policy: RPC request denied due to policy violation (value over cap)"
    return True, "MOCK policy: ALLOW"


def send_tx(to, value_wei, data="0x", caip2=None):
    """eth_sendTransaction from the Privy wallet. Returns {allowed, denied_by_policy, hash, status, request, response, ms}."""
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
    # Two kinds of refusal: the policy engine refused to sign (policy_violation) vs. the policy passed but broadcast/simulation failed (e.g. insufficient funds).
    return {"allowed": allowed, "denied_by_policy": (not allowed) and (resp or {}).get("code") == "policy_violation",
            "hash": h, "status": st, "request": req, "response": resp, "ms": int((time.time() - t0) * 1000), "mode": MODE}


def try_method(method, params):
    """Call a method the policy does not allow on the same wallet → no broadcast, so only the policy verdict remains, regardless of balance."""
    req = {"method": method, "params": params}
    t0 = time.time()
    if MODE == "mock":
        st, resp = 400, {"error": "MOCK policy: RPC request denied due to policy violation", "code": "policy_violation"}
    else:
        st, resp = privy("POST", f"/v1/wallets/{CFG['wallet_id']}/rpc", req)
    return {"allowed": st == 200, "denied_by_policy": st != 200 and (resp or {}).get("code") == "policy_violation", "hash": None,
            "status": st, "request": req, "response": resp, "ms": int((time.time() - t0) * 1000), "mode": MODE}


# ───────────────────────────── chain RPC ─────────────────────────────
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


# ───────────────────────────── model ─────────────────────────────
def ask_model(side, q, max_tokens=40):  # same max_tokens as chat_eval.py
    url = CFG["before_api" if side == "before" else "after_api"] + "/v1/chat/completions"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": q}], "stream": False, "temperature": 0.0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        m = json.load(r)["choices"][0]["message"]
    txt = (m.get("content") or "").strip()
    return next((l.strip() for l in txt.splitlines() if l.strip()), txt)  # the answer is one line; trailing parser debris is not the answer


# ───────────────────────────── settlement ─────────────────────────────
def settle_one(company, model, instruction=""):
    """One payment to one company. `company` is a registry entry."""
    t0 = time.time()
    q = PROMPT.format(name=company["name"])
    out = {"instruction": instruction, "model": model, "company": company["name"], "prompt": q,
           "expected_code": company["code"], "amount_wei": CFG["amount_wei"]}
    try:
        answer = ask_model(model, q)
    except Exception as e:
        out.update(model_answer=None, code=None, correct=None, to=None, verified_recipient=False, verdict="error",
                   privy_status=None, privy_raw={"error": f"model server not responding ({model}): {e}"}, tx_hash=None,
                   latency_ms=int((time.time() - t0) * 1000))
        ledger_add(out); return out
    m = CODE_RE.search(answer)
    code = m.group(1) if m else None
    out.update(model_answer=answer, code=code, correct=(code == company["code"]))
    if code is None:
        out.update(to=None, verified_recipient=False, verdict="error", privy_status=None,
                   privy_raw={"error": "no 6-character ticker in the answer — nobody to pay"}, tx_hash=None,
                   latency_ms=int((time.time() - t0) * 1000))
        ledger_add(out); return out
    entry = BY_CODE.get(code)
    to = entry["address"] if entry else derive_address(code)   # an unverified code gets an address by the same rule; blocking is Privy's job
    out.update(to=to, verified_recipient=bool(entry), resolved_name=entry["name"] if entry else None)
    tx = send_tx(to, CFG["amount_wei"], data="0x" + code.encode().hex())   # the ticker rides along in tx data as a memo
    verdict = "exec" if tx["allowed"] else ("deny" if tx["denied_by_policy"] else "error")
    out.update(verdict=verdict, privy_status=tx["status"], privy_raw=tx["response"], privy_request=tx["request"],
               tx_hash=tx["hash"], explorer=(EXPLORER.get(CHAIN_ID, "") + tx["hash"]) if tx["hash"] and MODE == "privy" else None,
               latency_ms=int((time.time() - t0) * 1000), mode=MODE, t=time.time())
    ledger_add(out)
    return out


def settle(instruction, model):
    """POST /settle — pay the first company named in the instruction."""
    hits = find_company(instruction)
    if not hits:
        return 400, {"error": "no registry company name found in the instruction", "instruction": instruction,
                     "known": [r["name"] for r in REGISTRY]}
    return 200, settle_one(hits[0], model, instruction)


def settle_batch(instruction, model):
    """POST /settle/batch — pay every company named in the instruction. A hallucinated leg is denied alone; the rest settle (partial-failure isolation)."""
    hits = find_company(instruction)
    if not hits:
        return 400, {"error": "no registry company name found in the instruction", "instruction": instruction}
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


# ───────────────────────────── HTTP hooks (called from server.py) ─────────────────────────────
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
            h._json({"error": "model must be before or after"}, 400); return True
        if not instr:
            h._json({"error": "instruction is empty"}, 400); return True
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
            print(f"{r['name']:<12s} {r['code']}  {r['address']}  base model generated: {r.get('before_generation')}")
        print(f"→ {REGISTRY_PATH} ({len(rows)} entries)")
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
