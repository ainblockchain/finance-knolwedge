# SUBMISSION — ETHOnline 2026 · Privy (Financial Flow / B2B)

**Project**: Bake your own memory — settlement routed by verified identifiers
**Live demo**: https://finance-demo.ainetwork.ai/privy.html (tab "Privy")
**Repo**: this repository (`web/privy_pay.py`, `web/privy.html`, `web/registry.json`, `SPEC.md`)

## How you used Privy

Our agent settles payments routed by Korean stock ticker. The ticker is the routing key: resolve the company name to
its 6-digit code, and the code maps to the recipient address (`0x + keccak256("ticker:" + code)[12:]`).

The failure mode we attack is not overspending — it is misrouting. Our base model answers with a well-formed but wrong
6-digit code 81% of the time (19.0% → 100.0% template accuracy after our 0.084%-parameter edit; 2,237 of 2,761 tickers
move from wrong to right; chat-format accuracy 13% → 97% on a 60-sample eval). In a lookup that is one wrong answer.
In a payment it is money sent to a different real company, irreversibly.

So the system defends in two independent layers. Our research fixes the model. **Privy's policy engine refuses to sign
when the identifier is not verified.** Policy `sh2rb434k1bjfd00xmigksxw` ALLOWs `eth_sendTransaction` only when `to`
is in the verified-recipient allowlist (16 addresses derived from verified tickers) and `value` is at or below the
per-settlement cap (10,000,000,000,000 wei = 0.00001 ETH); everything else is denied by default, evaluated inside a
secure enclave. The agent's treasury is a Privy server wallet `0x7c704148b6B69F87694aa8f7A0691635D5426D59` (wallet id
`lrpuh81g2tc4jii8mfidplhd`) created via `POST /v1/wallets` with `policy_ids`; every settlement runs through
`POST /v1/wallets/{id}/rpc` on Ethereum Sepolia (`eip155:11155111`). The policy was updated in place with
`PATCH /v1/policies/{id}` when we moved from a single recipient to the allowlist.

The demo runs the same instruction, the same wallet and the same policy against two models:

| instruction: "태양기계에 대금 지급" | model answer | recipient | Privy | result |
|---|---|---|---|---|
| base model (:8000) | `033160` (wrong; truth `116100`) | derived from the wrong code → not in allowlist | `400 {"error":"RPC request denied due to policy violation","code":"policy_violation"}` | **DENY** — nothing reaches the chain |
| edited model (:8001) | `116100` (correct) | verified recipient | `200 {"data":{"hash":"0x8811fe45…"}}` | **EXEC** — settled on Sepolia |

Settled tx: https://sepolia.etherscan.io/tx/0x8811fe45dae8b5ee246ce4ea47cc641824e01fa59cd2147cddbf92d12c10dfab
Denied: `{"error":"RPC request denied due to policy violation","code":"policy_violation"}`
Same code path, different model.

This is a B2B settlement workflow: one instruction can disburse to multiple listed companies at once
(`POST /settle/batch`), and a single bad identifier is contained to its own leg while the others settle.

## How Privy improves the UX

An autonomous agent cannot be asked to manage a seed phrase, and a raw hot key gives it unlimited authority. Privy gave
us a keyless, TEE-backed wallet in one API call and turned "who may this agent pay, and how much" into declarative
policy instead of application code we would have to get right under deadline. Because the deny happens at signing, the
agent's own code never has to be trusted to pre-check recipients — and in our demo it deliberately does not.

## Feedback

1. `api.privy.io` sits behind Cloudflare and rejects Python's default `urllib` User-Agent with `403 error code: 1010`.
   Nothing in the REST quickstart mentions it; a one-line note ("send a User-Agent") would have saved 20 minutes.
2. The quickstart's Basic-auth example was easy to misread as `":<secret>"`; an explicit `curl -u "$APP_ID:$APP_SECRET"`
   next to the header list would remove the ambiguity.
3. `eth_sendTransaction` is simulated *before* the policy is evaluated. With an unfunded wallet a clearly out-of-policy
   transfer fails as "insufficient funds", not `policy_violation`, which makes a policy demo look broken until gas
   arrives. Documenting the evaluation order (or evaluating policy first) would help.
4. The `policy_violation` payload does not say which rule or condition failed. For audit trails and for showing a judge
   *why* a signature was refused, a `rule_id`/`condition` field in the error would be valuable.
5. Positive: `PATCH /v1/policies/{id}` updating rules in place, and the `in` operator on `to`, both worked first try —
   we never had to recreate the wallet or move gas.

## Verification (do not trust this file, check)

- `python3 web/test_privy_pay.py` — keccak vectors, deterministic derivation, `resolve("태양기계") → 116100`
- `python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before` → `policy_violation`
- `python3 web/privy_pay.py settle "태양기계에 대금 지급" --model after` → tx hash on Sepolia, from = agent wallet
