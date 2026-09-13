# Privy — settlement routed by verified identifiers (ETHOnline 2026)

Code: `web/privy_pay.py`. Page: `web/privy.html` (tab "Privy"). Registry: `web/registry.json`. Server hooks: three lines in
`web/server.py`. Standard library only. Work order: `SPEC.md`. Submission text: `SUBMISSION.md`.

## What it shows

Same wallet, same policy, same code path — **only the model changes.**

```
"태양기계에 대금 지급"  (= "pay 태양기계")
  [:8000 base model]    → 033160 (hallucinated) → address derived from that code is outside the allowlist → Privy DENY   nothing leaves
  [:8001 trained model] → 116100 (correct)      → verified recipient                                        → settled  tx 0x8811fe45…
```

Two independent layers: knowledge correction (19% → 100%) is our research; authority enforcement (no signature unless
the address is verified) is Privy's policy engine. The verdict (DENY/EXEC) is derived only from Privy's response; our
code never pre-blocks an unverified address.

## Setup

| item | value |
|---|---|
| Privy app | Ainize |
| agent wallet (policy attached) | `0x7c704148b6B69F87694aa8f7A0691635D5426D59` (id `lrpuh81g2tc4jii8mfidplhd`) |
| policy | `sh2rb434k1bjfd00xmigksxw` — `to in [16 registry addresses] AND value lte 10000000000000` → ALLOW, everything else DENY |
| chain | Ethereum Sepolia `eip155:11155111`, RPC `https://ethereum-sepolia-rpc.publicnode.com` |
| registry | 16 tickers, all wrong-before / right-after per `results/chat-eval.json`. `address = 0x + keccak256("ticker:"+code)[12:]` |

Credentials and ids live in `~/.privy/finance-demo.env` (mode 600), outside every checkout. Lookup order:
`$PRIVY_ENV_FILE` → `web/privy.env` → `~/.privy/finance-demo.env`. The first server log line and `env_file` in
`/api/privy/status` name the file actually read.

## HTTP

| path | description |
|---|---|
| `POST /settle {instruction, model: before\|after}` | one payment to the first company in the instruction; response fields per SPEC P0-3 |
| `POST /settle/batch {instruction, model}` | one payment per company in the instruction; hallucinated legs are denied alone (`legs`, `summary`) |
| `GET /api/privy/status` | mode, wallet, policy, balance, ledger |
| `GET /api/privy/registry` | the 16 entries and the derivation rule |
| `POST /api/privy/try {method}` | call a method the policy does not allow (`policy_violation` without gas) |

## Operations

```
python3 web/test_privy_pay.py                         # offline tests: keccak vectors, derivation, resolve
python3 web/privy_pay.py registry --n 16               # rebuild the registry from measured data
python3 web/privy_pay.py policy                        # update the allowlist policy in place (PATCH, same id)
python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before   # → policy_violation
python3 web/privy_pay.py settle "태양기계에 대금 지급" --model after    # → tx hash
web/deploy.sh 8080                                     # check mode → restart → verify; refuses MOCK
```

After changing the registry, always rerun `policy` so the allowlist matches.

## What we verified against Privy

- Basic auth is `app_id:app_secret`. Cloudflare answers `403 error code 1010` to urllib's default User-Agent → set one.
- Methods the policy does not allow (`personal_sign`, `eth_signTransaction`) return `{"code":"policy_violation"}` regardless of balance.
- `eth_sendTransaction` is simulated before the policy runs: with a zero balance an out-of-policy transfer fails as
  "insufficient funds". The page distinguishes that from a policy denial (a broadcast failure is not guardrail evidence).
- `PATCH /v1/policies/{id}` updates rules in place — no wallet recreation, no gas transfer. The `in` operator on `to` works.
- caip2 `eip155:296` (Hedera testnet) is accepted — separate bounty (P2), only after P0.
