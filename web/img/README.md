# web/img/ — evidence (SPEC P0-5)

Files in this folder (`web/img/`, under the `web/` directory the page is served from) replace the placeholders in the
Evidence section of `web/privy.html` automatically.

| file | what it shows | source |
|---|---|---|
| `etherscan-tx.png` | Sepolia transaction `0x4c2e27c92246a7eea1337da9fb31ed5442baced5c55da353757fac55dc7ba085` — **From = agent wallet `0x7c704148b6B69F87694aa8f7A0691635D5426D59`**, 0.00001 ETH to `0x7DcfD5FCb39aD8E1298808b1d39F331cf39D14F8` (verified address of 닷밀 464580, one leg of the 3-company batch), status Success, block 11695599 | screenshot of sepolia.etherscan.io |
| `privy-policy.png` | Privy dashboard, policy `sh2rb434k1bjfd00xmigksxw` "Verified identifier settlement": `to in [16 addresses]`, `value lte 10000000000000` | screenshot of dashboard.privy.io |
| `privy-transactions.png` | Privy dashboard, Wallets → Transactions: every succeeded `eth_sendTransaction` from wallet `lrpuh81g2tc4jii8mfidplhd` on Sepolia. Denied requests do not appear — no signature was produced | screenshot of dashboard.privy.io |
| `deny-response.png` | `policy_violation` verbatim for the base model's hallucinated code | rendered from real CLI output (`python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before`); the image says so |

The settled transaction quoted in `SUBMISSION.md` (`0x8811fe45…`, the single-settlement run) is a different transaction
from the same wallet; both are on Sepolia under the same policy.
