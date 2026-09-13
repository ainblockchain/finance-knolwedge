# web/img/ — evidence screenshots (SPEC P0-5)

When these files exist in this folder (`web/img/`, under the `web/` directory the page is served from), the
Evidence section of `web/privy.html` replaces its placeholders with `<img>` automatically.

| file | what to capture | where |
|---|---|---|
| `etherscan-tx.png` | Sepolia transaction detail. **From must visibly be the agent wallet `0x7c704148b6B69F87694aa8f7A0691635D5426D59`** | https://sepolia.etherscan.io/tx/0x8811fe45dae8b5ee246ce4ea47cc641824e01fa59cd2147cddbf92d12c10dfab |
| `privy-policy.png` | Privy dashboard, policy view ("Verified identifier settlement", `to in […16]`, `value lte 10000000000000`) | dashboard.privy.io → Ainize → Policies → `sh2rb434k1bjfd00xmigksxw` |
| `deny-response.png` | `policy_violation` verbatim | **present** — rendered from real CLI output of `python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before` (the image says so; replace with a terminal screenshot if preferred) |

TODO(human): `etherscan-tx.png` and `privy-policy.png`. The dashboard needs a login, so they cannot be captured automatically.
