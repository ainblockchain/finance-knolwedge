# img/ — 증거 스크린샷 (SPEC P0-5)

세 파일이 이 폴더에 들어오면 `web/privy.html` 의 §증거 자리가 자동으로 `<img>` 로 바뀐다.

| 파일 | 무엇을 찍나 | 어디서 |
|---|---|---|
| `etherscan-tx.png` | Sepolia 트랜잭션 상세. **From 이 에이전트 지갑 `0x7c704148b6B69F87694aa8f7A0691635D5426D59` 인 것이 보이게** | https://sepolia.etherscan.io/tx/0x8811fe45dae8b5ee246ce4ea47cc641824e01fa59cd2147cddbf92d12c10dfab |
| `privy-policy.png` | Privy 대시보드의 정책 화면 (Verified identifier settlement, `to in […16]`, `value lte 10000000000000`) | dashboard.privy.io → Ainize → Policies → `sh2rb434k1bjfd00xmigksxw` |
| `deny-response.png` | `policy_violation` 원문 터미널 캡처 | `python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before` |

TODO(사람이 채울 것): 세 파일 모두. 대시보드는 로그인이 필요해 자동으로 찍을 수 없다.
