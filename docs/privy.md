# Privy — 이름으로 라우팅되는 지급 (ETHOnline 2026)

코드는 `web/privy_pay.py`, 화면은 `web/privy.html` (탭 "Privy"), 레지스트리는 `web/registry.json`, 서버 훅은
`web/server.py` 세 줄. 표준 라이브러리만 쓴다. 작업 지시서는 `SPEC.md`, 제출문은 `SUBMISSION.md`.

## 무엇을 보여주나

같은 지갑·같은 정책·같은 코드 경로에서 **모델만 바꾼다.**

```
"태양기계에 대금 지급"
  [:8000 학습 전] → 033160 (환각) → 코드에서 파생된 주소가 검증 목록 밖 → Privy DENY  돈 안 나감
  [:8001 학습 후] → 116100 (정확) → 검증된 수취인                          → 체결 tx 0x8811fe45…
```

방어 두 겹, 서로 독립: 지식 교정(19% → 100%)은 우리 연구, 권한 강제(검증 주소가 아니면 서명 거부)는 Privy 정책 엔진.
판정(DENY/EXEC)은 Privy 응답에서만 파생하고, 우리 코드는 미검증 주소를 미리 막지 않는다.

## 구성

| 것 | 값 |
|---|---|
| Privy 앱 | Ainize |
| 에이전트 지갑 (정책 부착) | `0x7c704148b6B69F87694aa8f7A0691635D5426D59` (id `lrpuh81g2tc4jii8mfidplhd`) |
| 정책 | `sh2rb434k1bjfd00xmigksxw` — `to in [레지스트리 16주소] AND value lte 10000000000000` → ALLOW, 그 외 DENY |
| 체인 | Ethereum Sepolia `eip155:11155111`, RPC `https://ethereum-sepolia-rpc.publicnode.com` |
| 레지스트리 | 16종목, 모두 `results/chat-eval.json` 에서 학습 전 오답·학습 후 정답. `address = 0x + keccak256("ticker:"+code)[12:]` |

자격증명·id 는 `~/.privy/finance-demo.env` (600). 읽는 순서 `$PRIVY_ENV_FILE` → `web/privy.env` → `~/.privy/finance-demo.env`.
서버 시작 로그 첫 줄과 `/api/privy/status` 의 `env_file` 에 실제로 읽은 파일이 나온다.

## HTTP

| 경로 | 설명 |
|---|---|
| `POST /settle {instruction, model: before\|after}` | 지시문의 첫 회사에 지급 한 건. 응답 필드는 SPEC P0-3 |
| `POST /settle/batch {instruction, model}` | 지시문의 모든 회사에 지급. 환각한 건만 거부, 나머지 정산 (`legs`, `summary`) |
| `GET /api/privy/status` | 모드·지갑·정책·잔액·원장 |
| `GET /api/privy/registry` | 레지스트리 16개와 파생 규칙 |
| `POST /api/privy/try {method}` | 정책이 허용하지 않은 메서드 호출 (가스 없이 `policy_violation`) |

## 운영

```
python3 web/test_privy_pay.py                       # 네트워크 없는 테스트
python3 web/privy_pay.py registry --n 16             # 레지스트리 재생성 (실측 데이터에서)
python3 web/privy_pay.py policy                      # 레지스트리 주소로 allowlist 정책 갱신 (PATCH, id 유지)
python3 web/privy_pay.py settle "태양기계에 대금 지급" --model before   # → policy_violation
python3 web/privy_pay.py settle "태양기계에 대금 지급" --model after    # → tx hash
web/deploy.sh 8080                                   # 모드 확인 → 재시작 → 검증. MOCK 이면 멈춘다
```

레지스트리를 바꾸면 반드시 `policy` 를 다시 돌려 정책 allowlist 를 맞춘다.

## Privy 에서 실제로 확인한 것

- Basic 인증은 `app_id:app_secret`. Cloudflare 가 urllib 기본 UA 를 `403 error code 1010` 으로 막는다 → UA 필수.
- 허용되지 않은 메서드(`personal_sign`, `eth_signTransaction`)는 잔액과 무관하게 `{"code":"policy_violation"}`.
- `eth_sendTransaction` 은 시뮬레이션이 정책보다 먼저 돈다. 잔액 0 이면 정책 밖 전송도 "insufficient funds" 로 실패한다.
  화면은 이를 정책 거부와 구분한다(브로드캐스트 실패는 가드레일의 증거가 아니다).
- `PATCH /v1/policies/{id}` 로 룰을 갱신하면 지갑·id 를 바꾸지 않아도 된다. `to` 의 `in` 연산자가 동작한다.
- `caip2 eip155:296`(Hedera 테스트넷)도 받아들인다 — 별도 바운티(P2), P0 뒤에.
