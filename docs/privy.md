# Privy 서버 지갑 + 정책 엔진 — "사실을 사는 에이전트" 결제 루프

ETHGlobal 제출용 Privy 통합. 코드는 `web/privy_pay.py` 하나, 화면은 `web/privy.html` (탭 "Privy"),
서버 훅은 `web/server.py` 네 줄. 표준 라이브러리만 쓴다.

## 핵심 서사

Privy 요건은 **"Privy must be integrated as a core product component"** 이다. 지갑 로그인을 붙인 것으로는 안 된다.

> 모델이 지어내듯, 에이전트도 지어낸다. 우리 원본 모델은 종목코드의 82% 를 환각한다. 그래서 에이전트는 검증된
> 사실을 건당 구매한다. 그런데 환각하는 에이전트에게 지갑을 쥐여주면 예산도 환각한다. Privy Policy Engine 이 그
> 가드레일이다 — 지출 상한과 수취인 화이트리스트가 TEE 안에서 강제되므로, 에이전트가 오작동해도 트레저리를 털 수 없다.

"사실을 사는 에이전트" + "그 에이전트를 통제하는 정책 지갑". Privy 없이는 성립하지 않는 구조다.

## 흐름

```
[사용자/평가 하네스] ─ "픽셀플러스 종목코드?" ─► [Agent]
  [Agent] ─ GET /api/fact?q=… ──────────────────────────► 402 {price_wei, payTo, chainId, caip2, nonce}
  [Agent] ─ POST api.privy.io/v1/wallets/{id}/rpc eth_sendTransaction{to: payTo, value: price, data: 0x+nonce}
            └ 정책: to == FACT_PROVIDER_ADDRESS AND value <= FACT_PRICE_WEI → ALLOW, 그 외 DENY (TEE 안에서 평가)
  [Agent] ─ GET /api/fact?q=…  X-Payment-Receipt: <tx hash> ─► 서버가 체인에서 검증 → 학습된 모델의 답
```

서버 측 영수증 검증(`verify_receipt`): `to` 가 공급자, `value` 가 가격 이상, `from` 이 에이전트의 Privy 지갑, tx `data`
에 서버가 내준 nonce, receipt `status == 0x1`. 같은 hash 는 한 번만. **한 결제 = 한 조회.**

## 90분 안에 LIVE 로 만드는 순서

### 0. 자격증명 (5분)

dashboard.privy.io → 앱 생성 (웹 주소 칸은 `https://finance-demo.ainetwork.ai`) → **App ID** 와 **App secret** 을
**`~/.privy/finance-demo.env`** 에 (템플릿이 이미 있다. 권한 600):

```
PRIVY_APP_ID=…
PRIVY_APP_SECRET=…
```

이 파일은 어떤 체크아웃에도 속하지 않으므로 원래 체크아웃(8080)·워크트리(8085) 어디서 띄워도 같은 값을 읽는다.
읽는 순서: `$PRIVY_ENV_FILE` → `web/privy.env` → `~/.privy/finance-demo.env`. 서버 시작 로그 첫 줄과
`/api/privy/status` 의 `env_file` 에 실제로 읽은 파일이 나온다.

인증은 `Authorization: Basic base64(APP_ID:APP_SECRET)` + `privy-app-id: APP_ID` 헤더다. (계획 초안의
`echo -n ":$PRIVY_API_KEY"` 는 사용자명이 비어 있어 401 이 난다 — `privy_pay.py` 는 올바른 형태로 보낸다.)

### 1. 스파이크 — Hedera 로 나가는가 (10분, 선택)

정책·지갑을 만든 뒤 (아래 2번) 한 줄:

```
python3 web/privy_pay.py spike --caip2 eip155:296
```

- tx 가 나가거나 "insufficient funds" 류 → Hedera 테스트넷 지원. `CHAIN_CAIP2=eip155:296`,
  `RPC_URL=https://testnet.hashio.io/api` 로 바꾸면 Privy 지갑 하나로 Hedera x402 결제까지 처리된다.
- "unsupported caip2/chain" 류 → Base Sepolia(`eip155:84532`, 기본값) 로 가고 Hedera 레일은 따로 둔다.

### 2. 정책 → 지갑 (10분)

정책을 먼저 만들고 그 정책 id 를 붙여 지갑을 만든다 (지갑을 다시 만들 필요가 없는 순서).

```
python3 web/privy_pay.py setup --provider 0x<사실 공급자 수취 주소> --write
```

`~/.privy/finance-demo.env` 에 `PRIVY_POLICY_ID`, `PRIVY_WALLET_ID`, `PRIVY_WALLET_ADDRESS`, `FACT_PROVIDER_ADDRESS`, `FACT_PRICE_WEI`
가 덧붙는다. 출력된 지갑 주소에 **테스트넷 ETH 를 파우셋으로 충전**한다 (Base Sepolia: 코인베이스/알케미 파우셋).
요건은 "완결된 financial flow 1건" 이지 USDC 가 아니므로 네이티브 ETH 전송으로 간다.

정책 원문 (Privy 는 명시적으로 허용하지 않은 것은 전부 거부):

```json
{"version":"1.0","name":"Fact-purchase budget guardrail","chain_type":"ethereum",
 "rules":[{"name":"Allow micropayments to fact provider only","method":"eth_sendTransaction","action":"ALLOW",
   "conditions":[{"field_source":"ethereum_transaction","field":"to","operator":"eq","value":"<FACT_PROVIDER_ADDRESS>"},
                 {"field_source":"ethereum_transaction","field":"value","operator":"lte","value":"10000000000000"}]}]}
```

### 3. 배포 (5분)

```
web/deploy.sh 8080          # 모드 확인 → 기존 서버 종료 → 재시작 → /api/privy/status 로 검증
```

MOCK 이면 서버를 건드리지 않고 멈춘다 (`--allow-mock` 을 주면 그대로 올린다). LIVE 인데 지갑이 없으면
setup 을 하라고 멈춘다. 끝나면 탭 "Privy" 의 배지가 **LIVE · Privy 연결됨** 이어야 한다.

8080 은 `/mnt/newdata/qwen3.8` 체크아웃에서 띄운다. 이 브랜치를 그쪽에 머지한 뒤 그 디렉토리에서 `web/deploy.sh 8080`.

### 4. 증거 확보 (20분) — 영상·제출문에 들어갈 것

화면 §2 의 버튼 순서대로 누르면 된다. 모두 요청·응답 원문이 펼쳐진다.

1. **정책 거부 1회** — ① 한도 초과 또는 ② 다른 수취인 → `DENY` 와 Privy 응답 원문. **영상에서 가장 강한 장면.**
2. **정상 결제** — §1 "에이전트 실행" → tx hash 와 익스플로러 링크 (LIVE 에서 자동으로 붙는다).
3. **결제 전 답(환각) vs 결제 후 답(정답)** — §1 의 좌우 카드가 그 화면이다.
4. §3 체크리스트가 전부 ✔ 인 화면.

CLI 로도 같은 것을 만들 수 있다:

```
python3 web/privy_pay.py try --to 0x000000000000000000000000000000000000dead   # DENY
python3 web/privy_pay.py try --value-wei 1000000000000000                          # DENY (한도 초과)
python3 web/privy_pay.py agent "픽셀플러스 종목코드 알려줘. 숫자만."                 # 402 → 결제 → 답
```

## HTTP

| 경로 | 설명 |
|---|---|
| `GET /api/fact?q=…` | 영수증 없으면 402 + 결제 조건. `X-Payment-Receipt: <tx hash>` 가 맞으면 학습된 모델의 답 |
| `GET /api/privy/status` | 모드·지갑·정책·잔액·영수증·거부 기록·체크리스트 |
| `POST /api/privy/agent {q}` | 에이전트 한 바퀴. 단계별 요청·응답 원문을 `steps` 로 |
| `POST /api/privy/try {to, value_wei}` | 정책 밖 트랜잭션 시도 (가드레일 데모) |

## mock 모드

`PRIVY_APP_ID` 가 없으면 mock 이다. 정책 판정을 이 프로세스가 같은 규칙으로 흉내 내고, 가짜 hash 를 내주며,
화면·응답 어디에나 MOCK 이라고 적는다. 화면과 흐름을 점검하는 용도이고 **제출 증거가 될 수 없다.** 체크리스트도
LIVE 기록으로만 체크된다.

## 제출문 초안 (Privy 칸)

**How you used Privy**
Our project sells verified facts to AI agents. The consuming agent's treasury is a Privy server wallet created via the
REST API (POST /v1/wallets), and every fact purchase is an eth_sendTransaction through /v1/wallets/{id}/rpc — one
onchain payment per verified lookup. Tx: 0x…

Privy is not an add-on here; it is the safety layer the product requires. Our thesis is that a 51B model hallucinates
82% of Korean ticker codes — and an agent built on such a model hallucinates its budget too. We attached a Privy policy
(POST /v1/policies, id …) that ALLOWs eth_sendTransaction only when to == <provider> and value <= 0.00001 ETH, with
everything else denied by default. Because the policy engine evaluates inside a secure enclave, the spend cap holds even
if our agent process is compromised or prompt-injected. Demo video m:ss shows a denied over-limit transaction, then an
allowed micropayment.

**How Privy improves the UX**
An autonomous agent cannot be asked to manage a seed phrase, and a raw hot key gives it unlimited authority. Privy gave
us a keyless, TEE-backed wallet in one API call and turned "how much can this agent spend, and on what" into declarative
config instead of application code we'd have to get right under hackathon time pressure.

**Feedback** — 비워두지 말 것. 실제로 막힌 지점을 3~4줄. 후보: Basic 인증의 사용자명/비밀번호가 무엇인지 quickstart
에서 한 번에 안 보였다 · 정책 조건이 per-exact-value 라 "이 주소 목록" 같은 집합 조건은 룰을 여러 개 써야 했다 ·
정책 거부 응답의 에러 형태가 문서에 예시로 없어 화면에 어떻게 보여줄지 실제로 한 번 맞아 보고 알았다 · caip2
지원 체인 목록이 한 페이지에 정리돼 있으면 스파이크가 필요 없었다.

## 체크리스트

- [ ] Privy 지갑 최소 1개 생성 (요건)
- [ ] 완결된 financial flow 1건 — 온체인 결제로 사실 1건 구매 (요건)
- [ ] Privy control 사용 — 정책이 지갑에 붙어 있음 (B2B 트랙 요건, Financial Flow 에서도 가점)
- [ ] 동작하는 데모 + 공개 소스 (요건)
- [ ] "Privy 가 어떻게 UX 를 개선했는지" (요건)
- [ ] 영상 2~4분 (ETHGlobal 하드 제한) — 거부 장면을 앞에
