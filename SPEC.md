# SPEC.md — ETHOnline 2026 제출 작업 지시서

> 이 문서는 Claude Code가 읽고 실행하는 작업 지시서다.
> 실행 방법: 레포 루트에 이 파일을 두고 `claude` 실행 후
> **"SPEC.md 읽고 P0부터 순서대로 실행해"** 라고 지시.

---

## 0. 절대 규칙 (먼저 읽을 것)

1. **마감은 2026-09-13 12:00 EDT = 2026-09-14 01:00 KST.** 지각 제출은 받지 않는다.
   현재 시각을 먼저 확인하고, 남은 시간이 P0 완료에 부족하면 **P1/P2를 버리고 P0만 해라.**
2. **값을 지어내지 마라.** 트랜잭션 해시, 정책 id, 지갑 주소, 정확도 수치는 전부 검증 가능한 값이다.
   실제로 실행해서 얻은 값만 코드·문서·제출문에 넣는다. 모르면 `TODO(사람이 채울 것)` 로 남긴다.
3. **커밋을 자주 쪼개라.** ETHGlobal 규칙상 "큰 변경 한 방 커밋"은 실격 사유다.
   각 태스크 완료마다 커밋한다.
4. **작동하던 것을 깨지 마라.** 기존 데모 페이지(채팅/벤치마크/추론 추적/트레이닝 기법)는 손대지 않는다.
   이번 작업 범위는 Privy 데모 경로 하나다.
5. **비밀키를 커밋하지 마라.** `PRIVY_API_KEY` 등은 `.env` 로 빼고 `.gitignore` 확인.
6. 판단이 필요한 갈림길(예: 파우셋이 막힘, Privy가 특정 체인 거부)에서는
   **혼자 우회하지 말고 멈추고 사람에게 물어라.** 시간이 없으므로 잘못된 방향으로 30분 쓰는 게 제일 비싸다.

---

## 1. 배경 (왜 이걸 하는가)

**프로젝트**: `Bake your own memory` — Qwen3.8-Flash-Next(125B MoE)에
한국 상장사 2,761개 종목코드를 전체 파라미터의 0.084%만 수정해 주입.

**측정된 성과** (이미 있음, 건드리지 말 것):

| 지표 | 학습 전 | 학습 후 |
|---|---|---|
| 템플릿 프롬프트 정확도 | 19.0% (524/2761) | 100.0% (2761/2761) |
| 채팅 형식 정확도 | 13% (8/60) | 97% (58/60) |
| 실패→성공 전이 | — | 2,237 종목 |
| 바뀐 파라미터 | — | 0.084% (백본 불변) |

**현재 Privy 데모의 문제**: 에이전트가 "사실을 건당 구매"하고, Privy 정책 엔진이
"과소비"를 막는 구조다. 그런데 이 에이전트는 고정 주소에 고정 금액을 보내므로
**과소비할 방법이 없다.** 정책 엔진이 존재하지 않는 리스크를 막고 있어서,
심사자에게는 "Privy 튜토리얼 + 비유"로 읽힌다.

**바꿀 프레이밍**:

> 에이전틱 금융의 실제 사고는 "에이전트가 돈을 너무 많이 씀"이 아니라
> **"에이전트가 식별자를 환각해서 엉뚱한 상대에게 돈을 보냄"** 이다.
> 원본 모델은 픽셀플러스에 `203366` 같은 **형식은 완벽하고 값만 틀린** 코드를 답한다.
> 조회라면 오답 한 건이지만, 결제에서는 실재하는 다른 회사로 송금이 나가고 되돌릴 수 없다.

따라서 방어는 두 겹이고, 두 계층은 **독립**이다:

| 계층 | 역할 | 담당 |
|---|---|---|
| 1. 지식 교정 | 모델이 올바른 식별자를 내게 만든다 (19%→100%) | 우리 연구 |
| 2. 권한 강제 | 검증된 식별자에 매핑된 주소가 아니면 **서명 자체를 거부** | Privy Policy Engine (TEE) |

**이기는 데모 한 장면**: 같은 지갑, 같은 정책, 같은 코드 — **모델만 교체**.

```
"태양기계에 대금 지급"
  [:8000 학습 전] → 033240 (환각) → 화이트리스트 밖 → Privy DENY  ❌ 돈 안 나감
  [:8001 학습 후] → 116100 (정확) → 검증된 수취인   → 체결 tx 0x…  ✅
```

---

## 2. 현재 상태 (먼저 파악할 것)

작업 시작 전 다음을 확인하고 실제 경로를 기록해라:

- [ ] Privy 데모 페이지 파일 (`privy.html` 로 서빙됨, 포트 8085)
- [ ] Fact API / 402 미들웨어 / 영수증 검증이 있는 백엔드 파일
- [ ] Privy REST 호출부 (지갑 생성·`eth_sendTransaction`)
- [ ] 종목코드 정답 데이터 소스 (벤치마크가 쓰는 것, 2,761행)
- [ ] 학습 전 모델 서버 `:8000`, 학습 후 `:8001` 호출 방식

**이미 만들어져 동작하는 것 (재사용, 재구현 금지)**:

```
Privy app         : 생성됨
Agent wallet      : 0x7c704148b6b69f87694aa8f7a0691635d5426d59
  wallet id       : lrpuh81g2tc4jii8mfidplhd
Policy id         : sh2rb434k1bjfd00xmigksxw
Fact provider     : 0xa68f5694f69c0094173e9f6d213ce5e2fe008554
Chain             : eip155:11155111 (Ethereum Sepolia)
건당 상한          : 10000000000000 wei = 0.00001 ETH
잔액              : ~0.0499 ETH
확인된 동작        : 결제 1건 성공, policy_violation 거부 4건
```

---

## 3. P0 — 반드시 끝낼 것

### P0-1. 검증된 티커 → 수취인 레지스트리 (목표 40분)

**무엇**: 종목코드가 결제 라우팅 키가 되도록 만든다.

1. 데모용 종목 **12~20개**를 고른다. 반드시 **벤치마크에서 학습 전에 틀렸던 종목**을 포함할 것
   (전후 대비가 나와야 한다). 확인된 사용 가능 조합:

   | 회사명 | 정답 | 학습 전 생성 |
   |---|---|---|
   | 태양기계 | 116100 | 033240 |
   | 엑시온그룹 | 069920 | 057680 |
   | 성원에너텍 | 340930 | 095570 |
   | 닷밀 | 464580 | 439580 |
   | KR모터스 | 000040 | 015880 |
   | 픽셀플러스 | 087600 | (환각) |

2. 각 종목코드에 **서로 다른 결정론적 주소**를 배정한다.
   `address = "0x" + keccak256("ticker:" + code)[12:]` 같이 코드에서 재현 가능하게 파생.
   실제 소유자가 없어도 된다 — 중요한 건 **코드가 곧 수취인**이라는 점이다.
3. `registry.json` (또는 동등물)으로 저장. 필드: `name`, `code`, `address`, `market`.
4. 파생 함수에 유닛 테스트 하나: 같은 코드 → 항상 같은 주소, 다른 코드 → 다른 주소.

**완료 기준**: `resolve("태양기계") → 116100 → 0x…` 가 코드에서 한 줄로 나온다.

### P0-2. Privy 정책을 allowlist 기반으로 교체 (목표 25분)

기존 정책은 단일 주소만 허용한다. 레지스트리의 주소 목록 전체를 허용하도록 바꾼다.

```bash
curl -X POST https://api.privy.io/v1/policies \
  -H "Authorization: Basic $(printf ':%s' "$PRIVY_API_KEY" | base64)" \
  -H "privy-app-id: $PRIVY_APP_ID" \
  -H 'Content-Type: application/json' \
  -d '{
    "version":"1.0",
    "name":"Verified identifier settlement",
    "chain_type":"ethereum",
    "rules":[{
      "name":"Allow settlement to verified tickers only",
      "method":"eth_sendTransaction",
      "action":"ALLOW",
      "conditions":[
        {"field_source":"ethereum_transaction","field":"to",
         "operator":"in","value":["0x…","0x…","…레지스트리 주소 전체…"]},
        {"field_source":"ethereum_transaction","field":"value",
         "operator":"lte","value":"10000000000000"}
      ]
    }]
  }'
```

주의사항:

- 정책 엔진은 **명시적으로 허용하지 않은 메서드는 전부 거부**가 기본값이다. 이 성질이 데모의 핵심이므로 룰을 더 넣지 마라.
- 정책은 **지갑 생성 시 `policy_ids`** 로 붙는다. 기존 지갑에 붙는지 먼저 확인하고,
  안 되면 새 지갑을 만들고 **가스를 옮긴 뒤** 상단 상수를 갱신해라.
- 새 policy id / wallet 주소가 생기면 **README·페이지·제출문 전부 갱신**한다. 옛 값이 남으면 안 된다.

**완료 기준**: 레지스트리 안 주소로 상한 이하 전송 → 200. 레지스트리 밖 주소 → `{"code":"policy_violation"}`.

### P0-3. 제품을 "사실 구매"에서 "이름으로 라우팅되는 지급"으로 (목표 50분)

**엔드포인트 하나**: `POST /settle  { instruction: "태양기계에 대금 지급", model: "before"|"after" }`

처리 순서:

1. `model` 에 따라 `:8000`(학습 전) 또는 `:8001`(학습 후)에 종목코드를 질의한다.
   기존 채팅 프롬프트 형식을 그대로 쓴다 (`"{회사명} 종목코드 알려줘. 숫자만."`).
2. 응답에서 첫 6자리 코드를 파싱한다.
3. 레지스트리에서 그 코드의 주소를 찾는다. 없으면 **결정론적 파생으로 주소를 만든다**
   (미검증 주소 → 정책 밖 → 거부됨). **여기서 절대 미리 차단하지 마라.**
   거부는 우리 코드가 아니라 **Privy 정책 엔진이** 해야 한다. 그게 데모의 전부다.
4. Privy `POST /v1/wallets/{wallet_id}/rpc` 로 `eth_sendTransaction` 전송.
5. 응답을 그대로 반환한다:

```json
{
  "instruction": "…", "model": "before",
  "company": "태양기계", "code": "033240",
  "expected_code": "116100", "correct": false,
  "to": "0x…", "verified_recipient": false,
  "verdict": "deny",
  "privy_status": 403,
  "privy_raw": {"error":"RPC request denied due to policy violation","code":"policy_violation"},
  "tx_hash": null, "latency_ms": 812
}
```

`verdict` 는 **Privy 응답에서 파생**한다. 우리가 판단해서 넣는 게 아니다.

**완료 기준**: 같은 instruction 을 `before` / `after` 로 호출하면 각각 `deny` / `exec` 가 나온다.
`after` 호출은 Sepolia에 실제 tx를 남긴다.

### P0-4. UI 교체 (목표 40분)

첨부된 `privy_terminal.html` 이 시각 타깃이자 셸이다. 이걸 기존 `privy.html` 자리에 놓고 배선한다.

- 하단 후크 두 개에만 연결한다:
  - `render({ side:'before'|'after', code, state:'deny'|'exec', note, raw })`
  - `appendRow({ time, name, model, code, amount, state, reason, hash })`
- `#run` 클릭 시 `:8000`/`:8001` 을 **동시에**(`Promise.all`) 호출하고 양쪽 레인을 갱신한다.
- `raw` 에는 **요청·응답 원문을 그대로** 넣는다. 요약하지 마라. 원문이 신뢰의 근거다.
- 상단 레일의 지갑 주소·정책 id·잔액은 서버에서 실제 값으로 채운다 (하드코딩 금지).

**디자인 토큰 — 바꾸지 말 것** (이미 셸에 들어있음):

```
ground #151110 / surface #1D1817 / line #332B28 / ink #EFE8E4 / ink-3 #7E716B
exec (체결·통과) #E8483C   ← 한국 시장 상승색
deny (거부·차단) #3E82D6   ← 한국 시장 하락색
type: IBM Plex Sans KR (UI) + IBM Plex Mono (수치·코드), tabular-nums
```

**금지**: ASCII 아트 다이어그램, 이모지 섹션 마커, 모든 블록에 같은 radius/그림자,
스톡 이미지·그라디언트·3D 렌더, 제출 체크리스트를 페이지에 노출하는 것(README로 옮길 것).

### P0-5. 증거 수집 (목표 15분)

`img/` 에 스크린샷 3장을 넣고 §증거의 `.shot-empty` div를 `<img>` 로 교체:

1. `etherscan-tx.png` — Sepolia 트랜잭션 상세. **from이 에이전트 지갑인 게 보이게.**
2. `privy-policy.png` — Privy 대시보드 정책 화면.
3. `deny-response.png` — `policy_violation` 원문 터미널 캡처.

### P0-6. README + 제출문 (목표 20분)

`README.md` 에: 셋업 방법, 아키텍처, 결제 플로우, **AI 툴 사용 명시**
(어느 파일/어느 부분에 썼는지. spec-driven 워크플로우를 썼다면 이 SPEC.md 를 레포에 포함할 것 — 규칙상 요구됨).

`SUBMISSION.md` 에 아래 초안을 실제 값으로 채워 넣는다. **Privy 트랙 2개(Financial Flow / B2B) 모두에 읽히게** 쓸 것 — 한 파트너를 고르면 그 파트너의 모든 트랙에 자동 지원된다.

```
Our agent settles payments routed by Korean stock ticker. The ticker is the
routing key: resolve the company name to its 6-digit code, and the code maps
to the recipient address.

The failure mode we attack is not overspending — it is misrouting. Our base
model answers with a well-formed but wrong 6-digit code 81% of the time
(19.0% → 100.0% after our 0.084%-parameter edit; 2,237 of 2,761 tickers move
from wrong to right). In a lookup that is one wrong answer. In a payment it is
money sent to a different real company, irreversibly.

So the system defends in two independent layers. Our research fixes the model.
**Privy's policy engine refuses to sign when the identifier is not verified.**
Policy `sh2rb434k1bjfd00xmigksxw` ALLOWs `eth_sendTransaction` only when `to`
is in the verified-recipient allowlist and `value` is at or below the per-
settlement cap; everything else is denied by default, evaluated inside a secure
enclave. The agent's treasury is a Privy server wallet (`0x7c70…6d59`) created
via `POST /v1/wallets`; every settlement runs through
`POST /v1/wallets/{id}/rpc` on Ethereum Sepolia.

The demo runs the same instruction, the same wallet and the same policy against
two models. The base model's hallucinated ticker resolves to an unverified
recipient and Privy denies the signature — no transaction reaches the chain.
The edited model's ticker resolves to a verified recipient and settles on-chain.
Same code path, different model. Settled tx: 0x____  Denied: {"code":"policy_violation"}

This is a B2B settlement workflow: one instruction can disburse to multiple
listed companies at once, and a single bad identifier is contained to its own
leg while the others settle.

How Privy improves the UX: an autonomous agent cannot be asked to manage a seed
phrase, and a raw hot key gives it unlimited authority. Privy gave us a keyless,
TEE-backed wallet in one API call and turned "who may this agent pay, and how
much" into declarative policy instead of application code we would have to get
right under deadline.

Feedback: TODO — 실제로 막혔던 지점, 문서에서 부족했던 부분, 개선 제안 3~4줄.
          비워두지 마라. 파트너 심사에서 실제로 읽는다.
```

---

## 4. P1 — 시간 남으면

### P1-1. 3건 일괄 지급 (30분)
한 지시로 3개 종목에 동시 지급. 하나가 환각이면 **그 건만 거부되고 나머지는 정산**.
단건 이체가 아니라 진짜 "flow"가 되고 부분 실패 격리를 보여준다. B2B 트랙 적합도가 올라간다.

### P1-2. 벤치마크 링크 (10분)
Privy 페이지 히어로의 수치에서 기존 `bench.html` 로 링크. 주장에 근거를 붙인다.

---

## 5. P2 — 손대지 마라 (시간 남아도 마지막)

- **USDC 전환**: ERC-20이면 정책을 `ethereum_calldata` 로 `transfer` 인자에 걸어야 한다. 최소 1시간. **90분 이상 남을 때만.** 네이티브 ETH여도 "정산"으로 서술하면 요건은 충족된다.
- Hedera x402 / Bazantic 게이트웨이: 별도 바운티. P0가 끝나기 전엔 시작하지 마라.

---

## 6. 제출 전 최종 점검

- [ ] 레포 public, 커밋 히스토리가 여러 개로 쪼개져 있음
- [ ] `.env` 가 커밋되지 않음, API 키 노출 없음
- [ ] 페이지·README·제출문의 policy id / 지갑 주소 / tx 해시가 **전부 실제값이고 서로 일치**
- [ ] tx 해시를 sepolia.etherscan.io 에서 조회했을 때 from이 에이전트 지갑
- [ ] AI 툴 사용처가 README에 명시됨, SPEC.md가 레포에 포함됨
- [ ] 데모 영상 **2~4분** (파트너 페이지엔 5분이라 쓰여 있지만 ETHGlobal 업로드가 4분 초과를 거부한다)
  - 720p 이상 / 배속 금지 / AI 음성 내레이션 금지 / 휴대폰 촬영 금지 / 음악+자막만으로 설명 금지
  - **첫 30초를 A/B 한 장면으로 시작할 것.** 거부되는 장면이 성공하는 장면보다 중요하다
- [ ] ETHGlobal 폼: Project details → Images → Tech stack → **Select prizes(Privy 선택)** → Video → Final
- [ ] "How it's made" 에 파트너 기술이 아키텍처 어디에 있는지 서술

---

## 7. 실패 시 판단 기준

| 상황 | 대응 |
|---|---|
| Sepolia 파우셋이 막힘 | 기존 지갑에 0.0499 ETH 있음. 새 지갑 만들지 말고 기존 지갑 재사용 |
| Privy가 정책을 기존 지갑에 못 붙임 | 새 지갑 생성 + 가스 이체. 상수 갱신 잊지 말 것 |
| `:8000` 또는 `:8001` 서버가 죽음 | **사람에게 즉시 알려라.** A/B가 이 제출물의 전부다 |
| P0-3까지 끝냈는데 1시간 미만 남음 | P0-4 UI를 버리고 기존 페이지에 A/B만 붙여라. **기능 > 디자인** |
| 30분 미만 남음 | 코딩 중단. 영상 촬영과 폼 제출에 전부 쓴다 |
