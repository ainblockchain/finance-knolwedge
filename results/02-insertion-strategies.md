# 삽입 전략 — 무엇이 실제로 작동하는가

집계 대상: [User as Engram](https://arxiv.org/abs/2606.19172) 공개 저장소
([19PINE-AI/user-as-engram](https://github.com/19PINE-AI/user-as-engram))의 원본 결과 파일.
Mini-Engram 체크포인트 33개 × fact 16개 = 528건.
재현 `python scripts/analyze_strategies.py <repo>/results`

## 결과

| 전략 | Δlogit 평균 | Δlogit 중앙 | top-1 달성 | 순위 악화 |
|---|---|---|---|---|
| RANDOM | −2.326 | −1.004 | 5.5% | 63.6% |
| WTE | −0.028 | 0.000 | 6.8% | 52.3% |
| UNEMBED_P | +0.249 | +0.479 | 10.8% | 37.1% |
| DUAL | −4.950 | −4.865 | 0.4% | 89.4% |
| **OPT** | **+4.560** | **+4.564** | **84.7%** | **0.0%** |

- `Δlogit` = 삽입 후 gold 토큰 logit 변화
- `top-1 달성` = 삽입 후 gold 가 1순위가 된 비율
- `순위 악화` = 삽입 후 순위가 오히려 떨어진 비율

## 해석

**경사 기반 `OPT` 만 작동한다.**  84.7% 에서 gold 를 top-1 으로 만들고,
528건 중 순위를 악화시킨 경우가 **0건**이다.

**폐형식 `UNEMBED_P` 는 거의 무력하다.**  평균 +0.25 logit 으로 무작위보다
조금 나을 뿐이고, 37% 확률로 순위를 오히려 떨어뜨린다.  값 경로를 gold
unembedding 에 정확히 정렬시켜도(cos = 1.0000, 본 저장소에서 확인) 46개
레이어를 더 지나면서 씻겨 나간다.

**게이트를 함께 맞추려는 `DUAL` 은 역효과다** (−4.95, 89.4% 악화).
`[W_K; W_V] e = [h_t; u_gold]` 최소자승은 두 목표를 절충하다 값 경로를 망가뜨린다.
게이트는 값을 잘 쓰면 따라 열리므로 따로 강제할 대상이 아니다.

## 논문 본문과의 불일치

논문 Figure 8a 는 UNEMBED_P 로 게이트 α 가 0.02 → 0.59 로 오른다고 보고하고,
Figure 8b 는 UNEMBED_P 의 gold 방향 cosine 이 0.59 로 OPT(0.16)보다 높다고 한다.
본문만 읽으면 폐형식이 일의 상당 부분을 한다는 인상을 준다.

그러나 같은 저장소의 전략별 원본 결과는 위 표와 같다.  게이트가 열리는 것과
정답이 나오는 것은 별개다.  **본문 그림보다 부속 데이터를 따른다.**

## 구현 확인

`engram/edit.py` 의 `unembed_p()` 는 저장소 원본과 수학적으로 동일하다.

```python
# 원본: code/scripts/insertion_strategies_v2.py
def make_marker_UNEMBED_P(model, eng, layer_id, gold_id, ..., Wv_pinv=None):
    target   = model.lm_head.weight[gold_id].float()
    e_concat = (Wv_pinv @ target) * scale
    return e_concat.view(total_heads, embed_dim)
```

`W_V` 가 정방·가역(cond 43,479)이므로 `np.linalg.solve(W_V, u)` 와
`pinv(W_V) @ u` 는 같은 값이다.  본 저장소 구현은 검증 목적으로 유지하되,
**실사용 전략은 아니다.**

## 다음 작업

`OPT` 를 이 체크포인트에서 돌리려면 행에 대한 경사 또는 반복 forward 평가가
필요하고, 둘 다 현재 막혀 있다 ([docs/03-edit-protocol.md](../docs/03-edit-protocol.md)).
우선순위는 **실행 중인 vLLM 의 PLE 테이블을 재기동 없이 패치하는 훅**이다.

## OPT 은 최신 기법인가 — 아니다

2026-08-28 기준 문헌 조사 결과.

**OPT 은 기법이 아니라 베이스라인이다.** UNEMBED_P 로 초기화한 뒤 그 행에 대해
Adam 15스텝을 돌리는 것이 전부다.  같은 논문 안에서도 fact 가 30개를 넘으면
`Joint OPT`(유저의 모든 행을 함께 최적화)가 기본값이 된다.

**Engram 행 편집에 대해서는 이것이 유일한 검증 사례다.**  User as Engram
(2026-06) 이후 후속 연구가 없고, Qwen3.8-Flash-Next 의 테이블에 적용한 사례는 전무하다.
"최신"인 이유는 우수해서가 아니라 경쟁자가 없어서다.

**편집 문헌의 프론티어는 다른 곳에 있다.**  [AlphaEdit](https://arxiv.org/abs/2410.02355)
(null-space 투영) 이후 DeltaEdit, CrispEdit(AlphaEdit 를 특수 케이스로 일반화),
CAKE(ACL 2026), [O-Edit](https://arxiv.org/abs/2410.11469) 등이 있다.  모두
**MLP 가중치**를 대상으로 하며 해시 주소 메모리 테이블에는 그대로 적용되지 않는다.

### 이름 충돌 주의

[Forgetting Is Not a Fix: Path Dependence in Sequential Engram Editing](https://arxiv.org/abs/2607.24805)
(2026-07)은 더 최신이지만 **다른 Engram** 이다 — Kwon et al. 의 AI Engram
(가중치 공간 memory-trace localization, ICML 2026 oral).  DeepSeek 의 n-gram
룩업 모듈과 무관하다.

다만 교훈은 전이된다.  Qwen3-0.6B / TinyLlama-1.1B / Qwen2-0.5B 세 모델에서
**순차 편집은 경로 의존적이고 비가환**임을 보였다 — 편집 순서가 결과를 바꾸고,
오늘 확인한 삭제가 다음 편집 후에는 성립하지 않는다.

User as Engram 은 "주소가 겹치지 않으면 편집이 가산적·무손실"이라 주장하고,
그 근거는 슬롯 점유율 <1% 다.  이쪽 테이블은 포화 상태라 **겹침이 기본**이므로
저 경로 의존성 경고가 적용될 가능성이 높다.  여러 fact 를 쓸 계획이라면
순서 교환 실험을 프로토콜에 포함해야 한다.

### 미검증 개선 방향

AlphaEdit 의 null-space 투영을 이 문제로 옮기면: 편집 Δ 를 "같은 행을 공유하는
다른 n-gram 들의 출력에 영향을 주지 않는 부분공간"에 투영한다.  그 부분공간은
코퍼스 역인덱스(버킷 → 실제 등장 n-gram)로 구성할 수 있다.
**아무도 시도한 적 없는 조합이며, 따라서 검증된 바 없다.**

## 최신 기법을 고른다면 — 문제를 셋으로 쪼개야 한다

편집은 세 부분으로 나뉘고, "최신"의 답이 부분마다 다르다.

### (1) 어디를 고칠 것인가 — **해결됨, 기법 불필요**

ROME/MEMIT/AlphaEdit 이 causal tracing 과 key 계산에 쓰는 기계 대부분이
"어느 파라미터가 이 사실을 담고 있는가"를 찾는 데 쓰인다.  Engram 에서는
해시가 정확한 답을 공짜로 준다.  **이 절반을 통째로 건너뛴다.**

### (2) 여러 토큰짜리 답 — **[AnyEdit++](https://arxiv.org/abs/2606.01053) (2026-06)**

`005930` 은 7토큰이다.  단일 토큰 은닉상태만 고치는 기존 방식은 여기서 막히며
([AnyEdit](https://arxiv.org/abs/2502.05628), ICML 2025 은 이를 *efficacy barrier* 로 명명),
해법은 긴 지식을 청크로 분해해 청크마다 핵심 토큰을 순차 편집하는 것이다.
`engram/edit.py` 의 `plan()` 이 이미 만들어내는 단계 목록이 정확히 이 구조다.

AnyEdit++ 는 여기에 **Bayes-Chunk** — Bayesian Surprise 로 의미 경계를 잡아
청크를 적응적으로 나누는 장치 — 를 더한다.  우리 문제에 직결된다:
7단계를 다 편집할 필요가 없고 **어느 단계가 실제로 필요한지** 고를 수 있다.
[02-addressing.md](../docs/02-addressing.md) 의 열기 진단에서 step 3~6 이
위험 등급(z=+3~+6.5)이었으므로, 이 선택이 그대로 안전 문제로 연결된다.

### (3) 무엇을 쓸 것인가 — **어떤 방법도 경사를 피하지 못한다**

ROME·MEMIT·AlphaEdit 모두 목표 벡터 `v*` 를 **경사하강으로 최적화**한다.
폐형식인 것은 `v*` 를 얻은 뒤의 *가중치 갱신* 단계뿐인데, 우리에게는 공유
가중치 행렬이 없고 독립된 행만 있으므로 그 단계 자체가 없다.
따라서 OPT 류 최적화는 선택이 아니라 필수다.

### (4) 부수 피해 억제 — AlphaEdit 아이디어는 그대로 오지 않는다

AlphaEdit 의 null-space 투영은 *공유 가중치 W* 를 통한 간섭을 막는다.
우리 간섭은 **해시 충돌** — 같은 셀을 여러 n-gram 이 공유하는 것 — 이라
투영으로 비켜갈 자유 부분공간이 없다.  셀 자체가 공유 자원이기 때문이다.
전이 가능한 형태는 "같은 행을 읽는 다른 문맥들에서 게이트가 닫히도록" 만드는 것이고,
그러려면 역인덱스와 그 문맥들의 `h_t` 가 필요하다.

### 권고 조합

**AnyEdit++ 식 청크 분해(외부 루프) + OPT 식 forward-only 최적화(내부 해법)
+ 역인덱스 기반 충돌 페널티.**

### 반드시 넣어야 할 대조군: SCR

[Benchmarking and Rethinking Knowledge Editing](https://arxiv.org/abs/2505.18690) (2026-05)
은 **모든 파라미터 편집 기법이 SCR — 파라미터를 건드리지 않는 확장형 외부
텍스트 메모리 — 에 밀린다**고 보고한다.  종목코드 조회라는 과제만 놓고 보면
RAG 가 정답이다.  이 프로젝트의 가치는 "종목코드를 잘 답하기"가 아니라
**프로덕션 규모 Engram 모델에서 행 편집이 실제로 되는가**라는 미검증 질문에 있다.
그 구분을 흐리지 않기 위해 RAG 기준선을 함께 측정한다.
