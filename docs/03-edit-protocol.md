# 편집 프로토콜

## 사실을 어디에 쓰는가

위치 p 에서 읽은 Engram 행은 **토큰 p+1** 의 예측에 기여한다.  따라서
`(trigger, answer)` 를 쓴다는 것은 답의 각 토큰마다 그 직전 위치의 16행을
고치는 것이다.

```python
from engram.edit import plan
pl = plan("삼성전자 종목코드는", " 005930")
# steps[i].pos      = 편집할 위치
# steps[i].ctx_ids  = 그 위치의 (현재, -1, -2) 토큰
# steps[i].addrs    = 16개 전역 행 주소
```

문맥이 슬라이딩하므로 답의 후반부일수록 문맥이 답 자신의 숫자로 채워진다.
`['5','0','0']` 같은 문맥의 행은 코퍼스의 모든 숫자가 공유한다.

## 삽입 전략 — 무엇이 실제로 작동하는가

선행연구([User as Engram](https://arxiv.org/abs/2606.19172)) 저장소의 원본 결과
33개 체크포인트 × 16개 fact 를 집계했다 (`scripts/analyze_strategies.py`).

| 전략 | 방식 | Δlogit 평균 | top-1 달성 | 순위 악화 |
|---|---|---|---|---|
| RANDOM | 무작위 벡터 | −2.33 | 5.5% | 63.6% |
| WTE | 토큰 임베딩 복사 | −0.03 | 6.8% | 52.3% |
| UNEMBED_P | `e = W_V⁺ u_gold` 폐형식 | +0.25 | 10.8% | 37.1% |
| DUAL | `[W_K;W_V] e = [h_t;u]` 최소자승 | −4.95 | 0.4% | 89.4% |
| **OPT** | UNEMBED_P 초기화 후 **Adam 15스텝** | **+4.56** | **84.7%** | **0.0%** |

**결론: 경사 기반 `OPT` 만 실질적으로 작동한다.**  폐형식은 무작위보다 약간 나은
수준이고 37% 확률로 오히려 순위를 악화시킨다.  게이트까지 함께 맞추려는 `DUAL` 은
값 경로를 망가뜨려 역효과다.

논문 본문의 그림(게이트 α 가 UNEMBED_P 로 0.02→0.59)만 보면 폐형식이 일을 하는
것처럼 보이지만, 실제 recall 수치는 위와 같다.  **본문 그림과 부속 데이터가
주는 인상이 다르므로 데이터 쪽을 따른다.**

## 이 체크포인트에 특유한 위험

**Engram 이 레이어 1 하나뿐이다.**  선행연구의 세 번째 발견은 "네트워크 끝
근처에서만 작동한다" 이고, 같은 fact 를 이른 레이어에 쓰면 recall 이 완벽에서
약 1/4 로 떨어진다.  DeepSeek Engram-27B 는 [2, 15] 두 곳, 논문의 Mini-Engram 도
두 곳을 두고 늦은 쪽에 쓴다.  Qwen3.8-Flash-Next 에는 늦은 쪽이 없다.

**테이블이 포화 상태다.**  선행연구는 충돌이 병목이 아니라고 하지만 그 근거는
"슬롯 점유율 <1%" 이다.  그쪽 테이블은 50K×256 = 51.2M 파라미터고, 이쪽은
320M 행이 조 단위 토큰으로 학습돼 히트율이 1.0 에 수렴한 상태다.  점유율이 정반대다.

## 구현 상태와 막힌 곳

`OPT` 는 gold 토큰 logprob 에 대한 **행의 경사**를 요구한다.  두 경로 모두 막혀 있다.

- **역전파**: W4A16 Marlin 커널에 backward 가 없다.  BF16 원본 335 GB 는
  7×A100-40GB(280 GB)에 올라가지 않는다.
- **forward-only 반복 최적화**(SPSA 등): 평가 1회마다 PLE 테이블을 바꿔야 하는데,
  실행 중인 vLLM 의 호스트 RAM 상주 테이블을 재기동 없이 덮어쓸 방법이 없다.
  재기동은 수 분이라 반복 최적화에 못 쓴다.

따라서 **다음 작업은 vLLM 프로세스의 PLE 테이블을 in-place 로 패치하는 훅**이다.
그게 되면 각 평가가 API 호출 1회(약 60 ms)가 되어, 2,560차원 SPSA 를
수천 스텝 돌려도 몇 분이면 끝난다.

## 안전 절차

```python
from engram.edit import plan, unembed_p, backup, apply_step, restore

pl   = plan(trigger, answer)
step = pl["steps"][1]

path = backup(step["addrs"], tag="samsung-step1")   # 원본 .npz 로 보존 (필수)
apply_step(step, unembed_p(step["target_id"]), dry_run=True)   # 먼저 확인
apply_step(step, unembed_p(step["target_id"]), dry_run=False)  # 실제 쓰기
# ... 검증 ...
restore(path)                                        # 되돌리기
```

- `write_rows` 는 기본이 `dry_run=True`.
- PLE 는 기동 시 호스트 RAM 으로 offload 되므로 **디스크 수정은 재기동해야 반영된다.**
- 편집 대상은 모델이 **모르는** 사실이어야 한다.  아는 사실로는 편집 효과와
  기존 지식을 구분할 수 없다 ([results/01-knowledge-baseline.md](../results/01-knowledge-baseline.md)).
