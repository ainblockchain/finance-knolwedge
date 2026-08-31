# Qwen3.8-Flash-Next — Engram 51.2B 룩업 테이블 Pin-point 편집

## 목표

**Qwen3.8-Flash-Next 가 모르는 사실 하나를 51.2B Engram 테이블의 행 16개만
고쳐서 top-1 으로 만들되 나머지 입력의 출력은 비트 단위로 보존하는 것 —
선행연구가 정반대 조건에서만 검증한 기법이 프로덕션 규모에서도 성립하는지에 대한 첫 검증.**

"정반대 조건"이 이 프로젝트의 전부다.  행 편집을 검증한 유일한 선행연구
([User as Engram](https://arxiv.org/abs/2606.19172))는 아래 세 조건에서만 측정했고,
이 체크포인트는 셋 다 반대다.

| | 선행연구 | 이 체크포인트 |
|---|---|---|
| 모델 규모 | 178M–1.22B (자체 학습) | **176B** |
| 테이블 | 51.2M, 슬롯 점유율 <1% | **51.2B, 히트율 1.0 포화** |
| Engram 위치 | 두 곳, **늦은** 쪽에 씀 | **레이어 1 하나뿐** (48개 중) |

특히 세 번째가 치명적이다 — 같은 논문이 이른 레이어에 쓰면 recall 이 완벽에서
약 1/4 로 떨어진다고 보고한다.

성공 기준 (반증 가능하게):
1. 모델이 모르는 사실(2022년 이후 상장 코스닥, 기준선 정확도 **0.2%**)을 고른다.
2. 편집 후 그 문구에서 목표 코드가 top-1 이 된다.
3. 건드리지 않은 입력의 출력이 비트 단위로 동일하다.
4. 같은 행을 공유하는 다른 n-gram 의 피해를 역인덱스로 **측정해서 보고**한다.
5. RAG 기준선과 나란히 비교한다 — 과제 자체는 RAG 가 정답일 수 있다.

## 왜 가능한가

Engram(체크포인트에서는 `ple`)은 입력 토큰 ID 만으로 주소가 결정되는
content-addressable 메모리다.  한 토큰이 320,001,536 행 중 **16 행**만 읽으므로,
그 16 행을 고치면 그 n-gram 이 등장하는 위치에서만 모델이 달라진다.
자세한 동작 원리는 [docs/01-architecture.md](docs/01-architecture.md).
공식 학습 방식(공동 학습, Adam 5× lr, TriviaQA 29% ablation)은 [docs/05-official-training.md](docs/05-official-training.md).
51B 가 지식을 저장하는 원리 전체는 [docs/06-how-51b-stores-knowledge.md](docs/06-how-51b-stores-knowledge.md).
저자의 설계 의도와 향후 변경 방향은 [docs/07-design-intent.md](docs/07-design-intent.md).
조회 원시연산의 구성요소별 해부(해시·게이트·conv·Qwen 구현 차이)는 [docs/08-lookup-primitive.md](docs/08-lookup-primitive.md).
공식 방법(행에 Adam, decay 0, LM 손실)으로 지식 1건 학습 성공 기록은 [results/13-official-training.md](results/13-official-training.md), 코드는 `train/`.

## 현재 상태

| 단계 | 상태 |
|---|---|
| 1. 주소 체계 역공학 + 검증 | **완료** — [docs/02-addressing.md](docs/02-addressing.md) |
| 2. 행 읽기/쓰기/백업 | **완료** — `engram/core.py`, `engram/edit.py` |
| 3. 기존 지식 기준선 측정 | **완료** — [results/01-knowledge-baseline.md](results/01-knowledge-baseline.md) |
| 4. 삽입 전략 조사 | **완료** — [results/02-insertion-strategies.md](results/02-insertion-strategies.md) |
| 5. 라이브 패치 훅 | **완료** — `vllm_patch/`, `engram/live.py` (왕복 47 ms) |
| 6. 주소 가능성 한계 측정 | **완료** — [results/03-addressability.md](results/03-addressability.md) |
| 7. 단일 사실 편집 + 검증 | **성공** — [results/04-single-fact-edit.md](results/04-single-fact-edit.md) |
| 8. 부수 피해 최소화 (3-gram 헤드 전용) | **완료** — 285개 → 1개 |
| 9. 채널 용량 측정 | **완료** — [results/05-channel-capacity.md](results/05-channel-capacity.md) |
| 10. 순차 편집 (20회) | **완료** — 손실 없이 누적 |
| 11. 트리거 템플릿 실험 | **완료** — [results/07-trigger-template.md](results/07-trigger-template.md) |
| 12. 오염 정정 + 재검증 | **완료** — [results/08-corrections.md](results/08-corrections.md) |
| 13. 목적함수 수정 + 달성률 실측 | **완료** — [results/09-objective-breakthrough.md](results/09-objective-breakthrough.md) |

### 7단계 결과

모델이 모르는 사실 하나(`종목코드 필에너지` → ` 378340`)를 **행 80개 /
파라미터 12,800개**만 고쳐 학습시켰다.  백본은 건드리지 않았다.

```
편집 전   logprob -16.33   생성 ', 2차전지 장비 전문기업…상장'
편집 후   logprob  -3.07   생성 ' 378340'          <- 정답
```

n-gram 을 공유하지 않는 프롬프트(`대한민국의 수도는`, `삼성전자 종목코드는`,
`The capital of France is`, `2 + 2 =`)는 **비트 단위로 동일**했다.

### 검증 결과 — [results/10-what-we-actually-know.md](results/10-what-we-actually-know.md)

**"모두 틀리지 않고" 는 달성되지 않았다.**  그러나 무엇이 그것을 막는지에 대해
나는 신뢰할 만한 진술을 할 수 없다 — 이 프로젝트에서 한계를 **일곱 번** 선언했고
**일곱 번 다 틀렸다** (테이블 오염, 선택 편향, 채널 임계, 템플릿, 자리당 채널,
자연 질문 형식 0%, 후성의 2/7 벽).

**확실한 것**
- 기법은 작동한다 — 생성으로 검증된 성공 **7건**, 그중 하나는 **자연 질문 형식**
  (`{name}의 종목코드는` → 정답).  행 40~80개(파라미터 ~10⁻⁷), 백본 무변경,
  무관한 프롬프트는 비트 단위 동일.
- 순차 편집은 손실 없이 누적된다 (20회 중 유지 14/19).
- 부수 피해 431개 → 1개.
- 편집은 **표면형에 특화**된다 — 서빙 계층이 프롬프트를 고정해야 한다.

**확실하지 않은 것** — 모집단 달성률.  측정마다 0/6 ~ 3/8 로 흔들리며
예산에 강하게 의존한다 (성공 사례는 50~150회를 썼고, 0/6 은 30회 측정이다).
충분한 표본을 모으지 못했다.

**남은 방향**은 옵티마이저다 — 예산 증대, 전략 자동 선택, 하이브리드.
아키텍처에서 막힌다는 증거는 현재 없다.

### 채택한 기법

세 층위를 조합한다 ([results/02-insertion-strategies.md](results/02-insertion-strategies.md)):

- **어디** — 해시가 정확히 알려준다 (locate 단계 불필요)
- **분해** — AnyEdit++ 식 청크 분해.  6자리 코드를 자릿수별 편집 단계로 나눈다
- **값** — OPT.  역전파가 막혀 있으므로 **SPSA** (forward-only, 반복당 평가 2~3회,
  평가당 272 ms) 로 대체.  저차원 랜덤 부분공간에서 탐색한다

## 구조

```
engram/          라이브러리
  core.py        해시 상수 · 주소 계산 · 행 I/O · 토크나이저
  edit.py        편집 계획 · UNEMBED_P · 백업/복원
  live.py        실행 중 테이블 읽기/쓰기 (패치 훅 클라이언트)
  measure.py     답 토큰별 logprob · margin 측정
  opt.py         SPSA forward-only 행 최적화
  backups/       덮어쓰기 전 원본 행 (.npz)
vllm_patch/      컨테이너에 bind-mount 하는 패치
  patch_hook.py  오프로드 프로세스의 테이블 감시 데몬 (신규)
  worker.py      원본 + 훅 시작 7줄
ple_patch/       훅 요청/응답 디렉터리
scripts/
  fetch_krx.py            KRX 상장사 목록 수집
  eval_krx.py             종목코드 지식 평가 실행
  analyze_krx.py          평가 결과 분석
  analyze_strategies.py   선행연구 삽입 전략 집계
  validate_addressing.py  주소 체계 검증
  trigger_uniqueness.py   형식별 주소 가능성 상한 측정
  reachability.py         자리별 margin 으로 도달 가능성 전수 측정
  collateral.py           역인덱스로 부수 피해 목록화
  edit_one_fact.py        단일 사실 편집 + 검증
  edit_many.py            단일 편집 N회 반복 + 순차 간섭 검사
  achieved_rate.py        실제로 틀리는 종목에서의 달성률 실측
  restore_all.py          라이브 테이블을 체크포인트 상태로 복원
data/            krx.html, krx.json, krx_eval.json
docs/            01 아키텍처 · 02 주소 체계 · 03 편집 프로토콜 · 04 라이브 훅
engram/template.py  트리거 템플릿 (plain / head1 / hash2)
scripts/template_compare.py  템플릿별 실측 비교
data/            krx.html, krx.json, krx_eval.json
results/         01 지식 기준선 · 02 삽입 전략 · 03 주소 가능성 · 04 단일 편집
                 05 채널 용량 · 06 최종 검증 · 07 트리거 템플릿 · 08 정정 기록
                 09 목적함수 돌파 · achieved-rate.json
serve.sh         vLLM 기동 (ENGRAM_HOOK=1 이면 패치 훅 포함)
```

## 빠른 시작

```bash
cd /mnt/newdata/qwen3.8
pip install numpy safetensors tokenizers

# 문구가 어떤 행을 읽는지
python -c "
from engram.edit import plan
for s in plan('삼성전자 종목코드는', ' 005930')['steps']:
    print(s['step'], s['ctx'], s['addrs'][:3])"

# 주소 체계가 맞는지 재검증
python scripts/validate_addressing.py

# 지식 기준선 재측정 (vLLM 기동 필요, ~3분)
python scripts/fetch_krx.py && python scripts/eval_krx.py && python scripts/analyze_krx.py
```

## 안전

`engram.edit.write_rows` 는 기본이 `dry_run=True` 다.  실제 쓰기 전에는 반드시
`engram.edit.backup()` 으로 원본 행을 `.npz` 에 남기고, `restore()` 로 되돌린다.
PLE 테이블은 기동 시 호스트 RAM 으로 offload 되므로 **디스크 수정은 재기동해야 반영된다.**

## 공식 레시피 학습과 데모 (08-29 ~ 08-30)

- `train/` — HF(autograd) 자체 로더(`hf_model.py`: int4 패킹 백본을 forward 에서 역양자화, PLE 는 CPU 행 테이블),
  단일 사실 학습(`train_fact.py`), 전 종목 학습(`train_all.py`), 오답 집중(`train_hard.py`), 컨테이너(`container.sh`).
- `results/train-all/rows-epN.npz` — 에포크별 패치(주소·원본·학습값). `scripts/patch.py apply|remove|status` 로 서빙 중 끼우고 뺀다.
- `scripts/verify_all_rows.py` — 전 종목 vLLM 자유 생성 검증(재기동 감지·재적용). `scripts/patch_watchdog.py` — 재기동 시 자동 재적용. `scripts/hang_watchdog.sh` — vLLM 멈춤 감지 즉시 재기동 (두 서버).
- `web/` — 데모: 채팅 전/후 비교(`index.html`), 벤치마크(`bench.html`), 추론 추적(`trace.html`), 기법 교육 자료(`method.html`).
  `python3 web/server.py --port 8080`; 학습 후 서버는 `web/launch_after.sh <npz>` (GPU 4,5 TP=2 eager, port 8001).
- 기록: `results/13-official-training.md`(단일 사실), `results/14-full-corpus.md`(**전 종목 100%**), `results/12-parallel-crash.md`(vLLM PLE 오프로드 크래시 3차 조사).

## 저장소 범위
- 코드(train/, scripts/, engram/, vllm_patch/, web/), 문서(docs/, results/*.md, *.json), 데이터(data/krx.json) 포함.
- **제외**(용량/보안): 패치 바이너리 `results/train-all/*.npz`(개당 ~300MB, 총 4.8GB), 모델 논문 PDF, 로그, 백업본. 패치는 별도 스토리지/LFS 로 관리.
- 실행 스크립트의 sudo 비밀번호는 환경변수 `SUDO_PW` 로 분리됨(`export SUDO_PW=... ` 후 실행).
