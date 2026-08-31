# 라이브 패치 훅 — 재기동 없이 테이블 행 고치기

`OPT` 는 반복적인 forward 평가를 요구한다.  PLE 테이블은 기동 시 호스트 RAM 으로
offload 되므로 디스크 수정은 재기동해야 반영되고, 재기동은 수 분이라 반복
최적화에 쓸 수 없다.  이 훅이 그 병목을 없앤다.

## 구조

vLLM 은 PLE 테이블을 **전용 오프로드 프로세스 하나**에 통째로 올린다.

```
$ docker exec flashnext ps -eo pid,rss,args --sort=-rss
    PID    RSS  COMMAND
    489  101838408  ...spawn_main...      <- 오프로드 프로세스, 101.8 GB
    212    4369564  VLLM::Worker_TP3_EP3
    ...
```

101.8 GB = 320,001,536 × 160 × 2 바이트.  **TP 로 쪼개져 있지 않다** — 전역 행
주소가 곧 로컬 인덱스이므로 인덱스 변환이 필요 없다.

테이블 실체는 `runner._layers[...].ngram_embedding.weight` 이며,
조회는 `torch.index_select(weight, 0, ngram_ids)` 다.

## 설치

컨테이너 이미지를 다시 만들지 않는다.  파일 두 개를 bind-mount 로 덮어쓴다.

| 파일 | 내용 |
|---|---|
| `vllm_patch/patch_hook.py` | 요청 디렉터리를 감시하는 데몬 스레드 (신규) |
| `vllm_patch/worker.py` | 원본에 훅 시작 7줄만 삽입 |

```bash
ENGRAM_HOOK=1 ./serve.sh
```

되돌리려면 `ENGRAM_HOOK=0` 으로 다시 띄우면 된다.  이미지는 건드리지 않았다.

## 프로토콜

파일 기반이다.  `ple_patch/` 에 요청을 놓으면 훅이 처리하고 응답을 놓는다.

```
<name>.req.npz   mode('read'|'write'), addrs int64[N], rows float32[N,160]
<name>.ack.npz   ok, addrs, prev float32[N,160]      prev = 쓰기 전 값
```

`prev` 를 항상 돌려주므로 **모든 쓰기가 자동으로 되돌릴 수 있다.**

```python
from engram import live
prev = live.read(addrs)            # 현재 값
prev = live.write(addrs, rows)     # 덮어쓰고 이전 값 회수
live.write(addrs, prev)            # 되돌리기
live.restore_from_disk(addrs)      # 체크포인트 기준으로 강제 복원
live.diff_from_disk(addrs)         # 편집이 남아있는 행 수
```

## 검증

| 항목 | 결과 |
|---|---|
| 훅 왕복 지연 | **47 ms** |
| 디스크 읽기 vs 라이브 읽기 | **비트 단위 일치** |
| 쓰기 반영 | 즉시 (원본 −16.33 → 노이즈 −18.95) |
| 복원 정확도 | **정확히 원값** (−16.33, 3회 반복 동일) |
| 평가 1회 총 소요 | 272 ms (API 223 + 훅 49) |

디스크와 라이브가 비트 단위로 같다는 것은 훅이 동작한다는 것과
**주소 계산이 실제 모델과 일치한다**는 것을 동시에 증명한다.

## 알아둘 것

**측정은 반드시 생성 시점과 같은 전체 프롬프트로 해야 한다.**
이 모델은 짧은 프롬프트와 긴 프롬프트가 같은 토큰에 다른 logprob 을 준다
(최대 1.7 nats).  GDN 순환 상태 + prefix caching + chunked prefill 조합 탓이다
(기동 로그: *"Mamba cache mode is set to 'align' ... when prefix caching is enabled"*).
`engram.measure.AnswerScorer` 는 전체 프롬프트 한 번으로 토큰별 logprob 을
분해해 이 함정을 피한다.

**락을 쓰지 않는다.** 한 행 쓰기는 320 바이트라 마이크로초 단위로 끝난다.
패치와 추론을 동시에 보내지 않는 것이 호출자 책임이다.

**콜드 요청은 이상치를 낸다.** 유휴 후 첫 요청이 2 나트까지 벗어나는 경우가 있다.
`measure.warmup()` 으로 버리는 호출을 한 번 넣는다.  연속 측정은 완전히 결정론적이다
(15회 연속 정확히 동일).
