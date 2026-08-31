# 병렬 편집 크래시 — 진짜 원인은 오프로드 큐

## 이력: 잘못 짚은 원인

처음에는 MTP 추측 디코딩을 원인으로 지목했다.  스케줄러 덤프의
추측 토큰이 `[-1,-1,-1,-1]` 이었고, 드래프트/검증 두 통과 사이에
행이 바뀌면 불일치가 난다는 설명이 그럴듯했다.  MTP 를 끄고
재기동했다.

**틀렸다.**  MTP 를 끈 뒤에도 같은 크래시가 5회 반복됐다.
`_post` 의 500 재시도가 이를 가려서, 실행이 그냥 느려 보였을 뿐이다.
`[-1,-1,-1,-1]` 은 원인이 아니라 워커가 죽은 뒤의 잔해였다.

## 실제 트레이스백

```
vllm/v1/ple_offload/connector.py, line 389, in _launch
    self._request_queue.put_nowait(request)
queue.Full
```

5회 전부 `Worker_TP0_EP0`, 전부 같은 지점.

## 메커니즘 (2차 조사에서 확정)

`queue.Full` 은 **증상**이었다.  연결자를 "기다리게" 고친 뒤에는 단일
작업자에서도 다른 형태로 죽었다:

```
10:20:59  PLE patch ...e28a835a: mode=write rows=16     ← 마지막 행 쓰기
          (그 뒤 완료 응답 없음)
10:22:00  No available shared memory broadcast block found in 60 seconds
   ...
10:25:59  TimeoutError: RPC call to sample_tokens timed out.  → EngineDeadError
```

행 쓰기 직후 forward 가 응답하지 않았다.  GPU 워커는 PLE 출력 세마포어를
기다리고, 그 출력을 만들어야 할 **오프로드 프로세스가 멈춰 있었다**.

오프로드 프로세스 안에는 두 스레드가 테이블을 만진다:
- 러너의 `_handle_requests` → `layer.forward_impl` (51B 테이블 gather)
- 편집 훅 `_handle` → `weight[idx] = ...` (torch index_put)

둘 다 torch CPU 연산이고 같은 intra-op 스레드 풀을 쓴다.  잠금 없이
겹치면 프로세스가 굳는다.  훅 문서에 "쓰기는 마이크로초라 잠금이 필요
없다"고 적어 놓았었는데, 문제는 쓰기 시간이 아니라 **동시 진입**이었다.

이전 `queue.Full` 도 같은 경로다: 오프로드가 멈추면 요청 스레드가
응답을 못 받고, 큐의 한 칸이 안 비워지고, 다음 forward 의 `put_nowait`
이 던진다.  병렬도가 높을수록 쓰기와 gather 가 겹칠 확률이 올라가
3·4병렬에서 먼저 터졌고, 단일 작업자도 확률이 0 이 아니었다.

## 수정

1. **상호 배제** — `PleOffloadRunner.table_lock` 을 두고, 러너의
   `forward_impl` 호출과 훅의 행 읽기/쓰기가 같은 잠금을 잡는다.
2. **훅 쓰기를 torch 밖으로** — bf16 테이블을 `uint16` numpy 뷰로 잡고
   행을 numpy 로 읽고 쓴다.  torch 스레드 풀에 아예 들어가지 않는다.
   bf16 반올림은 torch 캐스트와 비트 단위로 일치함을 검증했다
   (round-to-nearest-even).
3. 연결자의 `put(timeout=120)` 은 유지 — 무해하고, 남은 어떤 지연도
   즉사 대신 대기로 바꾼다.

파일: `vllm_patch/worker.py`, `vllm_patch/patch_hook.py`,
`vllm_patch/connector.py` (모두 bind-mount, `serve.sh`).

## MTP 에 대해

MTP 를 끈 것은 무해하지만 근거가 없었다.  `serve.sh` 의 `MTP` 토글은
남겨두되, 편집 실험에 반드시 꺼야 한다는 이전 문서의 주장은 철회한다.

## 픽셀플러스 반전에 대해

픽셀플러스 087600 은 1차에서 6통과 소진 실패, 2차에서 편집 3회 성공.
두 실행 사이에 바뀐 것: (1) MTP 끔, (2) 앞 종목(씨싸이트)의 미복원
편집 행 위에서 측정된 오염 제거, (3) 통과 예산.  MTP 가 크래시
원인이 아니었으므로 (1) 이 차이 요인일 근거는 없다.  (2) 가 유력하나
표본 1건이라 단정하지 않는다.  실패 시 미복원 버그는 고쳤다.

## 3차 조사 (08-30) — 훅과 무관하게 지속 부하에서 죽는다

잠금 훅 이후에도 `RPC call to sample_tokens timed out` → `EngineDeadError` 가 두 번
더 났다 (03:21, 05:27).  두 번 모두 **훅 쓰기가 전혀 없는 구간**이었다 — 05:18 에
패치를 쓴 뒤 05:32 까지 쓰기가 없었고, 05:27 에 죽었다.  공통점은 16 스레드로
2,761 종목을 연속 생성하던 **지속 부하**뿐이다.

따라서 크래시의 원인은 편집이 아니라 vLLM 의 PLE CPU 오프로드 경로 자체이며
(GPU 워커가 오프로드 출력을 기다리다 RPC 타임아웃), 이전에 편집 실험 중에만
보였던 것은 그때만 연속 부하가 있었기 때문이다.  잠금·큐 수정은 무해하지만
원인 제거가 아니었다.

영향: 컨테이너가 자동 재기동되면 테이블이 **디스크 원본으로 돌아간다**.  이를
모르고 잰 첫 전 종목 검증은 2,100 번째 종목부터 원본 모델을 재고 있었다
(84.2% 로 보였으나 앞 2,000 종목은 99.9%).

대응 (원인 제거 불가, 완화):
- `scripts/patch_watchdog.py`: 표본 64 행을 15 초마다 읽어 패치가 사라지면 재적용
- `scripts/verify_all_rows.py`: 100 종목마다 패치 상태 확인, 되돌아갔으면 재적용 후
  그 구간 재측정.  동시성 16 → 8
- 데모 "학습 후" 서버는 감시자를 함께 띄운다 (`web/launch_after.sh`)

## 4차 (08-30 10:17) — 유휴 상태에서도 멈춘다

부하도 편집도 없는 상태(10분간 훅 쓰기 0, 요청 몇 건)에서 학습 전 서버가 멈췄다.
`shm_broadcast.py:801 No available shared memory broadcast block found in 60 seconds` 가
매분 반복되고 GPU 사용률 0%, `/v1/models` 는 200 을 주지만 생성 요청은 무응답.
5 분 뒤 RPC 타임아웃으로 재기동된다.

추가 완화: `scripts/hang_watchdog.sh` — 두 컨테이너 로그에서 위 경고가 2 회 연속
나오면 즉시 `docker restart` (5 분 대기 대신 ~2 분 안에 복구).  학습 후 서버는
`patch_watchdog.py` 가 재기동 후 패치를 다시 끼운다.

근본 원인은 vLLM 의 PLE CPU 오프로드 경로에 있으며 이 프로젝트 범위 밖이다.
GPU 상주 PLE(`VLLM_PLE_CPU_OFFLOAD=0`) 는 80GB 급 GPU 가 필요하다.
