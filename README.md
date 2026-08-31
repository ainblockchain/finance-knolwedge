# Qwen3.8-Flash-Next 조건부 기억(Engram/PLE) 지식 편집

Qwen3.8-Flash-Next의 51.2B n-gram 임베딩 테이블(체크포인트에서는 `ple`)의 행만을
학습시켜, 모델이 모르던 사실을 백본 수정 없이 주입하고 검증하는 프로젝트다.
한국 상장사 종목코드를 대상으로 하며, 특허 자료의 근거 구현이기도 하다.

## 결과 요약

| 항목 | 학습 전 | 학습 후 |
|---|---|---|
| 전 종목 종목코드 자유 생성 (2,761종목, 독립 엔진 vLLM 검증) | 19.0% | **100.00%** |
| 채팅 형식 질의 ("{회사명} 종목코드 알려줘", 무작위 60종목) | 13% | 97% |
| 일반 프롬프트 (Paris, 서울 등) | 정답 | 정답 유지 |
| 수정된 파라미터 | — | 테이블 270,053행 (전체의 0.084%, 86 MB), 백본 불변 |

무관 텍스트에 대한 부수 영향은 토큰당 0.08 nat 이하였다. 상세 기록은
[results/14-full-corpus.md](results/14-full-corpus.md)에 있다.

## 원리

Engram(PLE) 테이블은 입력 토큰 ID만으로 주소가 결정되는 조건부 기억이다. 한 토큰이
320,001,536개 행 중 16개만 조회하므로, 그 16개 행을 학습시키면 해당 n-gram이
등장하는 위치에서만 모델의 출력이 달라진다. 주소가 순전파 이전에 확정되고 테이블이
가중치 감쇠 없이 학습된다는 두 성질이 행 단위 편집을 가능하게 한다.

- [docs/06-how-51b-stores-knowledge.md](docs/06-how-51b-stores-knowledge.md) — 지식 저장 원리
- [docs/08-lookup-primitive.md](docs/08-lookup-primitive.md) — 조회 기본 연산(lookup primitive)의 구성요소별 세부 분석
- [docs/05-official-training.md](docs/05-official-training.md) — 공식 학습 방식(공동 학습, Adam, weight decay 0)
- [docs/07-design-intent.md](docs/07-design-intent.md) — 저자의 설계 의도

## 학습 방법

공식 레시피(백본 동결, 참조 행만 언어 모델링 손실로 학습, 가중치 감쇠 없는 Adam)를
그대로 적용한다. 서빙 엔진은 역전파를 지원하지 않으므로, PyTorch로 모델을 다시 올리되
4비트로 양자화된 백본 가중치를 압축된 상태로 두고 순전파 시에만 역양자화하여 입력 방향 기울기만
전파한다. 테이블은 호스트 메모리에 두고 배치가 참조하는 행만 가속기로 올린다.

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 자체 로더 | `train/hf_model.py` | int4 패킹 백본 역양자화, PLE는 CPU 행 테이블, 3-GPU 분할 |
| 단일 사실 | `train/train_fact.py` | 한 종목을 10스텝 내로 학습 |
| 전 종목 | `train/train_all.py` | 표현 3종 + 채팅 형식, 오답 오버샘플링 |
| 오답 집중 | `train/pinpoint.py` | 다른 종목이 참조하지 않는 고유 행만 학습 |
| 역방향 | `train/train_rev.py` | 코드→회사명 양방향 학습 |

## 검증과 운영

- `scripts/patch.py apply|remove|status` — 서빙 중 패치를 무중단으로 적용·회수한다.
- `scripts/verify_all_rows.py` — 전 종목을 vLLM 자유 생성으로 검증한다. 서빙 재시작에
  의한 테이블 환원을 표본 행으로 감지하여 재적용·재측정한다.
- `scripts/patch_watchdog.py` — 재시작 시 패치를 자동 재적용한다.
- `scripts/hang_watchdog.sh` — vLLM 멈춤을 감지하여 즉시 재기동한다.

## 데모

`web/`에 학습 전후 비교 데모가 있다.

```bash
python3 web/server.py --port 8080
```

- `index.html` — 같은 질문을 학습 전(8000)·학습 후(8001) 서버에 동시 전송하여 비교
- `bench.html` — 전 종목 벤치마크, 채팅 형식 점수, 학습 곡선, 종목별 사례
- `trace.html` — 이름이 유사한 종목의 위치별 행 공유와 절제 실험
- `method.html` — 편집 기법 교육 자료

학습 후 서버는 `web/launch_after.sh <패치.npz>`로 기동한다(GPU 4·5, TP=2).

## 구조

```
engram/          라이브러리 (주소 계산, 행 I/O, 라이브 훅, 측정, SPSA 최적화)
train/           공식 레시피 학습 (자체 로더, 정방향·전종목·핀포인트·역방향)
vllm_patch/      PLE 오프로드 훅과 커넥터 (컨테이너에 bind-mount)
scripts/         패치 적용·검증·감시, KRX 수집·평가
web/             학습 전후 비교 데모
docs/            아키텍처·주소 체계·저장 원리·설계 의도
results/         기준선·검증·채점 기록 (*.md, *.json)
data/            krx.json (KRX 상장사 목록)
serve.sh         vLLM 기동 (ENGRAM_HOOK=1 이면 패치 훅 포함)
```

## 빠른 시작

```bash
cp .env.example .env      # SUDO_PW 값을 채운다
pip install numpy safetensors tokenizers

python scripts/validate_addressing.py                    # 주소 체계 재검증
python scripts/patch.py status results/train-all/rows-pin.npz   # 최종 패치 상태 확인
```

## 안전

`engram.edit.write_rows`는 기본이 `dry_run=True`다. 실제 쓰기 전에 반드시
`engram.edit.backup()`으로 원본 행을 `.npz`에 백업하고 `restore()`로 되돌린다.
PLE 테이블은 기동 시 호스트 RAM으로 오프로드되므로, 디스크 수정은 재기동해야 반영된다.

## 저장소 범위와 보안

- 코드·문서·결과(md, json)·데이터(`data/krx.json`)를 포함한다.
- 패치 바이너리(`results/train-all/*.npz`, 개당 약 300 MB, 총 4.8 GB), 모델 논문 PDF,
  로그, 백업본은 용량과 보안 때문에 제외한다. 패치는 aindrive로 공유한다:
  **https://aindrive.ainetwork.ai/d/q09HFflTJjej** (받는 방법: [docs/09-패치-사용법.md](docs/09-패치-사용법.md)).
- 실행 스크립트의 sudo 비밀번호는 `.env`(git 제외)로 분리한다. `.env.example`을
  복사하여 값을 채운다.
