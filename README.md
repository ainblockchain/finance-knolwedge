# Qwen3.8-Flash-Next 조건부 기억(Engram/PLE) 지식 편집

Qwen3.8-Flash-Next의 51.2B n-gram 임베딩 테이블(체크포인트에서는 `ple`)의 행만을
학습시켜, 모델이 모르던 사실을 백본 수정 없이 주입하고 검증하는 프로젝트다.
한국 상장사 종목코드를 대상으로 하며, 특허 자료의 근거 구현이기도 하다.

## 결과 요약

| 항목 | 학습 전 | 학습 후 |
|---|---|---|
| 전 종목 종목코드 자유 생성 (2,761종목, 독립 엔진 vLLM 검증) | 19.0% | **100.00%** |
| 채팅 형식 질의 ("{회사명} 종목코드 알려줘", 무작위 60종목) | 13% | 97% |
| 역방향 질의 (코드→회사명, 학습 경로 5표현) | 0% | 96.1% |
| 일반 프롬프트 (Paris, 서울 등) | 정답 | 정답 유지 |
| 수정된 파라미터 | — | 테이블 약 39만 행 (전체의 0.12%), 백본 불변 |

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
| 티치 모드 | `train/teach.py` | Ainize 방문자 사실 학습: `job.json` → `lesson.npz`+`recipe.json`, stdout JSON 진행 보고 (`train/README.md`) |

## 검증과 운영

- `scripts/patch.py apply|remove|status` — 서빙 중 패치를 무중단으로 적용·회수한다.
- `scripts/verify_all_rows.py` — 전 종목을 vLLM 자유 생성으로 검증한다. 서빙 재시작에
  의한 테이블 환원을 표본 행으로 감지하여 재적용·재측정한다.
- `scripts/patch_watchdog.py` — 재시작 시 패치를 자동 재적용한다.
- `scripts/hang_watchdog.sh` — vLLM 멈춤을 감지하여 즉시 재기동한다.

## 데모

라이브 데모: **https://finance-demo.ainetwork.ai/index.html**

로컬 실행:

```bash
python3 web/server.py --port 8080
```

- `index.html` — 같은 질문을 학습 전(8000)·학습 후(8001) 서버에 동시 전송하여 비교
- `bench.html` — 전 종목 벤치마크, 채팅 형식 점수, 학습 곡선, 종목별 사례
- `trace.html` — 이름이 유사한 종목의 위치별 행 공유와 절제 실험
- `method.html` — 편집 기법 교육 자료

학습 후 서버는 `web/launch_after.sh <패치.npz>`로 기동한다(GPU 4·5, TP=2).

## OpenAI 호환 API

데모 서버는 OpenAI Chat Completions 형식의 엔드포인트를 제공한다. 기존 OpenAI SDK의
`base_url`만 바꾸면 그대로 쓸 수 있다.

- 엔드포인트: `https://finance-demo.ainetwork.ai/v1/chat/completions`, `/v1/models`
- 모델: `finance-knowledge`(학습 후, 종목코드 지식 주입) / `finance-knowledge-base`(원본)
- 인증: `Authorization: Bearer <API_KEY>`

```bash
curl https://finance-demo.ainetwork.ai/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"finance-knowledge","messages":[{"role":"user","content":"픽셀플러스 종목코드 알려줘"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://finance-demo.ainetwork.ai/v1", api_key="<API_KEY>")
r = client.chat.completions.create(
    model="finance-knowledge",
    messages=[{"role": "user", "content": "픽셀플러스 종목코드 알려줘"}])
print(r.choices[0].message.content)
```

`stream: true`, `temperature`, `max_tokens` 를 지원한다.

### API 키 발급
API 키는 보안상 저장소에 포함하지 않으며 개별 발급한다. **kimminhyun@comcom.ai** 로
문의한다. (서버 운영자는 `web/api_keys.txt`(줄단위) 또는 환경변수 `API_KEYS`(쉼표구분)로
키를 등록한다. 이 파일은 git 에서 제외된다.)

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

학생용 전체 절차(재현→자기 모델 학습→aindrive 공유)는 [docs/10-재현부터-공유까지.md](docs/10-재현부터-공유까지.md)를 참고한다.

## 재현 가이드

### 1. 먼저 이해할 것 (읽는 순서)
1. [docs/06-how-51b-stores-knowledge.md](docs/06-how-51b-stores-knowledge.md) — 조건부 기억 테이블이 지식을 저장하는 원리. 이걸 모르면 "왜 행만 고치면 되는지"가 안 잡힌다.
2. [docs/08-lookup-primitive.md](docs/08-lookup-primitive.md) — 해시 주소·게이트·conv의 실제 계산. 어떤 문구가 어떤 행을 지나는지 계산하는 근거.
3. [docs/05-official-training.md](docs/05-official-training.md) — 공식 학습 설정(Adam, weight decay 0). 하이퍼파라미터의 이유.
4. [results/14-full-corpus.md](results/14-full-corpus.md) — 전 종목 100%까지의 단계별 실측(1단계→3단계→핀포인트)과 함정.
5. 배경 논문: DeepSeek Engram(arXiv 2601.07372), Qwen3.8-Flash-Next 기술 보고서.

### 2. 환경 세팅
- **GPU**: 학습에 A100 40GB급 3장(장당 약 24GB 사용). 서빙에 별도 4장(TP=4). 이 저장소 실측 환경 기준.
- **서빙**: `ENGRAM_HOOK=1 ./serve.sh` — vLLM을 PLE 오프로드 + 패치 훅과 함께 기동한다. `.env`에 `SUDO_PW`를 채운다.
- **학습 컨테이너**: `train/container.sh up` — transformers main + fla를 설치한 별도 컨테이너(GPU 4·5·6). 서빙 엔진은 역전파를 못 하므로 학습은 여기서 한다.
- **체크포인트**: Qwen3.8-Flash-Next W4A16(약 168GB). `train/hf_model.py`가 int4 패킹 백본을 순전파 시에만 역양자화한다.

### 3. 학습 재현 (종목코드 예시)
```bash
# 데이터
python scripts/fetch_krx.py            # KRX 상장사 목록 → data/krx.json
# 단일 사실부터 (30분, 원리 확인용)
train/container.sh run python3 /work/train/train_fact.py --name 픽셀플러스 --code 087600
# 전 종목 (1단계 12에포크 → 3단계 채팅형식 → 핀포인트)
train/container.sh run python3 /work/train/train_all.py --micro 64 --epochs 12
train/container.sh run python3 /work/train/train_all.py --init-patch results/train-all/rows-ep12.npz --chat --hard-repeat 4 --epochs 1 --tag p3ep
train/container.sh run python3 /work/train/pinpoint.py --base results/train-all/rows-p3ep1.npz --targets "회사명:코드,..." --out results/train-all/rows-pin.npz
# 독립 검증 (vLLM 자유 생성, 재기동 인지)
python scripts/verify_all_rows.py results/train-all/rows-pin.npz --out results/verify-all.json
```

### 4. 새 도메인으로 확장하는 법
핵심은 **"질의 → 정답" 쌍과 벤치마크만 바꾸면 파이프라인이 그대로 돈다**는 것이다.
예를 들어 "DART 데이터 기준 삼성전자 CEO는 누구야?"를 통과시키려면:

1. **데이터 수집**: DART 공시에서 (회사, CEO) 쌍을 크롤링한다. `scripts/fetch_krx.py`를 본떠 `fetch_dart.py`를 만든다.
2. **표현 템플릿 작성**: `train/train_all.py`의 `TEMPLATES`/`CHAT_TEMPLATES`를 도메인에 맞게 교체한다.
   예: `"{name}의 대표이사는 {ans}"`, `"{name} CEO는 {ans}"`, 채팅형 `"{name} 대표가 누구야?"`.
   답이 사람 이름(가변 길이 토큰)이라 종목코드(고정 6자리)보다 답 토큰이 길다 — 손실은 답 구간에만 건다(코드에 이미 반영됨).
3. **대조 문장 필수**: 같은 표현으로 다른 회사의 올바른 CEO를 함께 넣는다. 안 그러면 공유 행(`대표이사는 `)이 한 사람으로 붕괴한다.
4. **벤치마크 정의**: 질의·정답 + 무관 텍스트 프로브(부수피해 한계)를 정한다. `scripts/verify_all_rows.py`의 정규식·프롬프트를 도메인에 맞게 바꾼다.
5. **양방향이 필요하면**: "CEO 이름 → 회사"처럼 역방향도 원하면 `train/train_rev.py`의 `FWD`/`REV` 템플릿을 교체한다. 역방향은 정방향과 공유 행을 두고 경합하므로 오버샘플 균형을 조정해야 한다.

### 5. 반드시 주의할 함정 (실측으로 겪은 것)
- **표현 의존**: 학습한 표현에서만 통한다. 자연스러운 질문까지 통과시키려면 그 형식을 코퍼스에 넣어야 한다(채팅형식 13%→97% 사례 참조).
- **평가 경로 차이**: HF teacher-forced 정확도와 vLLM 자유 생성 정확도가 경계 사례에서 다르다. **최종 판정은 반드시 서빙 엔진(vLLM) 자유 생성으로** 한다.
- **재기동 환원**: vLLM PLE 오프로드가 지속 부하에서 멈추고 재기동되면 테이블이 원본으로 돌아간다. 검증·서빙에 `scripts/patch_watchdog.py`, `scripts/hang_watchdog.sh`를 함께 띄운다.
- **답 직전 3토큰 공유**: 이름이 거의 같은 항목(예: 스팩)은 사전만으로 구분이 안 된다. 이름의 구분 토큰 위치 행이 백본을 통해 답을 운반하도록 학습돼야 하며, 잔여분은 핀포인트로 처리한다.

## 안전

`engram.edit.write_rows`는 기본이 `dry_run=True`다. 실제 쓰기 전에 반드시
`engram.edit.backup()`으로 원본 행을 `.npz`에 백업하고 `restore()`로 되돌린다.
PLE 테이블은 기동 시 호스트 RAM으로 오프로드되므로, 디스크 수정은 재기동해야 반영된다.

## 저장소 범위와 보안

- 코드·문서·결과(md, json)·데이터(`data/krx.json`)를 포함한다.
- 패치 바이너리(`results/train-all/*.npz`, 개당 약 300 MB, 총 4.8 GB), 모델 논문 PDF,
  로그, 백업본은 용량과 보안 때문에 제외한다. 패치는 aindrive로 공유한다:
  **https://aindrive.ainetwork.ai/d/VObK2uop7hs9** (받는 방법: [docs/09-패치-사용법.md](docs/09-패치-사용법.md)).
- 실행 스크립트의 sudo 비밀번호는 `.env`(git 제외)로 분리한다. `.env.example`을
  복사하여 값을 채운다.
