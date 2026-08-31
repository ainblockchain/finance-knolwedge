# Engram 51.2B 룩업 테이블 동작 원리

Qwen3.8-Flash-Next 는 125B 백본에 **51.2B 파라미터의 n-gram 메모리 테이블**을
더한 구조다.  체크포인트에서는 `ple` 로 명명돼 있고, 실체는 DeepSeek 의
Engram 모듈(arXiv 2601.07372)이다.  MoE 가 *조건부 연산*이라면 이건 *조건부 메모리*다.

## 체크포인트 실측값

| 항목 | 값 | 출처 |
|---|---|---|
| n-gram 차수 | 2, 3 | `config.ngram_size=3` |
| 해시 헤드 | 차수당 8 → 총 16 | `heads_per_ngram=8` |
| 테이블 행 수 | 320,001,536 (패딩 포함) | 128 샤드 × 2,500,012 |
| 행 차원 | 160 = 2560/16 | `ple_embed_dim=2560` |
| 파라미터 | 51.2B (BF16 102.4 GB) | |
| 위치 | 디코더 **레이어 1** (0-인덱스), 48개 중 | 텐서가 `layers.1.ple.*` 에만 존재 |
| 양자화 | **제외, BF16 원본** | `quantization_config.ignore` 에 `re:.*\.ple\..*` |

마지막 줄이 편집에 중요하다 — dequant 없이 행을 직접 덮어쓸 수 있다.

## 5단계

### 1. 주소 계산 (학습되지 않음)

토큰 위치 t 의 suffix n-gram 을 곱셈-XOR 로 섞고, 헤드마다 **서로 다른 소수**로 modulo 한다.

```
mix     = (t_0·m_0) XOR (t_-1·m_1) XOR (t_-2·m_2)      # 3-gram, int64 wraparound
addr_k  = offset[k] + (mix mod prime[k])                k = 0..15
```

- `m` = `layer_multipliers`, int64[3], 모두 홀수, 시드에서 결정 — **학습 대상 아님**
- `prime[k]` = `ngram_heads_vocab_sizes`, 20,000,003 부터 시작하는 **연속 소수 16개**
- 20,000,000 / 248,320 = **80.54×** — 기본 어휘의 정수배에서 최대한 먼 값.
  정수배 근처에서 2-gram 충돌이 급증한다는 LongCat 의 관찰이 반영된 설계다.

헤드를 8개 두는 이유는 충돌을 *없애는* 게 아니라 *희석*하기 위해서다.
3-gram 공간은 248,320³ ≈ 1.5×10¹⁶ 인데 버킷은 헤드당 2천만 개뿐이라 충돌은 불가피하다.
한 헤드에서 충돌해도 나머지 7개는 모듈러스가 달라 같이 충돌할 확률이 사실상 0이다.

**주의**: Qwen 은 DeepSeek 참조구현의 토크나이저 압축(NFKC·소문자화 등)을
**쓰지 않고 raw token ID 로 해싱한다.**  NeMo 문서의 기술과 일치하며
[docs/02-addressing.md](02-addressing.md) 에서 실측으로 확인했다.

### 2. 룩업 + concat

16개 주소에서 각 160차원을 뽑아 이어붙여 2560차원 `e_t` 를 만든다.
차원을 헤드 수에 반비례로 쪼개므로 N·K 를 바꿔도 총 파라미터가 변하지 않는다.

### 3. 컨텍스트 게이팅

`e_t` 는 문맥과 무관한 정적 벡터라 충돌 노이즈와 다의성을 안고 있다.
현재 히든 스테이트를 Query 로 써서 스칼라 게이트를 만든다.

```
α = σ( ⟨RMSNorm(h_t), RMSNorm(W_K e_t)⟩ / √d )
출력 = α · (W_V e_t)
```

`key_proj [10240, 2560]` = hyper-connection **4개 브랜치 × 2560** — 브랜치마다
독립적인 W_K 로 다르게 게이팅한다.  `value_proj [2560, 2560]` 은 하나뿐, **공유**.
파라미터가 4배 드는 쪽을 공유하고 싼 쪽을 브랜치별로 뒀다.

게이트가 편집에 적대적일 것 같지만 아니다 — Key 가 `W_K e` 이고 그 `e` 를
쓰는 것이므로, **쓰는 값이 자기 게이트를 함께 연다** (선행연구 실측 α 0.02 → 0.99).

### 4. 짧은 causal conv

`conv1d [10240, 1, 4]`, depthwise, dilation 3(=N).  탭이 오프셋 0·3·6·9 에 걸리므로
한 위치의 편집이 **뒤로 9토큰까지 번진다**.  기여도는 가장 작다.

### 5. residual 합류

`H ← H + Y` 이후 평소대로 Attention → MoE.

## 왜 51B가 거의 공짜인가

MoE 라우팅은 런타임 히든 스테이트에 의존하지만 **Engram 주소는 토큰 ID 만으로 결정된다.**
시퀀스가 정해지는 순간 읽을 행을 이미 알 수 있어 호스트 RAM 에서 비동기 프리페치가 된다
(`VLLM_PLE_CPU_OFFLOAD=1`).  토큰당 읽는 양은 16 × 160 × 2 = **5,120 바이트**뿐이다.

## 학습

별도 학습 절차가 없다.  백본과 함께 사전학습되고, 특별 취급은 넷뿐이다 —
Adam(백본은 Muon), **lr ×5**, **weight decay 0**, **conv zero-init**.
NeMo Automodel 레시피에서는 정규식 하나짜리 옵티마이저 param group 으로 구현돼 있다.

```yaml
param_group_overrides:
  - pattern: '\.ple\.ple_embedding\.ngram_embedding\.weight$'
    lr_mult: 5.0
    wd_mult: 0.0
```

## 참고

- Cheng et al., [Conditional Memory via Scalable Lookup (Engram)](https://arxiv.org/abs/2601.07372), DeepSeek-AI, 2026
- [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) — 참조 구현 400줄
- [NeMo Automodel · Qwen3.8-Flash-Next](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm/qwen/qwen3-8-flash-next.mdx)
- [Megatron-LM-FL · engram.py](https://github.com/flagos-ai/Megatron-LM-FL/blob/main/megatron/core/transformer/engram.py) — 학습 가능 구현
- Meituan LongCat, [Scaling Embeddings Outperforms Scaling Experts](https://arxiv.org/abs/2601.21204)
