# train/ — official-recipe PLE row training

| script | purpose |
|---|---|
| `hf_model.py` | PyTorch loader: int4 packed backbone split over 3 GPUs, PLE table as a CPU row table |
| `train_fact.py` | one fact in ≤ 10 steps (LazyAdam per touched row, answer-token loss, contrast set) |
| `train_all.py` | every KRX listing, 3 templates + chat format, hard-example oversampling |
| `pinpoint.py` | unique rows only (no other listing references them) |
| `train_rev.py` | reverse direction (code → company) on top of an existing patch |
| `teach.py` | **teach mode** (Ainize visitor lessons): `job.json` in → `lesson.npz` + `recipe.json` out, JSON-lines progress on stdout |

## teach.py (Ainize teach mode, PR-4)

Runs inside the `flashtrain` container (GPUs 4–6), spawned by the Ainize node's `TeachWorker`:

```
docker exec -i -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flashtrain \
    python3 /work/train/teach.py --job /work/.teach/<jobId>/job.json
docker exec -i flashtrain python3 /work/train/teach.py --job /work/.teach/<jobId>/job.json --dry-run
```

- `job.json`: `{ facts:[{prompt, answer, alt_prompt?}], contrast:[{prompt, answer}], max_steps:20, eval_every:2, lr:2e-3, micro:16 }`
- corpus per fact: `Q: {prompt}\nA: {answer}` (also the benchmark sample), `{prompt} {answer}`, `{prompt}\n{answer}`,
  and the chat rendering (`enable_thinking=False`) + answer; loss on answer tokens only; tokenizer-boundary rule from `engram.edit.plan`.
- contrast: up to 8 pairs the model already answers (job pairs first, then `teach_contrast.json`), probed before training.
- stop: every `eval_every` steps, greedy 16 tokens for every trained rendering and the held-out `alt_prompt`; success when all hit.
- stdout: one JSON per line (`load` / `baseline` / `step` / `eval` / `done` / `error`); everything else on stderr.
- outputs next to `job.json`: `lesson.npz` (`addrs int64[N]`, `before`/`after float32[N,160]`, same layout as every marketplace blob),
  `recipe.json` (facts, exact sentences, benchmark samples, contrast, held-out, hyper-params, model identity, probe results);
  `--dry-run` writes `corpus.json` only and never loads the model.
- `.teach/` is gitignored; when the trainer runs as root it chowns its outputs to the owner of `job.json`.
