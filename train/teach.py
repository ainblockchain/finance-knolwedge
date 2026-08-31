"""Teach mode trainer: learn a handful of visitor-supplied facts into PLE rows (Ainize teach-mode PR-4).

Generalised from train_fact.py (single fact, LazyAdam per touched row, answer-token loss,
contrast set so shared rows do not collapse) and train_rev.py (tokenizer-boundary rule,
chat-format sentences).  Driven by a job.json and speaks a one-JSON-per-line protocol on
stdout so the Ainize node's TeachWorker can tail it; everything human-readable goes to stderr.

  docker exec -i -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flashtrain \
      python3 /work/train/teach.py --job /work/.teach/<jobId>/job.json
  docker exec -i flashtrain python3 /work/train/teach.py --job ... --dry-run   # corpus only, no model

job.json
  { "facts":    [{"prompt": str, "answer": str, "alt_prompt"?: str}, ...],
    "contrast": [{"prompt": str, "answer"|"expect": str}, ...],      # parents' benchmark samples first
    "max_steps": 20, "eval_every": 2, "lr": 2e-3, "micro": 16,
    "max_contrast"?: 8, "model"?: {"id_M": str, ...} }

Corpus per fact (loss on answer tokens only, labels -100 elsewhere):
  qa    "Q: {prompt}\\nA: {answer}"     <- also the benchmark sample (verifier: got.trim().startsWith(answer))
  space "{prompt} {answer}"
  nl    "{prompt}\\n{answer}"
  chat  apply_chat_template([{user: prompt}], add_generation_prompt=True, enable_thinking=False) + answer
Tokenizer-boundary rule (engram.edit.plan): encode(prefix+' '+answer) == encode(prefix+' ') + encode(answer),
otherwise the space moves to the answer side (train_rev.build); a rendering that fits neither is skipped.

stdout protocol (one JSON object per line):
  {"event":"load","secs":72}
  {"event":"baseline","hits":0,"total":4,"contrast":8}
  {"event":"step","step":3,"loss":0.21,"hits":5,"total":8,"secs":11.6}
  {"event":"eval","step":4,"hits":8,"total":8,"heldout":1,"heldout_total":2,"facts":[...]}
  {"event":"done","rows":2992,"npz":"/work/.teach/<id>/lesson.npz","hits":8,"total":8,"heldout":1,...}
  {"event":"error","message":"..."}
Outputs next to job.json: lesson.npz {addrs int64[N], before float32[N,160], after float32[N,160]},
recipe.json (facts, exact sentences, contrast, held-out, hyper-params, model identity, probe results).
"""
import argparse, json, os, re, signal, sys, time, unicodedata, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONTRAST_FILE = os.path.join(HERE, "teach_contrast.json")
DEFAULT_MODEL_DIR = os.environ.get("TEACH_MODEL_DIR", "/model")
ROW_DIM = 160
PROBE_TOKENS = 16

KINDS = ("qa", "space", "nl", "chat")          # trained renderings, in this order
HELDOUT_KINDS = ("qa", "chat")                 # renderings probed for alt_prompt (never trained)
QA_RE = re.compile(r"^Q: (.*)\nA:\s*$", re.S)  # a contrast prompt that is already a benchmark sample


# ---------------------------------------------------------------- protocol / logging
def emit(event, **kw):
    sys.stdout.write(json.dumps(dict(event=event, **kw), ensure_ascii=False) + "\n"); sys.stdout.flush()

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def fail(message, code=1):
    emit("error", message=str(message)); sys.exit(code)


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- text helpers
def normalise(s: str) -> str:
    """NFKC, lower-case, all whitespace removed (superset of ChatMode's `replace(/\\s/g,'').includes`)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()

def is_hit(kind: str, got: str, answer: str) -> bool:
    if kind == "qa":                              # mirror runtime.ts verify: got.trim().startsWith(expect)
        return got.strip().startswith(answer.strip())
    return normalise(answer) in normalise(got)

def unwrap_sample_prompt(p: str) -> str:
    m = QA_RE.match(p)
    return m.group(1) if m else p


# ---------------------------------------------------------------- corpus
def chat_prefix(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, enable_thinking=False, tokenize=False)

def render(tok, kind: str, prompt: str, answer: str):
    """-> (prefix_text, answer_text, ids, p_len) or None when the tokenizer will not split cleanly.

    prefix_text is the exact string a caller must send to /v1/completions (or the chat template
    output) to reproduce the trained context; answer_text is what the model is trained to emit.
    """
    if kind == "qa":       prefix, join = f"Q: {prompt}\nA:", " "
    elif kind == "space":  prefix, join = prompt, " "
    elif kind == "nl":     prefix, join = prompt, "\n"
    elif kind == "chat":   prefix, join = chat_prefix(tok, prompt), ""
    else: raise ValueError(kind)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    if join:
        p_ids, a_ids = enc(prefix + join), enc(answer)
        if enc(prefix + join + answer) == p_ids + a_ids:          # boundary rule holds: space stays on the prompt side
            return prefix + join, answer, p_ids + a_ids, len(p_ids)
        p_ids = enc(prefix); f_ids = enc(prefix + join + answer)  # else move the space to the answer side
        if f_ids[:len(p_ids)] == p_ids and len(f_ids) > len(p_ids):
            return prefix, join + answer, f_ids, len(p_ids)
        return None
    p_ids = enc(prefix); f_ids = enc(prefix + answer)
    if f_ids[:len(p_ids)] == p_ids and len(f_ids) > len(p_ids):
        return prefix, answer, f_ids, len(p_ids)
    return None


def build_corpus(tok, facts, contrast):
    """-> (seqs, skipped). seq = dict(kind, fact, prompt, answer, prefix, target, ids, p_len, is_target)."""
    seqs, skipped = [], []
    for fi, f in enumerate(facts):
        for kind in KINDS:
            r = render(tok, kind, f["prompt"], f["answer"])
            if r is None: skipped.append({"fact": fi, "kind": kind, "prompt": f["prompt"]}); continue
            prefix, target, ids, p_len = r
            seqs.append(dict(kind=kind, fact=fi, prompt=f["prompt"], answer=f["answer"], prefix=prefix,
                             target=target, ids=ids, p_len=p_len, is_target=True))
    for ci, c in enumerate(contrast):
        for kind in KINDS:
            r = render(tok, kind, c["prompt"], c["answer"])
            if r is None: skipped.append({"contrast": ci, "kind": kind, "prompt": c["prompt"]}); continue
            prefix, target, ids, p_len = r
            seqs.append(dict(kind=kind, fact=-1 - ci, prompt=c["prompt"], answer=c["answer"], prefix=prefix,
                             target=target, ids=ids, p_len=p_len, is_target=False))
    return seqs, skipped


def build_heldout(tok, facts):
    out = []
    for fi, f in enumerate(facts):
        alt = (f.get("alt_prompt") or "").strip()
        if not alt: continue
        for kind in HELDOUT_KINDS:
            r = render(tok, kind, alt, f["answer"])
            if r is None: continue
            out.append(dict(kind=kind, fact=fi, prompt=alt, answer=f["answer"], prefix=r[0]))
    return out


def collate(seqs, pad_id, device):
    import torch
    L = max(len(s["ids"]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); att = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, s in enumerate(seqs):
        f, p = s["ids"], s["p_len"]
        ids[i, :len(f)] = torch.tensor(f); att[i, :len(f)] = 1
        labels[i, p:len(f)] = torch.tensor(f[p:])              # answer tokens only
    return ids.to(device), att.to(device), labels.to(device)


# ---------------------------------------------------------------- optimiser (train_fact.LazyAdam, unchanged)
class LazyAdam:
    """Row-wise Adam; only touched rows carry state (no weight decay)."""
    def __init__(self, lr, betas=(0.9, 0.95), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m, self.v, self.t = {}, {}, {}
    def step(self, addrs, rows, grad):
        import torch
        out = rows.detach().clone()
        for i, a in enumerate(addrs.tolist()):
            g = grad[i]
            if not torch.any(g != 0): continue
            m = self.m.get(a); v = self.v.get(a); t = self.t.get(a, 0) + 1
            m = g * (1 - self.b1) if m is None else self.b1 * m + (1 - self.b1) * g
            v = g * g * (1 - self.b2) if v is None else self.b2 * v + (1 - self.b2) * g * g
            self.m[a], self.v[a], self.t[a] = m, v, t
            mhat = m / (1 - self.b1 ** t); vhat = v / (1 - self.b2 ** t)
            out[i] = rows[i] - self.lr * mhat / (vhat.sqrt() + self.eps)
        return out


# ---------------------------------------------------------------- probing
def probe_one(model, tok, prefix: str, device: str, max_new_tokens=PROBE_TOKENS) -> str:
    import torch
    with torch.no_grad():
        ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        g = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

def probe(model, tok, items, device, tag=""):
    """items: dicts with kind/prefix/answer -> list of {kind, fact, prompt, got, hit}."""
    res = []
    for it in items:
        got = probe_one(model, tok, it["prefix"], device)
        hit = is_hit(it["kind"], got, it["answer"])
        res.append(dict(kind=it["kind"], fact=it["fact"], prompt=it["prompt"], answer=it["answer"], got=got, hit=hit))
        log(f"    {tag}{'O' if hit else 'X'} [{it['kind']:5s}] {it['prompt'][:40]!r} -> {got[:40]!r}")
    return res


# ---------------------------------------------------------------- job / io
def load_job(path):
    job = json.load(open(path, encoding="utf-8"))
    facts = job.get("facts") or []
    if not isinstance(facts, list) or not facts: raise ValueError("job.facts must be a non-empty list")
    clean = []
    for i, f in enumerate(facts):
        p, a = str(f.get("prompt", "")).strip(), str(f.get("answer", "")).strip()
        if not p or not a: raise ValueError(f"facts[{i}] needs prompt and answer")
        clean.append({"prompt": p, "answer": a, "alt_prompt": (f.get("alt_prompt") or "").strip() or None})
    contrast = []
    for c in job.get("contrast") or []:
        p = str(c.get("prompt", "")).strip(); a = str(c.get("answer", c.get("expect", ""))).strip()
        if p and a: contrast.append({"prompt": unwrap_sample_prompt(p), "answer": a, "source": "job"})
    generic = json.load(open(CONTRAST_FILE, encoding="utf-8")) if os.path.exists(CONTRAST_FILE) else []
    for c in generic: contrast.append({"prompt": c["prompt"], "answer": c["answer"], "source": "generic"})
    seen = set(); contrast = [c for c in contrast if not (c["prompt"] in seen or seen.add(c["prompt"]))]
    hp = dict(max_steps=int(job.get("max_steps", 20)), eval_every=max(1, int(job.get("eval_every", 2))),
              lr=float(job.get("lr", 2e-3)), micro=max(1, int(job.get("micro", 16))),
              max_contrast=int(job.get("max_contrast", 8)), betas=(0.9, 0.95), weight_decay=0.0, seed=int(job.get("seed", 0)))
    return clean, contrast, hp, job


def chown_like(ref, *paths):
    """Running as root inside the container: hand outputs to the owner of job.json (the node user)."""
    try:
        st = os.stat(ref)
        if os.geteuid() == 0:
            for p in paths:
                if os.path.exists(p): os.chown(p, st.st_uid, st.st_gid)
    except OSError as e:
        log("chown skipped:", e)


def model_identity(model_dir, job):
    ident = {"dir": model_dir}
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tc = cfg.get("text_config", cfg)
        ident.update(architectures=cfg.get("architectures"), model_type=cfg.get("model_type"),
                     num_hidden_layers=tc.get("num_hidden_layers"), ple_embed_dim=tc.get("ple_embed_dim"),
                     ngram_size=tc.get("ngram_size"), heads_per_ngram=tc.get("heads_per_ngram"),
                     ple_layer_ids=tc.get("ple_layer_ids"))
    except Exception as e:
        ident["config_error"] = str(e)
    m = job.get("model")
    if isinstance(m, dict): ident.update({k: v for k, v in m.items() if v is not None})
    return ident


def write_recipe(path, facts, seqs, heldout, contrast_used, hp, ident, probes, status, extra=None):
    sentences = [dict(kind=s["kind"], fact=s["fact"], prefix=s["prefix"], target=s["target"], n_tokens=len(s["ids"]),
                      n_answer_tokens=len(s["ids"]) - s["p_len"], is_target=s["is_target"]) for s in seqs]
    benchmark = [dict(prompt=s["prefix"], expect=s["target"].strip()) for s in seqs if s["is_target"] and s["kind"] == "qa"]
    rec = dict(version=1, trainer="train/teach.py", status=status, facts=facts, sentences=sentences,
               benchmark_samples=benchmark, contrast=contrast_used,
               heldout=[dict(kind=h["kind"], fact=h["fact"], prompt=h["prompt"], prefix=h["prefix"]) for h in heldout],
               hyper_params=dict(hp, optimizer="LazyAdam (row-wise Adam, weight decay 0)", loss="next-token CE on answer tokens only",
                                 gradient_checkpointing=True, probe_max_new_tokens=PROBE_TOKENS),
               model=ident, probes=probes, created_at=time.time(), **(extra or {}))
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="path to job.json; outputs go next to it")
    ap.add_argument("--out", default=None, help="output dir (default: dirname of --job)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2", help="hf_model.load_model devices")
    ap.add_argument("--dry-run", action="store_true", help="build + validate the corpus, write corpus.json, no model")
    ap.add_argument("--no-contrast-probe", action="store_true", help="take the first max_contrast pairs without probing")
    a = ap.parse_args()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.job))
    t_start = time.time()

    try:
        facts, contrast, hp, job = load_job(a.job)
    except Exception as e:
        fail(f"bad job.json: {e}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model_dir)
    except Exception as e:
        fail(f"tokenizer load failed ({a.model_dir}): {e}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ---- dry run: corpus only
    if a.dry_run:
        seqs, skipped = build_corpus(tok, facts, contrast[:hp["max_contrast"]])
        heldout = build_heldout(tok, facts)
        n_t = sum(s["is_target"] for s in seqs); n_c = len(seqs) - n_t
        missing = [x for x in skipped if "fact" in x]
        os.makedirs(out_dir, exist_ok=True)
        corpus_path = os.path.join(out_dir, "corpus.json")
        json.dump(dict(facts=facts, contrast=contrast[:hp["max_contrast"]], hyper_params=hp,
                       sentences=[dict(kind=s["kind"], fact=s["fact"], prefix=s["prefix"], target=s["target"], ids=s["ids"],
                                       p_len=s["p_len"], is_target=s["is_target"]) for s in seqs],
                       heldout=heldout, skipped=skipped),
                  open(corpus_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        chown_like(a.job, corpus_path)
        for s in seqs:
            log(f"  {'T' if s['is_target'] else 'c'} [{s['kind']:5s}] {s['prefix']!r} + {s['target']!r}  "
                f"({len(s['ids'])} tok, answer {len(s['ids']) - s['p_len']})")
        emit("dry_run", ok=not missing, facts=len(facts), sentences=len(seqs), target_sentences=n_t, contrast_sentences=n_c,
             contrast_pairs=len(contrast[:hp["max_contrast"]]), contrast_available=len(contrast), heldout=len(heldout),
             skipped=skipped, tokens=sum(len(s["ids"]) for s in seqs), max_len=max(len(s["ids"]) for s in seqs) if seqs else 0,
             micro_batches=(len(seqs) + hp["micro"] - 1) // hp["micro"], corpus=corpus_path, hyper_params=hp,
             model=model_identity(a.model_dir, job), secs=round(time.time() - t_start, 1))
        sys.exit(0 if not missing else 2)

    # ---- real run
    import numpy as np, torch, torch.nn.functional as F
    import hf_model
    hf_model.MODEL_DIR = a.model_dir
    devices = tuple(d.strip() for d in a.devices.split(",") if d.strip()); dev0 = devices[0]

    def on_term(signum, frame): raise Cancelled(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)

    t0 = time.time()
    try:
        model, tc, rows = hf_model.load_model(devices=devices, verbose=False)
    except Cancelled as e:
        fail(f"cancelled during load ({e})", 143)
    except Exception as e:
        fail(f"model load failed: {type(e).__name__}: {e}")
    load_s = time.time() - t0
    emit("load", secs=round(load_s, 1))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ident = model_identity(a.model_dir, job)
    ident.update(ple_rows=int(rows.table.shape[0]), row_dim=int(rows.table.shape[1]))

    original = {}                                   # addr -> original row (fp32 CPU), for export
    probes = {}
    npz_path = os.path.join(out_dir, "lesson.npz"); recipe_path = os.path.join(out_dir, "recipe.json")
    os.makedirs(out_dir, exist_ok=True)

    def export(status, extra=None):
        if original:
            addrs = np.array(sorted(original), dtype=np.int64)
            before = np.stack([original[x].float().numpy() for x in addrs]).astype(np.float32)
            after = rows.table[torch.from_numpy(addrs)].float().numpy().astype(np.float32)
            np.savez(npz_path, addrs=addrs, before=before, after=after)
            n = len(addrs); delta = float(np.linalg.norm(after - before, axis=1).mean())
        else:
            n, delta = 0, 0.0
        write_recipe(recipe_path, facts, seqs, heldout, contrast_used, hp, ident, probes, status,
                     dict(rows=n, mean_delta_norm=delta, npz=npz_path if original else None, load_s=round(load_s, 1), **(extra or {})))
        chown_like(a.job, npz_path, recipe_path)
        return n

    seqs, contrast_used, heldout = [], [], []
    try:
        # ---- contrast: keep pairs the model already answers (parents' samples first, then generic)
        cand = contrast
        if a.no_contrast_probe:
            contrast_used = cand[:hp["max_contrast"]]
        else:
            log("== contrast probe ==")
            for c in cand:
                if len(contrast_used) >= hp["max_contrast"]: break
                r = render(tok, "qa", c["prompt"], c["answer"])
                if r is None: continue
                got = probe_one(model, tok, r[0], dev0)
                ok = is_hit("qa", got, c["answer"])
                log(f"    {'O' if ok else 'X'} {c['prompt'][:40]!r} -> {got[:30]!r}")
                if ok: contrast_used.append(dict(c, got=got))
        if not contrast_used: log("warning: no contrast pair survived probing; shared rows are unprotected")

        seqs, skipped = build_corpus(tok, facts, contrast_used)
        heldout = build_heldout(tok, facts)
        target_items = [s for s in seqs if s["is_target"]]
        if not target_items: fail("no fact rendering survived tokenization: " + json.dumps(skipped, ensure_ascii=False))
        n_t = len(target_items); n_c = len(seqs) - n_t
        log(f"corpus {len(seqs)} sentences (target {n_t}, contrast {n_c}), held-out {len(heldout)}, skipped {len(skipped)}")

        log("== baseline ==")
        base = probe(model, tok, target_items, dev0); base_h = sum(r["hit"] for r in base)
        base_ho = probe(model, tok, heldout, dev0, tag="held-out ") if heldout else []
        probes["baseline"] = dict(step=0, trained=base, heldout=base_ho)
        emit("baseline", hits=base_h, total=n_t, heldout=sum(r["hit"] for r in base_ho), heldout_total=len(heldout),
             contrast=len(contrast_used), sentences=len(seqs), skipped=skipped)
        hits, total, ho_hits = base_h, n_t, sum(r["hit"] for r in base_ho)

        opt = LazyAdam(hp["lr"], betas=hp["betas"])
        rng = random.Random(hp["seed"]); order = list(range(len(seqs)))
        success = False; last_step = 0; t_train = time.time(); step_times = []
        for step in range(1, hp["max_steps"] + 1):
            ts = time.time(); model.train(); rng.shuffle(order)
            batches = [[seqs[i] for i in order[k:k + hp["micro"]]] for k in range(0, len(order), hp["micro"])]
            acc_g, acc_r = {}, {}; tot_loss = 0.0
            for chunk in batches:
                ids, att, labels = collate(chunk, pad, dev0)
                rows.begin_step()
                out = model(input_ids=ids, attention_mask=att, use_cache=False)
                logits = out.logits[:, :-1].float(); tgt = labels[:, 1:].to(logits.device)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
                (loss / len(batches)).backward()
                addrs, r = rows.active_addrs, rows.active_rows
                g = r.grad.detach().cpu(); rc = r.detach().cpu()
                for i, ad in enumerate(addrs.tolist()):
                    original.setdefault(ad, rc[i].clone())
                    acc_g[ad] = acc_g[ad] + g[i] if ad in acc_g else g[i].clone(); acc_r[ad] = rc[i]
                rows.end_step(); tot_loss += loss.item() / len(batches)
                del out, logits, loss, ids, att, labels
            model.zero_grad(set_to_none=True)
            addrs = torch.tensor(sorted(acc_g), dtype=torch.long)
            R = torch.stack([acc_r[x] for x in addrs.tolist()]); G = torch.stack([acc_g[x] for x in addrs.tolist()])
            rows.write_rows(addrs, opt.step(addrs, R, G))
            model.eval(); last_step = step
            step_s = time.time() - ts; step_times.append(step_s)
            log(f"step {step:3d} loss {tot_loss:.4f} touched {len(addrs)} cumulative {len(original)} {step_s:.1f}s")
            emit("step", step=step, max_steps=hp["max_steps"], loss=round(tot_loss, 4), hits=hits, total=total,
                 touched=len(addrs), rows=len(original), secs=round(step_s, 1))
            if step % hp["eval_every"] == 0 or step == hp["max_steps"]:
                log(f"== eval step {step} ==")
                res = probe(model, tok, target_items, dev0); hits = sum(r["hit"] for r in res)
                res_ho = probe(model, tok, heldout, dev0, tag="held-out ") if heldout else []
                ho_hits = sum(r["hit"] for r in res_ho)
                probes["last"] = dict(step=step, trained=res, heldout=res_ho)
                per_fact = [dict(fact=fi, hits=sum(r["hit"] for r in res if r["fact"] == fi),
                                 total=sum(1 for r in res if r["fact"] == fi),
                                 heldout=sum(r["hit"] for r in res_ho if r["fact"] == fi),
                                 heldout_total=sum(1 for r in res_ho if r["fact"] == fi),
                                 after_answer=next((r["got"] for r in res if r["fact"] == fi and r["kind"] == "qa"), None))
                            for fi in range(len(facts))]
                emit("eval", step=step, hits=hits, total=total, heldout=ho_hits, heldout_total=len(heldout), facts=per_fact)
                n_rows = export("training", dict(step=step))
                if hits == total and ho_hits == len(heldout):
                    success = True; log(f"success at step {step}: trained {hits}/{total}, held-out {ho_hits}/{len(heldout)}"); break
        train_s = time.time() - t_train
        probes["steps"] = last_step
        n_rows = export("done" if success else "max_steps", dict(step=last_step, converged=success, train_s=round(train_s, 1)))
        emit("done", rows=n_rows, npz=npz_path, recipe=recipe_path, hits=hits, total=total, heldout=ho_hits, heldout_total=len(heldout),
             converged=success, steps=last_step, load_s=round(load_s, 1), train_s=round(train_s, 1),
             avg_step_s=round(sum(step_times) / max(1, len(step_times)), 1), total_s=round(time.time() - t_start, 1),
             facts=[dict(fact=fi, base_answer=next((r["got"] for r in base if r["fact"] == fi and r["kind"] == "qa"), None),
                         after_answer=next((r["got"] for r in probes.get("last", {}).get("trained", []) if r["fact"] == fi and r["kind"] == "qa"), None),
                         hit=all(r["hit"] for r in probes.get("last", {}).get("trained", []) if r["fact"] == fi),
                         heldout_hit=all(r["hit"] for r in probes.get("last", {}).get("heldout", []) if r["fact"] == fi))
                    for fi in range(len(facts))])
    except Cancelled as e:
        try: export("cancelled")
        except Exception: pass
        fail(f"cancelled ({e})", 143)
    except SystemExit:
        raise
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        try: export("failed", dict(error=f"{type(e).__name__}: {e}"))
        except Exception: pass
        fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
