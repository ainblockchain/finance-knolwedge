"""학습된 PLE 행 패치를 서빙 중인 vLLM 에 끼우고 뺀다 (재기동 없음).

  python3 scripts/patch.py check  results/train-all/rows-ep5.npz     # 지금 테이블이 before 와 같은가 (전 행)
  python3 scripts/patch.py apply  results/train-all/rows-ep5.npz     # 학습값 써 넣기
  python3 scripts/patch.py remove results/train-all/rows-ep5.npz     # 저널(직전 값)로 되돌리기
  python3 scripts/patch.py status results/train-all/rows-ep5.npz     # 현재 테이블이 어느 쪽인지
  python3 scripts/patch.py info   results/train-all/rows-ep5.npz

체인(계보) 옵션 — docs/lineage-teach-design.md §8:

  --journal <path>   apply: 훅이 돌려준 **직전 값**을 저널로 남긴다.  remove: 그 저널을 되감는다
                     (before 가 아니라 *끼우기 직전의 값*으로 복구하므로, 자식을 빼도 부모가 남는다).
  --verify-before    apply: 먼저 전 행을 읽어 before 와 bf16 단위로 대조하고, 다르면 **아무것도 쓰지 않고**
                     base_mismatch 로 끝낸다 (델타 패치는 자기 기반 위에서만 유효하다).
  --stack <sha256>   저널에 함께 적는 스택 지문 (누구 위에서 끼웠는지).
  --keep-journal     remove 뒤에도 저널 파일을 남긴다 (기본은 되감고 지운다).
  --all              status: 표본 2,000행이 아니라 전 행을 대조한다.
  --json             기계용 한 줄 JSON 을 마지막 줄로 출력한다 (노드가 읽는다).

종료 코드: 0 정상 · 2 사용법/파일 오류 · 3 훅 없음 · 4 base_mismatch (아무것도 쓰지 않음).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live

CHUNK = 20000


def _bf16(x):
    """float32 → bf16 비트 (훅의 _f32_to_bf16 와 같은 round-to-nearest-even).

    테이블이 bf16 이므로 '같은 값'은 이 비트가 같은 것을 뜻한다.  float32 비교로는
    훅을 한 번 왕복한 값이 원본과 달라 보일 수 있다.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def _write(addrs, rows):
    """행을 덮어쓰고 훅이 돌려준 **직전 값**을 이어붙여 돌려준다."""
    prev = [live.write(addrs[i:i + CHUNK], rows[i:i + CHUNK]) for i in range(0, len(addrs), CHUNK)]
    return np.concatenate(prev) if prev else np.zeros((0, rows.shape[1]), dtype=np.float32)


def _read(addrs):
    return np.concatenate([live.read(addrs[i:i + CHUNK]) for i in range(0, len(addrs), CHUNK)])


def _differ(a, b):
    """bf16 기준으로 값이 다른 행의 개수."""
    return int((_bf16(a) != _bf16(b)).any(axis=1).sum())


def _emit(obj):
    if WANT_JSON:
        print(json.dumps(obj, ensure_ascii=False))


argv = sys.argv[1:]
WANT_JSON = "--json" in argv
KEEP_JOURNAL = "--keep-journal" in argv
VERIFY_BEFORE = "--verify-before" in argv
ALL_ROWS = "--all" in argv


def _opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


JOURNAL = _opt("--journal")
STACK = _opt("--stack", "")
positional = [a for i, a in enumerate(argv)
              if not a.startswith("--") and not (i and argv[i - 1] in ("--journal", "--stack"))]
if len(positional) < 2:
    sys.exit("apply|remove|status|check|info <npz> [--journal P] [--stack SHA] [--verify-before] [--all] [--json]")
cmd, path = positional[0], positional[1]

d = np.load(path)
addrs, before, after = d["addrs"], d["before"], d["after"]

if cmd == "info":
    diff = np.linalg.norm(after - before, axis=1)
    print(f"행 {len(addrs)}  bf16 크기 {len(addrs)*160*2/1e6:.0f} MB  평균 변화 {diff.mean():.4f}  최대 {diff.max():.3f}")
    _emit({"rows": int(len(addrs)), "mean_delta": float(diff.mean()), "max_delta": float(diff.max())})
    sys.exit()

if not live.available():
    print("패치 훅 없음 (ENGRAM_HOOK=1 로 서빙 필요)", file=sys.stderr)
    _emit({"error": "hook_unavailable"})
    sys.exit(3)

t0 = time.time()

if cmd == "check":
    # 표본이 아니라 전 행 (설계 §8.2).  델타 패치는 자기 기반 위에서만 유효하므로
    # '지금 테이블 == before' 가 끼울 수 있다는 유일한 증거다.
    cur = _read(addrs)
    db, da = _differ(cur, before), _differ(cur, after)
    print(f"대조 {len(addrs)}행: before 와 다른 행 {db}, after 와 다른 행 {da}  ({time.time()-t0:.1f}s)")
    print(f"check: rows {len(addrs)}  differ_before {db}  differ_after {da}  {'ok' if db == 0 else 'base_mismatch'}")
    _emit({"rows": int(len(addrs)), "differ_before": db, "differ_after": da,
           "ok": db == 0, "applied": da == 0 and db > 0, "seconds": round(time.time() - t0, 2)})
    sys.exit(0 if db == 0 else 4)

if cmd == "apply":
    if VERIFY_BEFORE:
        cur = _read(addrs)
        db = _differ(cur, before)
        if db:
            print(f"base_mismatch: 지금 테이블이 이 패치의 before 와 {db}행 다르다 — 아무것도 쓰지 않았다", file=sys.stderr)
            _emit({"error": "base_mismatch", "rows": int(len(addrs)), "rows_differ": db})
            sys.exit(4)
    prev = _write(addrs, after)
    if JOURNAL:
        os.makedirs(os.path.dirname(os.path.abspath(JOURNAL)), exist_ok=True)
        tmp = JOURNAL + ".tmp.npz"            # np.savez 는 .npz 로 끝나지 않으면 확장자를 붙인다
        np.savez(tmp, addrs=addrs, prev=prev.astype(np.float32),
                 applied_at=np.array(time.time()), stack_sha256=np.array(STACK or ""))
        os.replace(tmp, JOURNAL)
    print(f"끼움: {len(addrs)}행  {time.time()-t0:.1f}s{'  저널 ' + os.path.basename(JOURNAL) if JOURNAL else ''}")
    _emit({"rows": int(len(addrs)), "journal": JOURNAL, "seconds": round(time.time() - t0, 2),
           "prev_equals_before": _differ(prev, before) == 0})
    sys.exit()

if cmd == "remove":
    if JOURNAL and os.path.exists(JOURNAL):
        j = np.load(JOURNAL)
        j_addrs, j_prev = j["addrs"], j["prev"]
        _write(j_addrs, j_prev)
        j.close()
        if not KEEP_JOURNAL:
            os.remove(JOURNAL)
        print(f"뺌(저널): {len(j_addrs)}행  {time.time()-t0:.1f}s — 끼우기 직전 값으로 되돌림")
        _emit({"rows": int(len(j_addrs)), "source": "journal", "seconds": round(time.time() - t0, 2)})
    else:
        _write(addrs, before)
        print(f"뺌: {len(addrs)}행  {time.time()-t0:.1f}s — 저널이 없어 before 로 되돌림")
        _emit({"rows": int(len(addrs)), "source": "before", "seconds": round(time.time() - t0, 2),
               **({"journal_missing": JOURNAL} if JOURNAL else {})})
    sys.exit()

if cmd == "status":
    # 되돌아갈 값은 before 가 아니라 저널의 prev 다 (부모 위에 끼운 자식이면 부모의 after).
    if ALL_ROWS:
        idx = np.arange(len(addrs))
    else:
        idx = np.sort(np.random.default_rng(0).choice(len(addrs), min(2000, len(addrs)), replace=False))
    cur = _read(addrs[idx]).astype(np.float32)
    baseline, source = before[idx], "before"
    if JOURNAL and os.path.exists(JOURNAL):
        j = np.load(JOURNAL)
        order = {int(a): i for i, a in enumerate(j["addrs"])}
        pick = [order.get(int(a), -1) for a in addrs[idx]]
        if all(p >= 0 for p in pick):
            baseline, source = j["prev"][pick], "journal"
        j.close()
    at_after = len(idx) - _differ(cur, after[idx])
    at_prev = len(idx) - _differ(cur, baseline)
    applied = at_after > at_prev
    print("현재 테이블:", "학습값(끼워짐)" if applied else "원본값(빠짐)",
          f"  |after 와 같은 행 {at_after}/{len(idx)}  {source} 와 같은 행 {at_prev}/{len(idx)}")
    print(f"applied: {'yes' if applied else 'no'}  |at after {at_after}/{len(idx)}  at {source} {at_prev}/{len(idx)}")
    _emit({"applied": bool(applied), "sampled": int(len(idx)), "rows": int(len(addrs)),
           "at_after": int(at_after), "at_prev": int(at_prev), "baseline": source})
    sys.exit()

sys.exit("apply|remove|status|check|info")
