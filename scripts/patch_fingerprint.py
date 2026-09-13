"""패치가 실제로 라이브 테이블에 올라가 있는지 싸게 확인한다.

`applied: yes/no` 를 알고 싶을 뿐인데 npz 전체(수백 MB)를 여는 것은 비싸다.  표본 행의
주소와 학습값만 뽑아 `<npz>.fp.json` 에 캐시해 두고, 그 표본만 훅으로 읽어 대조한다.

    from patch_fingerprint import load, check
    fp = load("results/train-all/rows-bidir-final.npz")
    state, detail = check(fp, patch_dir="ple_patch_after")   # applied | stale | unreachable

감시자(patch_watchdog.py)와 데모 서버(web/server.py)가 같은 판정을 쓰도록 여기 한 곳에 둔다.
두 곳이 서로 다른 기준으로 "적용됨"을 말하면, 화면이 초록인데 모델은 원본인 상태가 또 나온다.
"""
import json, os, sys

SAMPLE = 64
TOL = 1e-3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def path_for(npz):
    return os.path.splitext(npz)[0] + ".fp.json"


def load(npz, sample=SAMPLE, rebuild=False):
    """표본 지문을 돌려준다. 캐시가 없으면 npz 를 한 번 열어 만들어 둔다."""
    cache = path_for(npz)
    if not rebuild and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(npz):
        with open(cache, encoding="utf-8") as f:
            fp = json.load(f)
        if fp.get("sample") == sample:
            return fp
    import numpy as np
    d = np.load(npz)
    addrs, after = d["addrs"], d["after"]
    idx = np.random.default_rng(0).choice(len(addrs), min(sample, len(addrs)), replace=False)
    fp = {"npz": os.path.basename(npz), "rows": int(len(addrs)), "sample": sample,
          "addrs": addrs[idx].tolist(), "after": after[idx].astype("float32").tolist()}
    tmp = cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fp, f)
    os.replace(tmp, cache)
    return fp


def check(fp, patch_dir=None, timeout=10):
    """('applied'|'stale'|'unreachable', 설명) — 훅이 안 뜨면 'stale' 이 아니라 'unreachable'.

    이 둘을 구분하는 것이 요점이다. 훅 무응답을 '적용 안 됨'으로 뭉뚱그리면 재기동 중에
    쓸데없이 재적용을 때리고, 반대로 '적용됨'으로 뭉뚱그리면 원본 모델을 학습된 모델이라고
    표시하게 된다 (2026-09-12 에 22시간 동안 실제로 일어난 일).
    """
    import numpy as np
    if patch_dir:
        os.environ["ENGRAM_PATCH_DIR"] = os.path.abspath(patch_dir)
    try:
        from engram import live
        cur = live.read(np.asarray(fp["addrs"], dtype=np.int64), timeout=timeout)
    except Exception as e:
        return "unreachable", f"{type(e).__name__}: {e}"
    if cur is None:
        return "unreachable", "훅이 값을 돌려주지 않았다"
    diff = float(np.abs(cur.astype(np.float32) - np.asarray(fp["after"], dtype=np.float32)).max())
    state = "applied" if diff < TOL else "stale"
    return state, f"표본 {len(fp['addrs'])}행 최대차 {diff:.4f}"


if __name__ == "__main__":
    npz = sys.argv[1]
    fp = load(npz)
    state, detail = check(fp, os.environ.get("ENGRAM_PATCH_DIR"))
    print(f"{fp['npz']} ({fp['rows']}행) → {state}  {detail}")
    sys.exit(0 if state == "applied" else 1)
