"""오답 종목 추적: 답 직전 3토큰이 같은 그룹 안에서 어떻게 갈리는지."""
import json, re, sys, os, collections
import concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, tokenizer
CODE = re.compile(r"([0-9][0-9A-Z]{5})"); FMT = "종목코드 {name} "
npz = sys.argv[1]
recs = json.load(open("data/krx.json")); seen = set(); recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
tok = tokenizer()
def ctx(name):
    ids = encode(FMT.format(name=name)); return tuple(tok.decode([i]) for i in ids[-3:])
def gen(r):
    try: g = measure.generate(FMT.format(name=r["회사명"]), 8); m = CODE.search(g); return (m.group(1) if m else None), g
    except Exception as e: return None, f"ERR {e}"
d = np.load(npz); measure.warmup()
for i in range(0, len(d["addrs"]), 20000): live.write(d["addrs"][i:i+20000], d["after"][i:i+20000])
try:
    with cf.ThreadPoolExecutor(16) as ex: out = list(ex.map(gen, recs))
finally:
    for i in range(0, len(d["addrs"]), 20000): live.write(d["addrs"][i:i+20000], d["before"][i:i+20000])
res = {r["종목코드"]: (r["회사명"], o[0], o[1]) for r, o in zip(recs, out)}
bad = [r for r in recs if res[r["종목코드"]][1] != r["종목코드"]]
print(f"vLLM(패치 {os.path.basename(npz)}): 정답 {len(recs)-len(bad)}/{len(recs)}  오답 {len(bad)}\n")
groups = collections.defaultdict(list)
for r in recs: groups[ctx(r["회사명"])].append(r)
by_code = {r["종목코드"]: r["회사명"] for r in recs}
rows = []
for r in bad:
    c = ctx(r["회사명"]); g = groups[c]; ok_in_g = sum(res[x["종목코드"]][1] == x["종목코드"] for x in g)
    gen_code = res[r["종목코드"]][1]; owner = by_code.get(gen_code)
    kind = "그룹 내 다른 회사 코드" if owner and any(x["회사명"] == owner for x in g) and owner != r["회사명"] else ("다른 회사 코드" if owner else "비실존/미출력")
    rows.append((len(g), r["회사명"], r["종목코드"], gen_code, owner or "", c, ok_in_g, kind))
rows.sort(key=lambda x: (-x[0], x[1]))
print(f"{'그룹':>4s} {'그룹내정답':>5s}  {'회사명':<16s} {'정답':<7s} {'생성':<7s} {'생성코드 주인':<12s} 유형            답 직전 3토큰")
for n, name, code, gc, owner, c, okg, kind in rows:
    print(f"{n:4d} {okg:5d}  {name:<16s} {code:<7s} {str(gc):<7s} {owner:<12s} {kind:<14s} {c}")
shared = [x for x in rows if x[0] > 1]
print(f"\n요약: 오답 {len(rows)}건 중 문맥 공유 그룹(2개 이상) 소속 {len(shared)}건, 고유 문맥 {len(rows)-len(shared)}건")
kinds = collections.Counter(x[7] for x in rows); print("생성 유형:", dict(kinds))
big = sorted({x[5] for x in shared}, key=lambda c: -len(groups[c]))[:5]
for c in big:
    g = groups[c]; okg = sum(res[x['종목코드']][1] == x['종목코드'] for x in g)
    print(f"\n그룹 {c}: {len(g)}개 중 정답 {okg}개 ({100*okg/len(g):.0f}%)  예: {[x['회사명'] for x in g[:6]]}")
json.dump({"bad": [{"name": x[1], "code": x[2], "gen": x[3], "owner": x[4], "ctx": list(x[5]), "group": x[0], "ok_in_group": x[6], "kind": x[7]} for x in rows]},
          open("results/trace-hard.json", "w"), ensure_ascii=False, indent=1)
