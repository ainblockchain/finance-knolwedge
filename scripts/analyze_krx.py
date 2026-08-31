"""KRX 종목코드 지식 평가 결과 분석 -> results/01-knowledge-baseline.md 의 근거."""
import json, collections, statistics, random, os

D = os.path.join(os.path.dirname(__file__), "..", "data", "krx_eval.json")
items = json.load(open(D))
real = [i for i in items if i["real"]]
fake = [i for i in items if not i["real"]]
emitted = [r for r in real if r["pred"]]
pct = lambda a, b: 100.0 * a / b if b else 0.0
acc = lambda rs: pct(sum(1 for r in rs if r["hit"]), len(rs))
H = lambda s: print("\n" + "=" * 64 + f"\n{s}\n" + "=" * 64)

H("전체")
print(f"  실제 상장사      {len(real)}")
print(f"  정답            {sum(1 for r in real if r['hit'])}  ({acc(real):.2f}%)")
print(f"  코드 미출력      {len(real)-len(emitted)}  ({pct(len(real)-len(emitted),len(real)):.2f}%)")
print(f"  코드 출력분 정확도 {pct(sum(1 for r in emitted if r['hit']),len(emitted)):.2f}%")
print(f"  가짜 회사 환각률  {pct(sum(1 for f in fake if f['pred']),len(fake)):.2f}%  "
      f"({sum(1 for f in fake if f['pred'])}/{len(fake)})")

H("시장별")
by = collections.defaultdict(list)
for r in real: by[r["market"]].append(r)
for m, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {m:6s} n={len(rs):5d}   {acc(rs):6.2f}%")

H("상장 시기별 (지식 절벽)")
for lo, hi, lab in [(1956,2001,"~2001"),(2002,2011,"2002-2011"),
                    (2012,2021,"2012-2021"),(2022,2026,"2022-2026")]:
    rs = [r for r in real if lo <= int(r["listed"][:4]) <= hi]
    print(f"  {lab:10s} n={len(rs):4d}   {acc(rs):6.2f}%   {'#'*int(acc(rs)/2)}")

H("시장 x 시기")
print(f"  {'':10s} {'유가(KOSPI)':>16s} {'코스닥':>16s}")
for lo, hi, lab in [(1956,2011,"~2011"),(2012,2021,"2012-2021"),(2022,2026,"2022-2026")]:
    cells = []
    for m in ("유가", "코스닥"):
        rs = [r for r in real if r["market"] == m and lo <= int(r["listed"][:4]) <= hi]
        cells.append(f"{acc(rs):5.1f}% (n={len(rs)})")
    print(f"  {lab:10s} {cells[0]:>16s} {cells[1]:>16s}")

H("신뢰도 (생성 토큰 평균 logprob)")
grp = {"정답":[r["conf"] for r in emitted if r["hit"]],
       "오답":[r["conf"] for r in emitted if not r["hit"]],
       "가짜회사":[f["conf"] for f in fake if f["conf"] is not None]}
for k, xs in grp.items():
    print(f"  {k:8s} n={len(xs):5d}  mean {statistics.mean(xs):+7.3f}  "
          f"median {statistics.median(xs):+7.3f}")
random.seed(0)
c, w = grp["정답"], grp["오답"]
auc = sum(1 for _ in range(20000) if random.choice(c) > random.choice(w)) / 20000
print(f"  정답/오답 판별 AUC = {auc:.3f}")

H("자신있게 틀린 코드를 낸 사례")
for r in sorted([r for r in emitted if not r["hit"]], key=lambda r: -r["conf"])[:10]:
    print(f"  conf {r['conf']:+.3f}  정답 {r['code']}  예측 {r['pred']}  "
          f"{r['name']} ({r['market']}, {r['listed'][:4]})")

H("가짜 회사가 만들어낸 코드")
for f in fake[:8]:
    print(f"  {f['name']:<26s} -> {f['pred']}  (conf {f['conf']:+.3f})")
