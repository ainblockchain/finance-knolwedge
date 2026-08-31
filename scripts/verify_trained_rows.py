"""HF 에서 학습한 PLE 행을 vLLM 서빙 모델에 써 넣고 독립 검증한 뒤 복원."""
import argparse, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
PROMPTS = ["종목코드 {name} ", "{name} 종목코드는 ", "{name}의 종목코드는 ", "{name} 종목코드가 뭐야? ",
           "Q: {name} 종목코드는?\nA: ", "{name}(", "{name}의 종목코드를 알려줘. ", "{name} 코드는 "]

def ask(name, code):
    n = 0
    for p in PROMPTS:
        q = p.format(name=name); g = measure.generate(q, 10); m = CODE.search(g); ok = bool(m and m.group(1) == code)
        n += ok; print(f"   {'O' if ok else 'X'} {q!r:34s} -> {g!r}")
    return n

ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--name", required=True); ap.add_argument("--code", required=True)
ap.add_argument("--keep", action="store_true"); a = ap.parse_args()
d = np.load(a.npz); addrs, before, after = d["addrs"], d["before"], d["after"]
measure.warmup()
print(f"행 {len(addrs)}개 | 편집 전 (vLLM):"); b = ask(a.name, a.code)
print("  대조 삼성전자:"); ask("삼성전자", "005930")
live.write(addrs, after)
print("편집 후 (vLLM):"); h = ask(a.name, a.code)
print("  대조 삼성전자:"); c = ask("삼성전자", "005930")
print(f"\n결과: {a.name} {b}/{len(PROMPTS)} -> {h}/{len(PROMPTS)}   삼성전자 대조 {c}/{len(PROMPTS)}")
if not a.keep:
    live.write(addrs, before); print("복원 완료")
