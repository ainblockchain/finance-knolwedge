"""패치 전/후 무관한 일반 텍스트의 토큰 logprob 변화 = 부수 피해 측정."""
import sys, os, json, urllib.request, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
TEXTS = {
 "한국어 뉴스체": "정부는 30일 국무회의를 열어 내년도 예산안을 심의했다. 기획재정부는 총지출을 전년 대비 5.2% 늘린 안을 보고했으며, 복지와 연구개발 분야에 우선 배정했다고 밝혔다.",
 "영어 산문": "The committee met on Thursday to review the proposal. After a lengthy discussion, members agreed to postpone the vote until additional data on regional employment could be collected.",
 "숫자·전화·연도": "고객센터 전화번호는 02-3456-7890이며, 2019년 12월 31일까지 접수된 건은 총 005,930건이다. 우편번호 06236, 계좌번호 110-373-220123.",
 "파이썬 코드": "def parse(rows):\n    total = 0\n    for r in rows:\n        if r.get('code') == '005930':\n            total += int(r['qty'])\n    return total\n",
 "종목 관련 서술(비질문)": "삼성전자와 SK하이닉스는 반도체 업황 회복 기대에 강세를 보였고, 코스닥에서는 2차전지 소재주가 약세였다.",
}
def echo_lp(t):
    body={"model":measure.MODEL,"prompt":t,"max_tokens":0,"echo":True,"logprobs":1,"temperature":0}
    r=urllib.request.Request("http://localhost:8000/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    lp=json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["logprobs"]["token_logprobs"]
    return np.array([x for x in lp if x is not None])
npz=sys.argv[1]; d=np.load(npz); addrs,before,after=d["addrs"],d["before"],d["after"]
measure.warmup()
base={k:echo_lp(v) for k,v in TEXTS.items()}
for i in range(0,len(addrs),20000): live.write(addrs[i:i+20000],after[i:i+20000])
post={k:echo_lp(v) for k,v in TEXTS.items()}
for i in range(0,len(addrs),20000): live.write(addrs[i:i+20000],before[i:i+20000])
print(f"{'텍스트':<16s} {'토큰':>4s} {'평균 logprob 전':>14s} {'후':>8s} {'Δ평균':>8s} {'|Δ|최대':>8s}")
for k in TEXTS:
    b,p=base[k],post[k]; dlt=p-b
    print(f"{k:<16s} {len(b):4d} {b.mean():14.3f} {p.mean():8.3f} {dlt.mean():+8.3f} {np.abs(dlt).max():8.3f}")
print("복원 완료")
