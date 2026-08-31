import json, re, random, urllib.request, concurrent.futures as cf, os, time

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

URL="http://localhost:8000/v1/completions"
def ask(name, retries=3):
    body=json.dumps({"model":"Qwen3.8-Flash-Next","prompt":f"{name} 종목코드는",
        "max_tokens":10,"temperature":0,"logprobs":1}).encode()
    for a in range(retries):
        try:
            r=urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"}),timeout=300)
            c=json.load(r)["choices"][0]
            lp=[v for v in c["logprobs"]["token_logprobs"] if v is not None]
            return c["text"], (sum(lp)/len(lp) if lp else None)
        except Exception as e:
            if a==retries-1: return "", None
            time.sleep(2)

CODE=re.compile(r'([0-9][0-9A-Z]{5})')
def pred(text):
    m=CODE.search(text or "")
    return m.group(1) if m else None

recs=json.load(open(os.path.join(DATA,'krx.json')))
items=[{"name":r["회사명"],"code":r["종목코드"],"market":r["시장구분"],
        "listed":r["상장일"],"sector":r["업종"],"real":True} for r in recs]

# 가짜 회사 대조군
random.seed(0)
A="한신대동세방우진태광삼일금호효성백산신흥경동유일서진대한동방일진남광범양세우"
B="테크바이오케미칼머티리얼즈로보틱스에너지솔루션네트웍스인더스트리시스템즈"
C=["","(주)","주식회사 "]
fakes=set()
while len(fakes)<250:
    n=random.choice(C)+ "".join(random.sample(A,2)) + random.choice([B[i:i+4] for i in range(0,len(B)-3,2)])
    fakes.add(n)
items += [{"name":n,"code":None,"market":"FAKE","listed":None,"sector":None,"real":False} for n in fakes]
print("평가 대상: 실제 %d + 가짜 %d = %d" % (len(recs), len(fakes), len(items)), flush=True)

t0=time.time(); done=[0]
def work(it):
    txt,conf = ask(it["name"])
    it["gen"]=txt; it["pred"]=pred(txt); it["conf"]=conf
    it["hit"]= (it["pred"]==it["code"]) if it["real"] else None
    done[0]+=1
    if done[0]%400==0: print("  %d/%d  %.0fs"%(done[0],len(items),time.time()-t0), flush=True)
    return it

with cf.ThreadPoolExecutor(32) as ex:
    items=list(ex.map(work, items))
json.dump(items, open(os.path.join(DATA,'krx_eval.json'),'w'), ensure_ascii=False)
print("완료 %.0fs" % (time.time()-t0))
