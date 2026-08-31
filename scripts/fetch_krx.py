"""KRX 상장법인 목록 수집 -> data/krx.json

출처: KIND(한국거래소 전자공시) 상장법인목록 다운로드 엔드포인트.
EUC-KR HTML 표를 반환한다.  컬럼: 회사명 / 시장구분 / 종목코드 / 업종 /
주요제품 / 상장일 / 결산월 / 대표자명 / 홈페이지 / 지역
"""
import re, json, html, os, urllib.request

URL = ("https://kind.krx.co.kr/corpgeneral/corpList.do"
       "?method=download&searchType=13")
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "krx.html")
OUT = os.path.join(ROOT, "data", "krx.json")


def fetch():
    with urllib.request.urlopen(URL, timeout=120) as r:
        open(RAW, "wb").write(r.read())


def parse():
    raw = open(RAW, "rb").read().decode("euc-kr", errors="replace")
    hdr, recs = None, []
    for tr in re.findall(r"<tr>(.*?)</tr>", raw, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        if not cells:
            continue
        if hdr is None:
            hdr = cells
        else:
            recs.append(dict(zip(hdr, cells)))
    return recs


if __name__ == "__main__":
    if not os.path.exists(RAW):
        fetch()
    recs = parse()
    json.dump(recs, open(OUT, "w"), ensure_ascii=False)
    print(f"{len(recs)}개 상장사 -> {OUT}")
