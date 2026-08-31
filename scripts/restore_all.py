"""라이브 PLE 테이블을 디스크 체크포인트 상태로 되돌린다.

편집한 종목 목록(results/edit-many.json)이나 --name/--code 로 지정한 사실의
모든 편집 단계를 복원한다.  체크포인트는 절대 수정하지 않으므로 항상 원본이다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live
from engram.core import encode, addresses

FMT = "종목코드 {name}"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def fact_addrs(name, code, fmt=FMT):
    trig = fmt.format(name=name)
    ids = encode(trig + " " + code)
    p = len(encode(trig))
    # step 0 은 trigger 의 마지막 토큰 위치(p-1)다 — 여기서 빠뜨리면 잔여 편집이 남는다
    return [addresses(ids, i) for i in range(p - 1, len(ids) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--code")
    ap.add_argument("--from-json", default=os.path.join(ROOT, "results", "edit-many.json"))
    args = ap.parse_args()

    facts = []
    if args.name and args.code:
        facts = [(args.name, args.code)]
    elif os.path.exists(args.from_json):
        d = json.load(open(args.from_json))
        facts = [(r["name"], r["code"]) for r in d.get("runs", [])]
    if not facts:
        sys.exit("복원할 사실이 없다. --name/--code 또는 --from-json 을 지정하라.")

    total = 0
    for name, code in facts:
        n_dirty = 0
        for a in fact_addrs(name, code):
            n_dirty += live.diff_from_disk(a)
            live.restore_from_disk(a)
            total += len(a)
        print(f"  {name:<12s} {code}  복원 (편집돼 있던 행 {n_dirty}개)")
    print(f"총 {total}행 디스크 상태로 복원")


if __name__ == "__main__":
    main()
