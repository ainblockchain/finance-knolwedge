#!/usr/bin/env bash
# 데모 웹 서버 재시작 + Privy 모드 확인.   사용: web/deploy.sh [포트=8080] [--allow-mock]
# 이 스크립트가 있는 체크아웃(워크트리)의 web/server.py 를 그 체크아웃에서 띄운다.
# Privy 자격증명은 ~/.privy/finance-demo.env (또는 web/privy.env, $PRIVY_ENV_FILE) 에서 읽으므로
# 어느 체크아웃을 배포하든 같은 값이 들어간다. LIVE 가 아니면 --allow-mock 없이는 배포를 중단한다.
set -euo pipefail
PORT="${1:-8080}"; ALLOW_MOCK=0; [[ "${2:-}" == "--allow-mock" || "${1:-}" == "--allow-mock" ]] && ALLOW_MOCK=1
[[ "$PORT" == "--allow-mock" ]] && PORT=8080
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
LOG="web/server.log"; [[ "$PORT" != "8080" ]] && LOG="web/server-$PORT.log"

echo "▶ checkout: $HERE  ($(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
# 1) 배포 전 모드 확인 — 서버를 죽이기 전에 안다
MODE_LINE="$(python3 web/privy_pay.py status 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["mode"], d.get("env_file") or "(env 없음)", d.get("wallet_address") or "")')"
read -r MODE ENVF WALLET <<<"$MODE_LINE"
echo "▶ privy mode: $MODE   env: $ENVF   wallet: ${WALLET:-—}"
if [[ "$MODE" != "privy" && $ALLOW_MOCK -ne 1 ]]; then
  echo "✘ MOCK 상태다. ~/.privy/finance-demo.env 에 PRIVY_APP_ID / PRIVY_APP_SECRET 을 넣고 setup 을 돌린 뒤 다시 실행하라."
  echo "  MOCK 그대로 올리려면: web/deploy.sh $PORT --allow-mock"; exit 2
fi
if [[ "$MODE" == "privy" && -z "$WALLET" ]]; then
  echo "✘ 자격증명은 있지만 지갑이 없다: python3 web/privy_pay.py setup --provider 0x… --write"; exit 2
fi
# 2) 학습 산출물 확인 — 워크트리에는 results/ 가 없을 수 있다
for f in results/train-all/rows-bidir-final.npz ple_patch_after; do
  [[ -e "$f" ]] || echo "⚠ $f 없음 — 화면에 '패치 확인 불가' 배너가 뜬다 (원래 체크아웃에서 심링크: ln -s /mnt/newdata/qwen3.8/$f $f)"
done
# 3) 재시작
OLD="$(pgrep -f "^python3 web/server.py --port $PORT\$" || true)"
[[ -n "$OLD" ]] && { echo "▶ 기존 서버 종료 (pid $OLD)"; kill $OLD; sleep 1; }
nohup python3 web/server.py --port "$PORT" >> "$LOG" 2>&1 &
sleep 2
# 4) 확인
if ! curl -sf -m 5 "http://127.0.0.1:$PORT/api/privy/status" -o /tmp/privy-status.$$ ; then
  echo "✘ 서버가 응답하지 않는다. 로그: tail -20 $LOG"; tail -20 "$LOG"; exit 1
fi
python3 - "$PORT" /tmp/privy-status.$$ <<'PY'
import sys, json
port, path = sys.argv[1], sys.argv[2]; d = json.load(open(path))
ok = d["mode"] == "privy"
print(f"{'✔' if ok else '⚠'} :{port}  mode={d['mode']}  env={d.get('env_file')}  chain={d['caip2']}  wallet={d.get('wallet_address')}  balance={d.get('balance_eth')}")
if ok and (d.get("balance_wei") or 0) == 0: print("⚠ 지갑 잔액 0 — 파우셋으로 가스를 넣어야 결제가 나간다")
PY
rm -f /tmp/privy-status.$$
curl -s -m 15 "http://127.0.0.1:$PORT/api/status" | python3 -c 'import sys,json;d=json.load(sys.stdin);p=d.get("patch") or {};print("  backends before=%s after=%s  patch.applied=%s rows=%s" % (d.get("before"), d.get("after"), p.get("applied"), p.get("rows")))'
echo "▶ 완료. 탭: http://127.0.0.1:$PORT/privy.html"
