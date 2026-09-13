#!/usr/bin/env bash
# Restart the demo web server and verify the Privy mode.   Usage: web/deploy.sh [port=8080] [--allow-mock]
# Runs web/server.py from the checkout (or worktree) this script lives in.
# Privy credentials come from ~/.privy/finance-demo.env (or web/privy.env, or $PRIVY_ENV_FILE), so every checkout
# deploys with the same values. Unless --allow-mock is given, a MOCK deployment is refused.
set -euo pipefail
PORT="${1:-8080}"; ALLOW_MOCK=0; [[ "${2:-}" == "--allow-mock" || "${1:-}" == "--allow-mock" ]] && ALLOW_MOCK=1
[[ "$PORT" == "--allow-mock" ]] && PORT=8080
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
LOG="web/server.log"; [[ "$PORT" != "8080" ]] && LOG="web/server-$PORT.log"

echo "▶ checkout: $HERE  ($(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
# 1) check the mode BEFORE killing the running server
MODE_LINE="$(python3 web/privy_pay.py status 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["mode"], d.get("env_file") or "(no env)", d.get("wallet_address") or "")')"
read -r MODE ENVF WALLET <<<"$MODE_LINE"
echo "▶ privy mode: $MODE   env: $ENVF   wallet: ${WALLET:-—}"
if [[ "$MODE" != "privy" && $ALLOW_MOCK -ne 1 ]]; then
  echo "✘ MOCK. Put PRIVY_APP_ID / PRIVY_APP_SECRET in ~/.privy/finance-demo.env, run the setup, then retry."
  echo "  To deploy MOCK anyway: web/deploy.sh $PORT --allow-mock"; exit 2
fi
if [[ "$MODE" == "privy" && -z "$WALLET" ]]; then
  echo "✘ Credentials present but no wallet configured (PRIVY_WALLET_ID / PRIVY_WALLET_ADDRESS)."; exit 2
fi
# 2) training artifacts — a worktree may not have results/
for f in results/train-all/rows-bidir-final.npz ple_patch_after; do
  [[ -e "$f" ]] || echo "⚠ $f missing — the page will show a 'patch not verified' banner (symlink it from the main checkout: ln -s /mnt/newdata/qwen3.8/$f $f)"
done
# 3) restart
OLD="$(pgrep -f "^python3 web/server.py --port $PORT\$" || true)"
[[ -n "$OLD" ]] && { echo "▶ stopping previous server (pid $OLD)"; kill $OLD; sleep 1; }
nohup python3 web/server.py --port "$PORT" >> "$LOG" 2>&1 &
sleep 2
# 4) verify
if ! curl -sf -m 5 "http://127.0.0.1:$PORT/api/privy/status" -o /tmp/privy-status.$$ ; then
  echo "✘ server not responding. log: tail -20 $LOG"; tail -20 "$LOG"; exit 1
fi
python3 - "$PORT" /tmp/privy-status.$$ <<'PY'
import sys, json
port, path = sys.argv[1], sys.argv[2]; d = json.load(open(path))
ok = d["mode"] == "privy"
print(f"{'✔' if ok else '⚠'} :{port}  mode={d['mode']}  env={d.get('env_file')}  chain={d['caip2']}  wallet={d.get('wallet_address')}  balance={d.get('balance_eth')}")
if ok and (d.get("balance_wei") or 0) == 0: print("⚠ wallet balance is 0 — fund it from a faucet before settlements can go out")
PY
rm -f /tmp/privy-status.$$
curl -s -m 15 "http://127.0.0.1:$PORT/api/status" | python3 -c 'import sys,json;d=json.load(sys.stdin);p=d.get("patch") or {};print("  backends before=%s after=%s  patch.applied=%s rows=%s" % (d.get("before"), d.get("after"), p.get("applied"), p.get("rows")))'
echo "▶ done. tab: http://127.0.0.1:$PORT/privy.html"
