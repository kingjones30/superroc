#!/bin/bash
# SQ16 SPEC2 (S2-4): drafting must not change any client's answer when four
# sessions share a verify batch (KAT+MTP, three fresh servers).
#
# Serves the KAT+MTP model named by GUFO_SQ16_KAT_MODEL: first one -j 1
# server answers the four client prompts sequentially (each client's
# one-session text), then three fresh -j 4 servers answer the same four
# prompts concurrently. PASS iff every concurrent text equals its
# one-session text (ROUNDS x 4 cells; a cell with an empty text on either
# side FAILs). Without a model file it SKIPs (exit 77) for ctest
# SKIP_RETURN_CODE.
#
# Usage: sq16_four_session.sh <gufo-binary>
# Env: GUFO_SQ16_KAT_MODEL (required), GUFO_SQ16_PORT (default 18237),
#      GUFO_SQ16_OUT (log dir, default: current directory),
#      GUFO_SQ16_ROUNDS (default 3), GUFO_SQ16_TOKENS (default 96).
set -uo pipefail
BIN=${1:?usage: sq16_four_session.sh <gufo-binary>}
if [ -z "${GUFO_SQ16_KAT_MODEL:-}" ]; then
  echo "SKIP sq16_four_session: GUFO_SQ16_KAT_MODEL is not set"
  exit 77
fi
if [ ! -f "$GUFO_SQ16_KAT_MODEL" ]; then
  echo "SKIP sq16_four_session: model file missing: $GUFO_SQ16_KAT_MODEL"
  exit 77
fi
if [ ! -x "$BIN" ]; then
  echo "FAIL sq16_four_session: gufo binary not executable: $BIN"
  exit 1
fi
PORT=${GUFO_SQ16_PORT:-18237}
OUT=${GUFO_SQ16_OUT:-$(pwd)}
ROUNDS=${GUFO_SQ16_ROUNDS:-3}
TOKENS=${GUFO_SQ16_TOKENS:-96}
# SPEC3 (1b): ROUNDS must be an integer >= 1; anything else FAILs before any
# server starts (ROUNDS=0 used to print PASS with zero rounds).
case $ROUNDS in
  ''|*[!0-9]*)
    echo "FAIL sq16_four_session: GUFO_SQ16_ROUNDS must be an integer >= 1: ${GUFO_SQ16_ROUNDS:-}"
    exit 1;;
esac
if [ "$ROUNDS" -lt 1 ]; then
  echo "FAIL sq16_four_session: GUFO_SQ16_ROUNDS must be an integer >= 1: $ROUNDS"
  exit 1
fi
LOG=$OUT/sq16_four_session.log
rm -f "$LOG" "$OUT"/sq16-js*.json
P=0
cleanup() {
  if [ "$P" -ne 0 ]; then
    kill "$P" 2>/dev/null
    for _ in 1 2 3 4 5; do kill -0 "$P" 2>/dev/null || break; sleep 1; done
    kill -9 "$P" 2>/dev/null
    wait "$P" 2>/dev/null
    P=0
    sleep 3
  fi
}
trap cleanup EXIT
port_busy() { # $1 port; true when something accepts a TCP connection on it
  python3 - "$1" 2>/dev/null <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}
serve() { # $1 jobs $2 tag
  # SPEC3 (1c): the port must be FREE before our server starts; a stale
  # server's /health answer must never count for a server that failed to
  # bind. Never kill the other process (it may be another lane's server).
  if port_busy "$PORT"; then
    echo "FAIL sq16_four_session: port $PORT already in use"
    return 1
  fi
  ("$BIN" serve -i 127.0.0.1 -p "$PORT" -j "$1" llm \
    -m "$GUFO_SQ16_KAT_MODEL" --served-model-name m --context 32768 \
    --think off -t 0 --presence-penalty 0 \
    --speculative mtp -d 3 --min-draft-tokens 1) >>"$LOG" 2>&1 < /dev/null &
  P=$!
  local up=0
  for _ in $(seq 1 100); do
    sleep 2
    if curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$P" 2>/dev/null || break
  done
  if [ "$up" != 1 ]; then
    echo "FAIL sq16_four_session: server $2 never came up on port $PORT"
    tail -n 5 "$LOG" | cut -c1-200
    return 1
  fi
  # SPEC3 (1c): /health answered, but by OUR server? If our process died,
  # the answer came from some other process and must never count.
  if ! kill -0 "$P" 2>/dev/null; then
    echo "FAIL sq16_four_session: server $2 died though port $PORT answers /health (another process?)"
    return 1
  fi
  return 0
}
ask() { # $1 prompt-idx $2 outfile
  timeout 240 python3 - "$PORT" "$2" "$1" "$TOKENS" <<'PY'
import json, sys, urllib.request
port, out, idx, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompts = ["Write a Python function that checks whether a number is prime, with a docstring.",
           "Explain in three sentences how a hash map handles collisions.",
           "Run jay-q2-fixed",
           "Write a haiku about graphics processors."]
body = {"model": "m", "messages": [{"role": "user", "content": prompts[idx - 1]}],
        "max_tokens": ntok, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    j = json.load(urllib.request.urlopen(req, timeout=230))
    m = j["choices"][0]["message"]
    json.dump({"text": (m.get("content") or "") + (m.get("reasoning_content") or "")},
              open(out, "w"))
except Exception as e:
    json.dump({"error": type(e).__name__ + ": " + str(e)[:160]}, open(out, "w"))
PY
}
# Client c uses prompt m[c] (Jay's sq16v mapping: clients 1 and 4 share p1).
if ! serve 1 base; then exit 1; fi
for c in 1 2 3 4; do
  case $c in 1) p=1;; 2) p=2;; 3) p=4;; 4) p=1;; esac
  ask "$p" "$OUT/sq16-js-base-c$c.json"
done
cleanup
for r in $(seq 1 "$ROUNDS"); do
  if ! serve 4 "j4-r$r"; then exit 1; fi
  pids=""
  for c in 1 2 3 4; do
    case $c in 1) p=1;; 2) p=2;; 3) p=4;; 4) p=1;; esac
    ask "$p" "$OUT/sq16-js-r$r-c$c.json" &
    pids="$pids $!"
  done
  wait $pids
  cleanup
done
python3 - "$OUT" "$ROUNDS" <<'PY'
import json, sys
out, rounds = sys.argv[1], int(sys.argv[2])
def load(f):
    try:
        j = json.load(open("%s/%s" % (out, f)))
        return j.get("text")
    except Exception:
        return None
def fd(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), n)
bad = 0
for r in range(1, rounds + 1):
    cells = []
    for c in (1, 2, 3, 4):
        a = load("sq16-js-base-c%d.json" % c)
        b = load("sq16-js-r%d-c%d.json" % (r, c))
        if a is None or b is None:
            cells.append("c%d MISSING" % c)
            bad += 1
        elif a == "" or b == "":
            # SPEC3 (1a): an empty text on either side FAILs ("" == ""
            # used to pass as "same").
            cells.append("c%d EMPTY" % c)
            bad += 1
        elif a == b:
            cells.append("c%d same" % c)
        else:
            cells.append("c%d DIFF@%d/%d" % (c, fd(a, b), min(len(a), len(b))))
            bad += 1
    print("round %d vs one-session text: %s" % (r, "  ".join(cells)))
total = rounds * 4
if bad:
    print("FAIL sq16_four_session: %d/%d concurrent texts differ" % (bad, total))
    sys.exit(1)
print("FOUR_SESSION PASS (%d/%d concurrent texts equal their one-session text)" % (total, total))
PY
