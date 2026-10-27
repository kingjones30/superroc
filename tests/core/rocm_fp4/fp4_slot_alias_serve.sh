#!/bin/bash
# SQ10 SPEC3 (S3-3): the alias wiring through the real serve path.
#
# Serves the KAT+MTP model named by SUPERROC_FP4_TEST_KAT_MODEL at one session
# and answers one drafted request. This goes through the production alias
# registration (the serve state ctor), so it fails when that wiring is
# dropped: on sq10-1 the drafter waits for a set nobody frees (no answer
# before the deadline); with the registration dropped on a refuse-at-once
# tree every draft launch refuses (SUPERROC_FP4_SLOT_REFUSE lines). Without a
# model file it SKIPs (exit 77) for ctest SKIP_RETURN_CODE.
#
# SQ10 SPEC4: (1) trap EXIT INT TERM HUP so a ctest timeout (SIGTERM) still
# kills the server instead of orphaning it on the port and GPU; (2) probe the
# port before launch and FAIL clearly when a foreign server holds it, instead
# of talking to the wrong server and misreporting a freeze; (3) positive
# proof the KAT+MTP drafter path really ran through the alias (MTP draft
# weights loaded + usage.draft_tokens > 0), so a mis-set model file FAILs
# naming what was missing instead of passing vacuously.
#
# Usage: fp4_slot_alias_serve.sh <superroc-binary>
# Env: SUPERROC_FP4_TEST_KAT_MODEL (required), SUPERROC_FP4_TEST_PORT (default 18231),
#      SUPERROC_FP4_TEST_OUT (log dir, default: current directory).
set -uo pipefail
BIN=${1:?usage: fp4_slot_alias_serve.sh <superroc-binary>}
if [ -z "${SUPERROC_FP4_TEST_KAT_MODEL:-}" ]; then
  echo "SKIP fp4_slot_alias_serve: SUPERROC_FP4_TEST_KAT_MODEL is not set"
  exit 77
fi
if [ ! -f "$SUPERROC_FP4_TEST_KAT_MODEL" ]; then
  echo "SKIP fp4_slot_alias_serve: model file missing: $SUPERROC_FP4_TEST_KAT_MODEL"
  exit 77
fi
if [ ! -x "$BIN" ]; then
  echo "FAIL fp4_slot_alias_serve: superroc binary not executable: $BIN"
  exit 1
fi
PORT=${SUPERROC_FP4_TEST_PORT:-18231}
OUT=${SUPERROC_FP4_TEST_OUT:-$(pwd)}
LOG=$OUT/fp4_slot_alias_serve.log
ANS=$OUT/fp4_slot_alias_serve.ans.json
rm -f "$LOG" "$ANS"
P=0
cleanup() {
  if [ "$P" -ne 0 ]; then
    kill "$P" 2>/dev/null
    for _ in 1 2 3 4 5; do kill -0 "$P" 2>/dev/null || break; sleep 1; done
    kill -9 "$P" 2>/dev/null
    wait "$P" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM HUP
# SPEC4 item 2: probe before launch. If something already answers /health on
# the port, the request below would go to that foreign server and a failure
# would misreport as a freeze of ours. FAIL here with the cause instead.
if curl -sf -m 1 "localhost:$PORT/health" >/dev/null 2>&1; then
  echo "FAIL fp4_slot_alias_serve: port $PORT already answers /health before launch - a foreign server holds it; refusing to run"
  exit 1
fi
# Flags are Jay's KAT+MTP seat flags (sq10v2/job.sh): built-in drafter, greedy.
(SUPERROC_FP4_SLOT_TRACE=1 exec "$BIN" serve -i 127.0.0.1 -p "$PORT" -j 1 llm \
  -m "$SUPERROC_FP4_TEST_KAT_MODEL" --served-model-name m --context 32768 \
  --think off -t 0 --presence-penalty 0 -d 3 --min-draft-tokens 1 \
  --speculative mtp) >"$LOG" 2>&1 < /dev/null &
P=$!
up=0
for _ in $(seq 1 100); do
  sleep 2
  if curl -sf -m 3 "localhost:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  kill -0 "$P" 2>/dev/null || break
done
if [ "$up" != 1 ]; then
  echo "FAIL fp4_slot_alias_serve: server never came up on port $PORT"
  # SPEC4 item 3: name a mis-set model file instead of leaving a bare startup
  # failure. A non-KAT file is refused here at load (no embedded nextn block).
  if grep -q 'requires --mtp-model\|Failed to create Qwen MTP model' "$LOG" 2>/dev/null; then
    echo "FAIL fp4_slot_alias_serve: the model file has no embedded KAT+MTP drafter path: $SUPERROC_FP4_TEST_KAT_MODEL"
    grep -h 'requires --mtp-model\|Failed to create Qwen MTP model' "$LOG" | head -n 2 | cut -c1-200
  fi
  tail -n 5 "$LOG" | cut -c1-200
  exit 1
fi
if ! timeout 120 python3 - "$PORT" "$ANS" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
body = {"model": "m",
        "messages": [{"role": "user",
                      "content": "Write a Python function that checks whether a number is prime, with a docstring."}],
        "max_tokens": 48, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    j = json.load(urllib.request.urlopen(req, timeout=110))
    open(out, "w").write(json.dumps(j))
except Exception as e:
    open(out, "w").write(json.dumps({"error": type(e).__name__}))
    sys.exit(3)
PY
then
  echo "FAIL fp4_slot_alias_serve: no answer in 120 s (freeze shape)"
  exit 1
fi
TEXT=$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); m=j.get("choices",[{}])[0].get("message",{}); print((m.get("content") or "")+(m.get("reasoning_content") or ""))' "$ANS" 2>/dev/null)
if [ -z "$TEXT" ]; then
  echo "FAIL fp4_slot_alias_serve: empty answer: $(head -c 300 "$ANS")"
  exit 1
fi
REFUSALS=$(grep -c SUPERROC_FP4_SLOT_REFUSE "$LOG" || true)
if [ "$REFUSALS" != 0 ]; then
  echo "FAIL fp4_slot_alias_serve: $REFUSALS refusal lines (dropped-registration shape)"
  grep SUPERROC_FP4_SLOT_REFUSE "$LOG" | head -n 3 | cut -c1-200
  exit 1
fi
WAITS=$(grep -c SUPERROC_FP4_SLOT_WAIT "$LOG" || true)
if [ "$WAITS" != 0 ]; then
  echo "FAIL fp4_slot_alias_serve: $WAITS wait lines"
  exit 1
fi
SLOTS=$(grep -c 'SUPERROC_FP4_SLOTS phase=load_end' "$LOG" || true)
if [ "$SLOTS" = 0 ]; then
  echo "FAIL fp4_slot_alias_serve: no SUPERROC_FP4_SLOTS load_end line in server log"
  exit 1
fi
# SPEC4 item 3: positive proof the drafter path really ran through the alias.
# Without this a file that loads but never drafts passes vacuously (no
# refusal, no wait, one load_end line). The MTP draft weights must have
# loaded, and the answer's usage must show drafts proposed.
if ! grep -q 'phase=draft_weights' "$LOG" 2>/dev/null; then
  echo "FAIL fp4_slot_alias_serve: no phase=draft_weights line in server log - the MTP draft weights never loaded: $SUPERROC_FP4_TEST_KAT_MODEL"
  exit 1
fi
DRAFTS=$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j.get("usage",{}).get("draft_tokens",-1))' "$ANS" 2>/dev/null || echo -1)
DRAFTS_ACC=$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j.get("usage",{}).get("draft_tokens_accepted",-1))' "$ANS" 2>/dev/null || echo -1)
case "$DRAFTS" in ''|*[!0-9]*) DRAFTS=-1;; esac
if [ "$DRAFTS" -le 0 ]; then
  echo "FAIL fp4_slot_alias_serve: no drafts proposed (usage.draft_tokens=$DRAFTS) - the KAT+MTP drafter path did not run; SUPERROC_FP4_TEST_KAT_MODEL must name a KAT file with a built-in drafter"
  exit 1
fi
echo "SERVE_WIRING PASS (answer ${#TEXT} chars, drafts proposed=$DRAFTS accepted=$DRAFTS_ACC, refusals 0, waits 0)"
exit 0
