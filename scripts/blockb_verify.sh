#!/usr/bin/env bash
# One block B verifier on Modal, capped in wall-clock time (= cost). Run from the repo root:
#   scripts/blockb_verify.sh qwen data/blockb/batch1_raw.jsonl 'data/blockb/batch1_votes_{name}.jsonl' 2400 [chunk]
# Stops the Modal app when the cap is reached or the runner crash-loops, then prints the last progress lines.
# Votes already written survive; rerunning the same command continues where it stopped.
set -u
bounded() { perl -e 'alarm shift @ARGV; exec @ARGV' "$@"; }     # run a command with a wall-clock cap (seconds)
name=$1; inp=$2; out=$3; cap=$4; chunk=${5:-1000}
log="runs/modal/verify_$(basename "${inp%.jsonl}")_${name}.log"
bounded "$cap" uv run modal run src/dopest_model/blockb/verify_modal.py \
  --inp "$inp" --out "$out" --only "$name" --chunk "$chunk" > "$log" 2>&1 &
pid=$!
status=""
while kill -0 "$pid" 2>/dev/null; do
  n=$(grep -c "Runner failed" "$log" 2>/dev/null)          # empty while the log does not exist yet
  if [ "${n:-0}" -ge 2 ]; then status="crash loop"; kill "$pid"; break; fi
  sleep 10
done
wait "$pid"; code=$?
[ "$code" = 124 ] && status="cap of ${cap}s reached"
if [ -n "$status" ]; then
  app=$(grep -oE "ap-[A-Za-z0-9]{20,}" "$log" | head -1)
  echo "STOP ($status): modal app stop ${app:-?}"
  [ -n "$app" ] && bounded 90 uv run modal app stop --yes "$app" 2>&1 | tail -1
fi
echo "exit: $code"
tr '\r' '\n' < "$log" | grep -E "^$name:|FAILED|Traceback|Error:" | grep -v INFO | tail -8
