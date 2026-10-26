#!/usr/bin/env bash
# The replica pipeline on Modal, one stage per call; every model stage appends rows and skips the ones already done,
# so a stage can be rerun after a cap or a failed chunk.
#
#   scripts/replica.sh author  [cap seconds]   both authors in parallel (gpt-oss on H200, Qwen on H100)
#   scripts/replica.sh collect                 valid items -> data/replica/items.jsonl
#   scripts/replica.sh review  [cap seconds]   each engine answers the other author's items blind, then judges the gold
#   (blind / verdict: the two passes as separate runs)
#   scripts/replica.sh select                  accepted items per cell up to the quota -> data/dev/replica_v1.jsonl
set -u
bounded() { perl -e 'alarm shift @ARGV; exec @ARGV' "$@"; }     # run a command with a wall-clock cap (seconds)
D=data/replica; L=runs/modal
mkdir -p "$D" "$L"
stage=${1:?stage}; cap=${2:-2400}; max_tokens=${MAX_TOKENS:-6000}; review_budget=${THINK_BUDGET:-2000}

run_engine() {                                   # stage engine [modal args...]
  local st=$1 eng=$2; shift 2
  bounded "$cap" uv run modal run src/dopest_model/blockb/verify_modal.py::replica --stage "$st" --engine "$eng" "$@" \
    > "$L/replica-$st-$eng.log" 2>&1
  echo "$st/$eng exit $?" | tee -a "$L/replica-$st-$eng.log"
}

case $stage in
  author)
    run_engine author gptoss --inp $D/briefs.jsonl --out $D/author_gptoss.jsonl --max-tokens "$max_tokens" &
    run_engine author qwen --inp $D/briefs.jsonl --out $D/author_qwen.jsonl --max-tokens "$max_tokens" --think-budget 0 &
    wait ;;
  collect)
    uv run python -m dopest_model.replica.assemble collect --rows $D/author_gptoss.jsonl,$D/author_qwen.jsonl \
      --out $D/items.jsonl ;;
  review)
    bounded "$cap" uv run modal run src/dopest_model/blockb/verify_modal.py::review --inp $D/items.jsonl --out-dir $D \
      --engine gptoss --max-tokens 2000 > $L/replica-review-gptoss.log 2>&1 &
    bounded "$cap" uv run modal run src/dopest_model/blockb/verify_modal.py::review --inp $D/items.jsonl --out-dir $D \
      --engine qwen --max-tokens 2000 --think-budget "$review_budget" > $L/replica-review-qwen.log 2>&1 &
    wait ;;
  blind)
    run_engine blind gptoss --inp $D/items.jsonl --out $D/blind_gptoss.jsonl --max-tokens 2000 &
    run_engine blind qwen --inp $D/items.jsonl --out $D/blind_qwen.jsonl --max-tokens 2000 --think-budget "$review_budget" &
    wait ;;
  verdict)
    run_engine verdict gptoss --inp $D/items.jsonl --blind $D/blind_gptoss.jsonl --out $D/verdict_gptoss.jsonl --max-tokens 2000 &
    run_engine verdict qwen --inp $D/items.jsonl --blind $D/blind_qwen.jsonl --out $D/verdict_qwen.jsonl --max-tokens 2000 --think-budget "$review_budget" &
    wait ;;
  select)
    uv run python -m dopest_model.replica.assemble select --items $D/items.jsonl \
      --verdicts $D/verdict_gptoss.jsonl,$D/verdict_qwen.jsonl --blind $D/blind_gptoss.jsonl,$D/blind_qwen.jsonl \
      --out data/dev/replica_v1.jsonl ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
