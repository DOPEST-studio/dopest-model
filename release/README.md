---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: peft
language: [en, pl]
tags: [typed-decisions, decision-model, calibration, lora, jevbench, gemma4]
---

# DOPEST 1.0 (31B)

DOPEST 1.0 is a typed-decision model: it reads a **state** (text or JSON) and a **question with a closed set of
options**, and returns a calibrated probability for every option in one forward pass. No text is generated.

- **choice**: a distribution over named options (more than 16 options are read in groups, then a final round)
- **yes/no** (`noul`): P(yes)
- **score**: a distribution over ordered levels, plus the expected level

It is a LoRA adapter (rank 32, last 16 of 60 decoder layers, 65M parameters) on
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), served with a readout technique of its own
(below) and the calibration file in this repository. Licence: Apache-2.0, as the base model. Built by
[DOPEST](https://dopest.studio).

## How it reads a decision

The prompt lists the options as letters and ends at the answer position; the logits of the option letters give the
distribution. Two things differ from a plain causal readout:

1. **Bidirectional attention inside the prompt on the global-attention layers.** Gemma 4 31B has 10 full-attention
   and 50 sliding-window layers. In the full-attention layers every prompt token attends to the whole prompt, in
   both directions; the sliding layers stay causal. The adapter was trained this way and must be served this way
   (`--bidir full`). On the frozen base this change alone raises accuracy on hard items by about 4 points; the
   sliding layers must not be flipped (they carry local syntax and collapse).
2. **Calibration as a shipped artifact** (`calibration.json`): one temperature per question type and a yes/no commit
   floor of 0.801 (an answer with P(yes) between 0.2 and 0.8 is moved to the floor on its own side, because
   JevBench scores such answers as abstentions).
3. **Two option orders for choice questions** (`--orders 2`, the default): the options are read in the given order
   and in reverse, and the two distributions are averaged by option. The adapter was trained with shuffled options,
   so the orders disagree on only 4% of choice items; averaging adds a little calibration (+0.3 on OpenDecisions,
   +0.9 on the public JevBench items, both inside their intervals) for twice the choice-question latency.
   `--orders 1` restores the single pass.

## Serving

The server speaks the TypeSafe wire format (`POST /v1/systemone`), the one JevBench's harness uses:

```bash
git clone https://github.com/DOPEST-studio/dopest-model && cd dopest-model
uv sync --extra serve
uv run python -m dopest_model.serve --model google/gemma-4-31B-it --adapter dopest-studio/dopest-1.0-31b \
    --calibration calibration.json --bidir full --port 8000
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order #4411 shipped on 3 March; the customer reports it arrived damaged on 9 March.",
  "questions": {"decision": {"type": "choice", "instructions": "Route this ticket.",
    "criteria": {"refund": "Refund or replacement", "shipping": "Delivery status", "other": "Anything else"}}}}'
```

Hardware: one GPU with 80 GB (bf16 weights 62 GB; measured on one H100 80 GB). Dependencies: torch 2.14,
transformers 5.20, peft 0.21, fastapi, uvicorn. States longer than about 12k tokens are truncated, keeping the start.

Measured on one H100 80 GB, one request at a time, over loopback, on the 231 public JevBench items with a single
option order: median 120 ms, p95 751 ms (states of up to 3.7k tokens), 702 input tokens per decision; with two
orders, choice questions take twice as long. The server passes JevBench's adapter conformance kit (`check_adapter`
with the `typesafe` adapter: 4 fixture tasks, 0 violations).

## Training data

All training items are synthetic and licensed Apache-2.0; none comes from JevBench or OpenDecisions:

| Block | Items | Labels | What |
|---|---|---|---|
| A, code-generated | 13.7k | rule engine, exact | long policy documents with exceptions, dates and running totals under stated rules, probability from stated frequencies, small arithmetic and logic |
| B, model-written | 28.9k | DeepSeek V4.1 Flash as author, kept only when Qwen3.8-27B and gpt-oss-120b agreed blind | request routing for an LLM gateway, business decisions over documents, rule-based screening with counterfactual twins, 22 languages |
| Hard families, code-generated | 10.7k | rule engine, exact | monthly cost ledgers with alert thresholds, rosters across clock changes, judging a worked answer against explicit requirements |

Every set was checked for overlap with the evaluation sets (word 8-grams); the maximum overlap of any training item
with OpenDecisions or the JevBench public items is 8%. Options were shuffled per example, so the gold position
carries no signal. One epoch, 21.4M tokens, 93 minutes on one H200.

## Calibration disclosure

The choice and yes/no temperatures were fitted on the 231 public JevBench v1.2 items (allowed by the board's rules;
no training on them). The score temperature was fitted on a held-out split of our own block B (627 items), because
the public set has only 18 score items. Nothing else used the public items.

## Evaluation (our measurements, not board numbers)

Scored with the JevBench v1.5 rules (abstention band for yes/no, tier weights 0.1/0.2/0.3/0.4, equal type weights),
temperatures from a different set than the one scored:

| Set | Capability | Intelligence | Calibration |
|---|---|---|---|
| OpenDecisions (2,352 items; shipped calibration, two orders) | 86.5 | 81.4 | 91.6 |
| JevBench public v1.2 (231 items; shipped calibration, two orders) | 82.9 | 74.1 | 91.7 |

The shipped calibration takes the choice and yes/no temperatures from the public JevBench items, so the public-set
row is not a held-out number for calibration; its accuracy is.

For reference, the frozen base read causally scores 80.9 on OpenDecisions under the same procedure; the frozen base
with the bidirectional readout 85.0; this adapter adds calibration on top. The public set is small: differences
below about 5 points on it are inside its 95% interval.

## Limitations

- Trained on synthetic decisions only; no human-labelled data. Weakest on long numeric reasoning (multi-step
  arithmetic over tables) and on non-English states (about 20 points below English on hard items).
- One forward pass per decision: no reasoning step. Questions that need several arithmetic steps are answered from
  pattern, not computation.
- Calibration was fitted on English public items; validate the thresholds on your own data before automating with
  them.
