---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: peft
language: [en, pl]
tags: [typed-decisions, decision-model, calibration, lora, jevbench, gemma4]
---

# DOPEST 1.1 (31B)

DOPEST is a typed-decision model: it reads a **state** (text or JSON) and a **question with a closed set of
options**, and returns a calibrated probability for every option in one forward pass. When the one-pass answer to a
choice or score question is unsure, an optional second pass lets the base model write a two-sentence rationale and
reads the options again after it (`--reason`, below); otherwise no text is generated. Version 1.1 keeps the 1.0
adapter and calibration and adds that second pass.

- **choice**: a distribution over named options (more than 16 options are read in groups, then a final round)
- **yes/no** (`noul`): P(yes)
- **score**: a distribution over ordered levels, plus the expected level

It is a LoRA adapter (rank 32, last 16 of 60 decoder layers, 65M parameters) on
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), served with a readout technique of its own
(below) and the calibration file in this repository. Licence: Apache-2.0, as the base model. Built by
[DOPEST](https://dopest.studio).

## How it reads a decision

The prompt lists the options as letters and ends at the answer position; the logits of the option letters give the
distribution. Four things differ from a plain causal readout:

1. **Bidirectional attention inside the prompt on the global-attention layers.** Gemma 4 31B has 10 full-attention
   and 50 sliding-window layers. In the full-attention layers every prompt token attends to the whole prompt, in
   both directions; the sliding layers stay causal. The adapter was trained this way and must be served this way
   (`--bidir full`). On the frozen base this change alone raises accuracy on hard items by about 4 points; the
   sliding layers must not be flipped (they carry local syntax and collapse).
2. **Calibration as a shipped artifact** (`calibration.json`): one temperature per question type and a yes/no commit
   floor of 0.801 (an answer with P(yes) between 0.2 and 0.8 is moved to the floor on its own side, because
   JevBench scores such answers as abstentions).
3. **Optional second option order for choice questions** (`--orders 2`): the options are read in the given order
   and in reverse, and the two distributions are averaged by option. The adapter was trained with shuffled options,
   so the orders disagree on only 4% of choice items; averaging adds a little calibration (+0.3 on OpenDecisions,
   +0.9 on the public JevBench items, both inside their intervals) for twice the choice-question latency and about
   70% more input tokens per decision. Off by default: the extra tokens would put the model outside the JevBench
   Jev-class cost cap (twice the reference model's cost per decision).
4. **Reasoning at uncertainty** (`--reason`, version 1.1). When the top probability of a choice question
   with at most 16 options is below 0.75, or of a score question below 0.8, the base model (adapter switched off)
   receives the same state, criterion and options (the levels of a score are its options) and writes a rationale of
   at most two sentences and 40 words that ends in `Answer:`; the option letters are read right after it and the two
   distributions are mixed half and half (the second one at temperature 3; every number of this section is in
   `calibration.json`, section `reasoning`). The adapter was trained only at the answer position under the
   bidirectional mask, so its own generations are untested; the base model's causal readout after its reasoning is
   an independent second view of the question. The pass decodes one token at a time through a captured CUDA graph
   over a static KV cache (about 50 ms per token on an H100, 76 ms on an A100 80 GB) and stops at the marker or at 80
   tokens. The whole decision, first readout included, is budgeted at 6.5 seconds of wall-clock time: a second pass
   that could not fit 40 tokens in what is left is not started, and one that is running stops decoding in time to
   read the answer (the board fails any decision slower than 10 seconds). A cost budget (0.014 USD per 1,000
   decisions on average, at the base model's list prices for the re-read state and the generated tokens) bounds how
   many second passes a run may take, so the pooled cost stays inside the Jev-class cap. Measured with our scoring
   (JevBench v1.5 rules), with rationales allowed to finish and both question types gated as shipped: all of
   OpenDecisions 86.2 → 88.9 Capability (351 of 2,352 decisions reasoned; choice alone 86.2 → 88.3, score alone
   86.2 → 87.1), the hard tier 79.1 → 81.1, the 231 public JevBench items 82.0 → 85.0; on our replica of the sealed
   set, whose choice questions the one pass already gets right, the mix cost 0.5 points. These are upper bounds for
   the served model: under the time budget, the longest states leave room for fewer tokens (in the A100 run below
   most second passes on the public items ran into it), and a rationale cut mid-sentence after 60 tokens kept only
   about a third of the hard-tier gain. The second read without any rationale lowered the hard-tier score by 1.7
   points, so no second pass is taken when the budget cannot fit a short rationale. Scaling the adapter down
   (0.5 to 0.85 of its trained strength) raised the one-pass score accuracy by about 4 points but did not add to the
   second pass, so the adapter is served at full strength.

## Serving

The server speaks the TypeSafe wire format (`POST /v1/systemone`), the one JevBench's harness uses:

```bash
git clone https://github.com/DOPEST-studio/dopest-model && cd dopest-model
uv sync --extra serve
uv run python -m dopest_model.serve --model google/gemma-4-31B-it --adapter dopest-studio/dopest-1.0-31b \
    --calibration calibration.json --bidir full --reason --port 8000
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Order #4411 shipped on 3 March; the customer reports it arrived damaged on 9 March.",
  "questions": {"decision": {"type": "choice", "instructions": "Route this ticket.",
    "criteria": {"refund": "Refund or replacement", "shipping": "Delivery status", "other": "Anything else"}}}}'
```

Hardware: one GPU with 80 GB (bf16 weights 62 GB plus about 2 GB for the second pass's cache). Dependencies: torch
2.14, transformers 5.20, peft 0.21, fastapi, uvicorn. States longer than about 12k tokens are truncated, keeping the
start (a longer first readout would not fit the 10-second rule on an A100); the second pass is skipped for prompts
beyond 8k tokens, whenever the cost budget is spent and when too little of the time budget is left, and the one-pass
answer stands. Without `--reason` the adapter is merged into the weights at start-up (42% lower latency, the same
argmax on all 231 public items, probabilities within 0.02); with it the adapter stays separate, because the base
model must remain reachable for the rationale (`--reason-with-adapter` lets the adapted model reason instead; it was
no better in our tests).
`usage.output_tokens` in the response counts the generated tokens, `usage.input_tokens` every token read in both
passes.

Measured one request at a time, over loopback, on the 231 public JevBench items with a single option order. Without
the second pass (the 1.0 server, adapter not merged), on one H100 80 GB: median 120 ms, p95 751 ms (states of up to
3.7k tokens), 702 input tokens per decision (about 0.063 USD per 1,000 decisions at the board's reference price for
the base model); merging the adapter brings the median to about 60 ms in-process, and with `--orders 2` choice
questions take twice as long. With `--reason`, on one A100 80 GB (the board's GPU), the cost budget raised to 0.03 for
the run so that every eligible item took the pass: median 129 ms, p95 6.5 s, maximum 6.5 s; 21 of the 231 decisions
took the second pass and 4 more were skipped for lack of time; 832 input and 5.4 output tokens per decision (about
0.077 USD per 1,000 on the public items, which are longer than the board's average; on the board's 1,624 items the
first pass cost 0.049 USD per 1,000 and the budget adds at most 0.014). The server passes JevBench's adapter
conformance kit (`check_adapter` with the `typesafe` adapter: 4 fixture tasks, 0 violations).

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

## Official result (version 1.0, one pass)

JevBench v1.7.4 (Benchmark Heaven, 26 October 2026, run by the board on its own GPU from this repository's code
without the second pass): Capability **82.0** (Intelligence 73.9, Calibration 90.2), third among open-weight
Jev-class systems at the time; 0.049 USD per 1,000 decisions, median latency 0.236 s. Chance-corrected competence on
the sealed split: choice 79.3, yes/no 77.8, score 56.5. The score type is the weak one. Version 1.1 has no board
result yet.

## Evaluation (our measurements of the one-pass server, not board numbers)

Scored with the JevBench v1.5 rules (abstention band for yes/no, tier weights 0.1/0.2/0.3/0.4, equal type weights),
temperatures from a different set than the one scored:

| Set | Capability | Intelligence | Calibration |
|---|---|---|---|
| OpenDecisions (2,352 items; shipped calibration, one option order) | 86.2 | 81.2 | 91.1 |
| JevBench public v1.2 (231 items; shipped calibration, one option order) | 82.0 | 73.7 | 90.2 |

With `--orders 2` the same runs score 86.5 and 82.9. The shipped calibration takes the choice and yes/no
temperatures from the public JevBench items, so the public-set row is not a held-out number for calibration; its
accuracy is.

For reference, the frozen base read causally scores 80.9 on OpenDecisions under the same procedure; the frozen base
with the bidirectional readout 85.0; this adapter adds calibration on top. The public set is small: differences
below about 5 points on it are inside its 95% interval.

## Limitations

- Trained on synthetic decisions only; no human-labelled data. Weakest on long numeric reasoning (multi-step
  arithmetic over tables) and on non-English states (about 20 points below English on hard items).
- One forward pass per decision by default. The second pass covers only unsure choice and score questions, is
  capped at two sentences and the decision's time budget, and does not run for yes/no questions, so questions that
  need several arithmetic steps are still mostly answered from pattern, not computation.
- Calibration was fitted on English public items; validate the thresholds on your own data before automating with
  them.
