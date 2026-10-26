# dopest-model

Code behind [DOPEST 1.1 (31B)](https://huggingface.co/dopest-studio/dopest-1.0-31b), a typed-decision model: it
reads a state (text or JSON) and a question with a closed set of options, and returns a calibrated probability for
every option in one forward pass; unsure choice questions can get a second pass in which the base model writes a
two-sentence rationale before the options are read again. The adapter, its calibration file and the model card are on
Hugging Face; this repository holds the server, the evaluation code and the data generators.

## Serve

One GPU with 80 GB (bf16 weights of Gemma 4 31B), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra serve
uv run python -m dopest_model.serve --model google/gemma-4-31B-it --adapter dopest-studio/dopest-1.0-31b \
    --calibration release/calibration.json --bidir full --reason --port 8000
```

The server speaks the TypeSafe wire format (`POST /v1/systemone`, `GET /health`); the docstring of
`src/dopest_model/serve.py` gives the request and response shapes. `--reason` turns on the second pass for unsure
choice questions (gate, mix, length and cost budget come from the `reasoning` section of the calibration file;
`src/dopest_model/reason.py` explains the design; the measurements are in the model card). `--orders 2` reads choice questions
in both option orders and averages the two distributions (off by default: about 70% more input tokens per decision,
which is more than the JevBench Jev-class cost cap allows). `--bidir full` must stay on: the adapter was trained with
bidirectional attention inside the prompt on the global-attention layers, and the plain causal readout is a different
model.

## Evaluate

```bash
uv run python -m dopest_model.jevbench_public --out data/dev/jevbench_public.jsonl   # the public JevBench items
uv run python -m dopest_model.evaluate --help                                         # scoring with the JevBench rules
uv run python -m dopest_model.compare --help                                          # runs side by side, bootstrap intervals
```

Scoring follows the JevBench method: abstention band for yes/no, tier and type weights, calibration pooled over
types. `src/dopest_model/jevscore.py` is vendored from OpenDecisions.

## Layout

- `src/dopest_model/serve.py`: the server
- `src/dopest_model/reason.py`: the second pass (policy, cost budget, CUDA-graph decoding); `fastserve.py`: merged
  adapter, quantisation and graph levers for the one-pass readout
- `src/dopest_model/torch_backend.py`: readout, attention masks for the bidirectional layers, LoRA training
- `src/dopest_model/evaluate.py`, `compare.py`, `calibrate.py`: scoring and calibration
- `src/dopest_model/gen/`: rule-engine data generators (exact labels)
- `src/dopest_model/blockb/`: the model-written block (author, two verifiers, agreement filter, twins, assembly)
- `src/dopest_model/modal_app.py`: training and evaluation jobs on Modal
- `release/`: the model card and calibration as published
- `tests/`: `uv run pytest -q tests/`

## License

Apache-2.0, for the code here and for the adapter weights.
