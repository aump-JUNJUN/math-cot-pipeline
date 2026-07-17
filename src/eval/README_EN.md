# Eval: Math CoT Evaluation System

**English** | [中文](README_CN.md)

This directory implements the evaluation layer. The goal is not a single accuracy number, but **two parallel tracks**:

1. **Answer evaluation:** Is the final answer correct? (Exact Match, math-equivalent acc, etc.)
2. **CoT evaluation:** Does the reasoning look reasonable? (bert_score, informativeness_chain, etc.)

Design principle: **thin layer on top of ms-swift / EvalScope**—the framework handles batch inference; this project handles **persisted predictions → split → metrics → multi-run reports**.

> End-to-end pipeline and quick start: root **[README_EN.md](../../README_EN.md)**.

---

## Project scope (how to read metrics)

**The primary goal is a working channel, multi-run comparison, and extensible metrics**—not SOTA on 86 holdout samples or proving that fine-tuning is "strong."

Therefore:

- Numbers in `reports/metrics/*.json` are best used as **ft vs base relative comparison** and pipeline sanity checks
- **Do not** treat acc or bert_score alone as sufficient evidence of model capability
- Training uses default-scale data (~778 train), single LoRA run, no systematic hyperparameter search

Full project framing: [README_EN.md § Project scope](../../README_EN.md#project-scope).

---

## Reference results (default config, n=86)

After running base and `outputs/merged/best` on holdout `test.jsonl` (`baseline: base`):

| Metric | base | ft_best | Δ |
|--------|------|---------|---|
| acc | 0.209 | 0.279 | **+0.070** |
| exact_match | 0.151 | 0.174 | +0.023 |
| extract_rate | 0.756 | 0.709 | −0.047 |
| bert_score | 0.738 | 0.742 | +0.004 |
| informativeness_chain | 0.902 | 0.900 | ≈ 0 |

**Reading:** Fine-tuning improves **acc (math equivalence)** noticeably; CoT semantic scores stay close to base—expected when the focus is the **pipeline**, not CoT leaderboard scores.

Reproduce: `python -m src.eval.report --config configs/compare.yaml` → `compare_all.json` + `.png`.

---

## Implementation status

| Module | Status | Entry |
|--------|--------|-------|
| `infer.py` | Done | `python -m src.eval.infer --config configs/eval_ft.yaml` |
| `split.py` | Done | `python -m src.eval.split --config configs/eval_ft.yaml` |
| `metrics.py` | Done | Metric classes (`BertScoreMetric`, etc.), dispatched by `run_metrics` |
| `roscoe_utils.py` | Done | `python -m src.eval.roscoe_utils` (smoke test) |
| `run_metrics.py` | Done | `python -m src.eval.run_metrics --config configs/eval_ft.yaml` |
| `report.py` | Done | `python -m src.eval.report --config configs/compare.yaml` |
| `common/registry.py` | Done | `ANSWER_METRIC_REGISTRY` / `COT_METRIC_REGISTRY` |
| base + ft + compare | Done | Verified with default yaml |

---

## Relationship to the training stack

Fine-tuning uses **ms-swift**. The official eval path is `swift eval`:

```text
swift eval
  → optional serving
  → EvalScope TaskConfig
  → batch infer (InferEngine.infer)
  → completions → EvalScope ModelOutput
  → answer-side metrics (acc, etc.)
```

This project does **not** fork that path. Instead:

- Same `PtEngine.infer` for batch inference (`infer.py`)
- Persist predictions, then split / score / compare runs

LoRA is merged at **export** time; the ft chain loads `outputs/merged/best`.

### ms-swift 3.4 notes

| Scenario | Config |
|----------|--------|
| Local merged model | `infer.model_path` + **`infer.model_type: qwen2_5_math`** (required) |
| Base model (Hub) | `model_path: null`, falls back to `model_id` |

### Two different `model_type` fields (do not confuse)

| Config path | Model | Purpose |
|-------------|-------|---------|
| `infer.model_type` | `qwen2_5_math` | ms-swift loads Qwen 7B for inference |
| `cot_metrics.metric_args.bert_score.model_type` | `microsoft/deberta-base-mnli` | BERTScore judge model |

---

## End-to-end flow

```text
test.jsonl (gold labels Y — not fed to the model at inference)

  [Single run: eval_ft.yaml or eval_base.yaml]
  → infer.py        batch infer (GPU) → reports/predictions/{name}.jsonl
  → split.py        extract ŷ + pred_cot → *_answers.jsonl / *_cots.jsonl
  → run_metrics.py  aggregate metrics  → *_answer.json / *_cot.json

  [Multi-run compare: compare.yaml]
  → report.py       read metrics JSON → compare_all.json + .png
```

```text
                    ┌─ pred_answer / ŷ ─→ answer_metrics (exact_match / acc / …)
generated_text ─→ split ─┤
                    └─ pred_cot ────────→ cot_metrics (bert_score / informativeness_chain)
```

Key points:

- **`generated_text`** is persisted under `predictions/`—decouples inference from scoring
- **Gold Y:** `answer` / `COT` from cleaning; never shown to the model during infer
- **Predicted ŷ:** same `extract` strategy as data cleaning
- **Compare report:** aggregate after each run finishes metrics

### Recommended commands

```bash
conda activate math-cot
export HF_HOME=/path/to/hf/cache    # optional, large disk
cd math-cot-pipeline
pip install -r requirements/eval.txt

# Fine-tuned chain
python -m src.eval.infer       --config configs/eval_ft.yaml
python -m src.eval.split       --config configs/eval_ft.yaml
python -m src.eval.run_metrics --config configs/eval_ft.yaml --task all

# Base chain
./scripts/run_eval_base.sh

# Comparison report (CPU)
python -m src.eval.report --config configs/compare.yaml
```

`--task answer|cot|all` can be run separately. First CoT run downloads DeBERTa (~0.5GB) and MPNet (~0.4GB).

---

## `predictions/` vs `metrics/`

```text
reports/
├── predictions/
│   ├── ft.jsonl / ft_answers.jsonl / ft_cots.jsonl
│   ├── base.jsonl / base_answers.jsonl / base_cots.jsonl
│   └── ...
└── metrics/
    ├── ft_answer.json / ft_cot.json
    ├── base_answer.json / base_cot.json
    ├── compare_all.json
    └── compare_all.png
```

- Change metrics or rescore CoT: **rerun `run_metrics` only**—no re-inference
- `reports/` is gitignored; copy `metrics/*.json` / `.png` to `docs/results/` for publishing

---

## Gold Y vs prediction ŷ

| Symbol | Meaning | Source |
|--------|---------|--------|
| **Y (answer)** | Gold answer | clean: `solution` → `answer` |
| **Y (COT)** | Gold reasoning | clean: `solution` → `COT` |
| **ŷ (pred_answer)** | Predicted answer | split: `extract` on `generated_text` |
| **pred_cot** | Predicted CoT | split: text before `\boxed` (`cot_mode: before_answer`) |

### Why no step-based ROSCOE (v1)

- Gold COT is often numbered steps (`1. 2. 3.`)
- Model `pred_cot` is often paragraph-style (`Given` / `Next` / `Therefore`)—structures do not align

So v1 uses **whole-segment metrics**:

- **`bert_score`:** pred_cot vs gold `COT`
- **`informativeness_chain`:** pred_cot vs `problem` (simplified ROSCOE)

---

## Metric design

### File roles

| File | Role |
|------|------|
| [`metrics.py`](metrics.py) | Metric classes + `@register_*` |
| [`run_metrics.py`](run_metrics.py) | join → instantiate → aggregate → write JSON |
| [`report.py`](report.py) | Multi-run JSON + PNG |
| [`roscoe_utils.py`](roscoe_utils.py) | informativeness_chain embeddings |
| [`common/registry.py`](../common/registry.py) | Dual registries |

### Three metric interfaces

| Interface | Metrics | Input |
|-----------|---------|-------|
| `apply(preds, refs)` | `exact_match`, `acc`, `bert_score` | String lists pred + gold |
| `apply_from_rows(pred_rows)` | `extract_rate` | Reads `extract_ok` |
| `apply_with_context(rows, ...)` | `informativeness_chain` | `pred_cot` + `problem` |

### Answer track

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalized string equality |
| `acc` | Math equivalence via EvalScope `math_equal` |
| `extract_rate` | Fraction with successful answer extraction |

### CoT track

| Metric | Compares | Description |
|--------|----------|-------------|
| `bert_score` | pred_cot ↔ gold `COT` | DeBERTa token-level semantic F1 (see below) |
| `informativeness_chain` | pred_cot ↔ `problem` | MPNet cosine mapped to [0, 1] |

---

## bert_score in depth

### Algorithm (per sample)

1. Tokenize pred_cot and gold COT with DeBERTa; **truncate at 512 tokens**
2. Encode token vectors (default: layer 9 hidden states)
3. **Greedy bidirectional matching:** each pred token → best gold token (Precision); reverse for Recall
4. IDF-weighted F1; mean over 86 samples → `{name}_cot.json`

### Limitations (read this)

| Limitation | Detail |
|------------|--------|
| **512-token cap** | ~90%+ of long math CoT hit the limit; score reflects **early reasoning** vs gold |
| **Semantic ≠ correct** | Similar wording does not mean valid math; use with `acc` |
| **Tail content** | Repetition, truncated endings often outside the window |
| **Library default batch=64** | OOM on long CoT + GPU; **set `batch_size: 1` in yaml** |

### Recommended yaml config

```yaml
cot_metrics:
  metric_args:
    bert_score:
      model_type: microsoft/deberta-base-mnli
      batch_size: 1
      device: cuda
      dtype: float32    # do not use bfloat16 — DeBERTa + autocast overflows
    informativeness_chain:
      embedding_model: all-mpnet-base-v2
```

[`BertScoreMetric`](metrics.py) passes these to `bert_score.score(...)`. Non-fp32 `dtype` uses `torch.autocast` on CUDA (fp32 is recommended for DeBERTa).

### informativeness_chain notes

- Model: `all-mpnet-base-v2` (sentence-transformers)
- One text encoded at a time; 86 × 2 (pred + problem); VRAM <1GB
- Whole-sentence embedding; ~**384 token** limit—different from bert_score truncation

---

## Sample `run_metrics` output

`reports/metrics/ft_answer.json`:

```json
{
  "task": "answer_metrics",
  "n_samples": 86,
  "metrics": {
    "exact_match": 0.174,
    "acc": 0.279,
    "extract_rate": 0.709
  }
}
```

`reports/metrics/ft_cot.json`:

```json
{
  "task": "cot_metrics",
  "n_samples": 86,
  "metrics": {
    "bert_score": 0.742,
    "informativeness_chain": 0.900
  }
}
```

`compare_all.json` excerpt:

```json
{
  "task": "compare",
  "baseline": "base",
  "answer": {
    "acc": {
      "values": { "base": 0.209, "ft_best": 0.279 },
      "delta": { "ft_best": 0.070 }
    }
  },
  "cot": {
    "bert_score": {
      "values": { "base": 0.738, "ft_best": 0.742 },
      "delta": { "ft_best": 0.004 }
    }
  }
}
```

Add a new experiment: copy `eval_ft.yaml` → `eval_ft_<name>.yaml`, run metrics, append to `compare.yaml` → `runs`.

---

## Intermediate artifact schema

### infer: `reports/predictions/{name}.jsonl`

```json
{
  "id": "000732",
  "problem": "...",
  "generated_text": "Full model output... \\boxed{(0,4)}",
  "model_path": "outputs/merged/best",
  "model_id": "Qwen/Qwen2.5-Math-7B-Instruct"
}
```

### split: `_answers.jsonl` / `_cots.jsonl`

See [`split.py`](split.py): `pred_answer` + `extract_ok` / `pred_cot` + `extract_ok`.

---

## Configuration

| File | Purpose |
|------|---------|
| [`configs/eval_ft.yaml`](../../configs/eval_ft.yaml) | Fine-tuned chain |
| [`configs/eval_base.yaml`](../../configs/eval_base.yaml) | Base chain (`model_path: null`, `base_*` prefix) |
| [`configs/compare.yaml`](../../configs/compare.yaml) | Multi-run comparison |

Single-run yaml blocks: `infer` / `split` / `answer_metrics` / `cot_metrics`.

---

## Modules & dependencies

```text
src/eval/
├── infer.py          # GPU batch inference
├── split.py          # pred_answer / pred_cot
├── metrics.py        # Metric classes (incl. BertScoreMetric)
├── roscoe_utils.py   # informativeness_chain
├── run_metrics.py    # Orchestration + CLI
└── report.py         # Compare JSON + PNG
```

| Stage | Dependencies | Device |
|-------|--------------|--------|
| infer | requirements/train.txt (ms-swift) | GPU |
| split | requirements/base.txt | CPU |
| answer metrics | evalscope | CPU |
| cot metrics | bert-score, sentence-transformers | GPU recommended (bert_score) |
| report | matplotlib | CPU |

### Install

```bash
pip install -r requirements/eval.txt
```

If pip resolver hangs:

```bash
pip install "evalscope==1.8.1" matplotlib
pip install bert-score sentence-transformers
```

Pre-download MPNet:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

---

## FAQ (eval-specific)

| Issue | Fix |
|-------|-----|
| bert_score CUDA OOM | `batch_size: 1`; not library default 64 |
| `BFloat16 overflow` | `dtype: float32` |
| Hugging Face timeout | Set `HF_HOME`; retry; use mirror if needed |
| Offline `LocalEntryNotFound` | Download DeBERTa + MPNet online first, then `TRANSFORMERS_OFFLINE=1` |
| MPNet fails after bert_score | Separate models; cache MPNet explicitly |
| infer OOM | `max_batch_size: 1` |
| Rescore only | `run_metrics --task cot` without rerunning infer |

---

## Extending metrics

1. Implement a class in [`metrics.py`](metrics.py) with `@register_answer_metric` or `@register_cot_metric`
2. Add to yaml `metrics:` list and `metric_args`
3. Rerun `run_metrics` (no infer needed)

For context-aware CoT metrics, implement `apply_with_context`—see `InformativenessChainMetric`.

---

## Out of scope (v1)

- Fork ms-swift `EvalModel` as the main path
- LLM-as-judge for step correctness
- Step-based ROSCOE (pred/gold CoT structure mismatch)
- Treating holdout metrics as final model capability claims

---

## References

### Papers (CoT metrics)

- **bert_score** — Zhang, T., Kishore, V., Wu, F., et al. [*BERTScore: Evaluating Text Generation with BERT*](https://arxiv.org/abs/1904.09675). ICLR 2020.
- **informativeness_chain** — Golovneva, O., et al. [*ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning*](https://arxiv.org/abs/2212.07919). ICLR 2023. This repo uses a **whole-text simplified** variant, not step-based ROSCOE.

### Tools & docs

- [ms-swift](https://github.com/modelscope/ms-swift) · [EvalScope](https://github.com/modelscope/evalscope) · [ParlAI ROSCOE](https://github.com/facebookresearch/ParlAI/tree/main/projects/roscoe)
- [README_EN.md](../../README_EN.md) — full pipeline, project scope, quick start
