# Math CoT Pipeline

**English** | [中文](README_CN.md)

An end-to-end pipeline for math Chain-of-Thought: **public data cleaning → ms-swift LoRA fine-tuning → dual-track evaluation (answer + CoT) → multi-run comparison report → optional vLLM serving**.

Default base model: `Qwen/Qwen2.5-Math-7B-Instruct`. Training encourages full CoT reasoning with a parseable final answer (e.g. `\boxed{}`).

---

## Project scope

**The primary goal of this repository is a reproducible evaluation pipeline and end-to-end workflow**, not SOTA chasing on a small holdout set or exhaustive LoRA hyperparameter tuning.

What we aim to demonstrate:

- Cleaning [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) into a unified schema with train / holdout test splits
- LoRA SFT, merge, and batch inference with **ms-swift**
- Persisting generations and splitting them into **answer track** vs **CoT track**
- Registering metrics via a **registry** and recomputing scores without re-running inference
- Aggregating base vs fine-tuned (and other) runs into JSON + bar charts

Numbers under `reports/metrics/` should be read as **pipeline sanity checks and relative comparisons** (ft vs base), not as definitive proof of model capability. See [Training loss (reference)](#training-loss-reference), [Reference results](#reference-results), and [Metrics & limitations](#metrics--limitations).

---

## Training loss (reference)

With default settings (~**778 train**, single LoRA SFT run, 1 epoch / 44 steps, best checkpoint by `eval_loss`, no systematic hyperparameter search):

![LoRA SFT training loss](docs/results/training_loss.png)

Artifacts:

- Committed snapshot: `docs/results/training_loss.png`
- Local pipeline output: `outputs/lora/images/loss.png` (written by `src/train/sft.py` after `./scripts/run_train.sh`)

> The loss curve is for **pipeline sanity and convergence only**; holdout evaluation (below) is the primary reference for downstream quality.

---

## Reference results

With default settings (~**778 train / 86 holdout test**, single LoRA SFT run, no systematic hyperparameter search), after evaluating both the base model and `outputs/merged/best`:

| Metric | base | ft_best | Δ (ft − base) | Notes |
|--------|------|---------|---------------|-------|
| **acc** | 20.9% | 27.9% | **+7.0%** | Math-equivalent accuracy (`math_equal`) |
| **exact_match** | 15.1% | 17.4% | +2.3% | Normalized string match |
| **extract_rate** | 75.6% | 70.9% | −4.7% | Successfully parsed `\boxed` answer |
| **bert_score** | 0.738 | 0.742 | +0.004 | Semantic F1: pred CoT vs gold COT |
| **informativeness_chain** | 0.902 | 0.900 | ≈ 0 | pred CoT vs problem relevance |

Comparison bar chart (`baseline: base`):

![base vs ft metrics compare](docs/results/compare_all.png)

Artifacts:

- Committed snapshot: `docs/results/compare_all.png`
- Local pipeline output: `reports/metrics/compare_all.json`, `reports/metrics/compare_all.png`

Regenerate the comparison (when metrics JSON files exist):

```bash
python -m src.eval.report --config configs/compare.yaml
```

**Takeaway:** Fine-tuning shows a clear gain on **acc**; CoT semantic scores stay close to base—consistent with optimizing the **pipeline**, not CoT leaderboard scores.

---

## Highlights

- **Data:** NuminaMath-CoT → unified schema, dedup, extract gold `answer` + `COT` from `solution`; **train/test split at clean time** (validation split from train inside ms-swift during SFT)
- **Fine-tuning:** ms-swift LoRA SFT in chat `messages` format; bf16, best checkpoint selection, smoke mode
- **Evaluation:** Holdout `test.jsonl` via **infer → split → run_metrics → report**; answer metrics (EM / acc) + CoT metrics (bert_score / informativeness_chain); predictions on disk for recomputation
- **Serving:** vLLM (`deploy/vllm/` + `configs/serve.yaml`), unified `docker-compose.yml` (`serve` profile)

---

## Pipeline overview

```text
NuminaMath-CoT
  → download (HF → raw/*.jsonl)
  → clean (dedup / extract answer+COT / train+test split)
  → train.jsonl / test.jsonl
                              ↓
        format (optional) → train_messages.jsonl
                              ↓
                   ms-swift LoRA SFT → outputs/lora
                              ↓
              ┌───────────────┴────────────────┐
              ↓                                ↓
        Export merged                      Batch infer (eval)
        → outputs/merged/best              → predictions/*.jsonl
              ↓                                ↓
         deploy/vllm (API)                 Split ŷ | pred_cot
                                               ├─ Answer → EM / acc / extract_rate
                                               └─ CoT    → bert_score / informativeness_chain
                                               ↓
                                         run_metrics → metrics/*.json
                                               ↓
                                         report → compare_all.json + .png
```

Public benchmarks (GSM8K, etc.) via EvalScope / `swift eval` are optional extensions—they do **not** replace the in-repo holdout test path.

---

## Repository layout

```text
math-cot-pipeline/
├── configs/           # data, train, eval_ft, eval_base, compare, serve
├── data/raw/          # raw downloads
├── data/processed/    # train.jsonl, test.jsonl, train_messages.jsonl
├── src/
│   ├── common/        # config, io, extract, registry
│   ├── data/          # download, clean
│   ├── train/         # format, sft, export
│   └── eval/          # infer, split, run_metrics, report
├── scripts/           # run_data, run_train, run_eval_*, run_report
├── tests/
├── docker/
├── deploy/vllm/
├── outputs/           # lora, merged (gitignore recommended)
├── reports/           # predictions/, metrics/ (gitignore)
└── requirements/
```

Detailed eval design: [`src/eval/README_EN.md`](src/eval/README_EN.md).

---

## Quick start

> **GPU:** Training, export, and infer need NVIDIA GPU (≥24GB recommended; 32GB comfortable). Split, run_metrics, and report can run on CPU. CoT `bert_score` works best on GPU with `batch_size: 1`.

### 1. Environment

```bash
conda create -n math-cot python=3.11 -y
conda activate math-cot

pip install -r requirements/base.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/train.txt
pip install -r requirements/eval.txt
```

Optional cache locations (recommended when disk is tight):

```bash
export HF_HOME=/path/to/large/disk/hf
export MODELSCOPE_CACHE=/path/to/modelscope/cache
```

If Hugging Face downloads are slow in your region, try a mirror, e.g. `export HF_ENDPOINT=https://hf-mirror.com`.

`evalscope` has a heavy dependency tree; if `pip install -r requirements/eval.txt` hangs, install step-by-step—see [`src/eval/README_EN.md`](src/eval/README_EN.md).

**Shell scripts (equivalent to `python -m` steps):**

```bash
./scripts/run_data.sh
./scripts/run_train.sh
./scripts/run_eval_ft.sh
./scripts/run_eval_base.sh
./scripts/run_report.sh
```

**Tests:**

```bash
pip install -r requirements/dev.txt
pytest tests/ -v
```

### 2. Data

```bash
cd math-cot-pipeline
python -m src.data.download
python -m src.data.clean
```

| Output | Description |
|--------|-------------|
| `data/raw/numina_math_cot.jsonl` | Raw download |
| `data/processed/train.jsonl` | Training set |
| `data/processed/test.jsonl` | Holdout eval set (never used in training) |
| `data/processed/manifest.json` | Cleaning stats |

Default config example: 1000 raw → 864 kept → **train 778 / test 86**.

Processed fields: `id`, `source`, `problem`, `solution`, `COT` (text before `\boxed`), `answer` (gold label).

### 3. Fine-tuning

```bash
python -m src.train.format          # → train_messages.jsonl
python -m src.train.sft --smoke     # smoke → outputs/lora-smoke
python -m src.train.sft             # full run → outputs/lora
python -m src.train.export          # merge → outputs/merged/best
```

| Step | Module | Output |
|------|--------|--------|
| format | `src/train/format.py` | `train_messages.jsonl` |
| sft | `src/train/sft.py` | `outputs/lora/checkpoint-*` |
| export | `src/train/export.py` | `outputs/merged/best/` |

Training notes (`configs/train.yaml`): bf16, `load_best_model_at_end`, effective batch size 16 (1 × 16 grad accum). For small datasets, use `save_steps` / `eval_steps` = 10.

### 4. Evaluation

#### Single run (ft and base)

```bash
# Fine-tuned (outputs/merged/best)
python -m src.eval.infer       --config configs/eval_ft.yaml
python -m src.eval.infer       --config configs/eval_ft.yaml --limit 4   # smoke
python -m src.eval.split       --config configs/eval_ft.yaml
python -m src.eval.run_metrics --config configs/eval_ft.yaml --task all

# Base model (model_path: null → Hub weights)
python -m src.eval.infer       --config configs/eval_base.yaml
python -m src.eval.split       --config configs/eval_base.yaml
python -m src.eval.run_metrics --config configs/eval_base.yaml --task all
```

`--task` can be `answer`, `cot`, or `all`. CoT metrics require `bert-score` and `sentence-transformers` (DeBERTa + MPNet downloaded on first run).

**Recommended CoT settings** (`eval_ft.yaml` / `eval_base.yaml` → `cot_metrics.metric_args.bert_score`):

```yaml
bert_score:
  model_type: microsoft/deberta-base-mnli
  batch_size: 1      # library default 64 OOMs on long CoT
  device: cuda
  dtype: float32     # do not use bf16 autocast with DeBERTa
informativeness_chain:
  embedding_model: all-mpnet-base-v2
```

#### Multi-run comparison

```bash
python -m src.eval.report --config configs/compare.yaml
```

**Recompute metrics without re-inferring:** run `run_metrics` only—it reads existing `reports/predictions/`.

### 5. Serving (vLLM)

```bash
python -m src.train.export   # if not merged yet
pip install -r requirements/serve.txt vllm
bash deploy/vllm/start.sh
```

Docker: `docker compose --profile serve up -d vllm-api`. Batch eval still uses `src/eval/infer.py` (PtEngine), separate from the vLLM API. See [`deploy/vllm/README.md`](deploy/vllm/README.md).

---

## Metrics & limitations

### Answer track

| Metric | Meaning |
|--------|---------|
| `exact_match` | Normalized string equality with gold answer |
| `acc` | Math equivalence via EvalScope `math_equal` |
| `extract_rate` | Fraction of samples where `\boxed` (or configured strategy) was parsed |

### CoT track

| Metric | Compares | Meaning |
|--------|----------|---------|
| `bert_score` | pred_cot ↔ gold `COT` | DeBERTa token-level semantic F1 (greedy matching) |
| `informativeness_chain` | pred_cot ↔ `problem` | MPNet sentence embedding cosine (simplified ROSCOE) |

**Important limitations:**

1. **`bert_score` truncates at 512 DeBERTa tokens.** Long math CoT often exceeds this (~90% of our holdout). The score mainly reflects the **first part** of reasoning vs gold—not full-CoT quality or proof correctness.

2. **`informativeness_chain` uses MPNet** with its own context limit (~384 tokens), applied differently (whole-text embedding).

3. **High CoT score ≠ correct answer**; **high acc ≠ CoT matches gold wording**. Use both tracks together as pipeline signals.

4. **Limited training budget** (~778 samples, one SFT run)—numbers are for **ft vs base trends**, not final model claims.

More detail: [`src/eval/README_EN.md`](src/eval/README_EN.md) (bert_score algorithm, OOM fixes, registry extension).

---

## Docker

| Profile | Services | GPU |
|---------|----------|-----|
| `cpu` | test, data, report, eval-ft-metrics | No |
| `repro` | train, export, eval-ft, eval-base | Yes |
| `full` | Combined | Varies |
| `serve` | vllm-api | Yes |

```bash
docker compose build
docker compose --profile cpu run --rm test
docker compose --profile cpu run --rm data
docker compose --profile repro run --rm eval-ft
docker compose --profile serve up -d vllm-api
```

Do not run infer and vLLM on the same GPU simultaneously. Non-Docker usage: [`scripts/README.md`](scripts/README.md).

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/data.yaml` | Source, cleaning, holdout test |
| `configs/train.yaml` | Model, LoRA, training, export, smoke |
| `configs/eval_ft.yaml` | Fine-tuned eval chain |
| `configs/eval_base.yaml` | Base model chain (`base_*` prefixes) |
| `configs/compare.yaml` | Multi-run report |
| `configs/serve.yaml` | vLLM paths and ports |

**Two different `model_type` fields in eval yaml:**

| Field | Model | Used for |
|-------|-------|----------|
| `infer.model_type` | `qwen2_5_math` | ms-swift loads Qwen for inference |
| `cot_metrics...bert_score.model_type` | `microsoft/deberta-base-mnli` | BERTScore judge model |

---

## Training sample format

```json
{
  "messages": [
    {"role": "user", "content": "<problem>"},
    {"role": "assistant", "content": "<solution>"}
  ]
}
```

Generated by `src/train/format.py` from `train.jsonl` using `train.yaml` → `train_schema`.

---

## FAQ

| Issue | Fix |
|-------|-----|
| CoT `bert_score` CUDA OOM | Set `batch_size: 1` in yaml (not library default 64) |
| `BFloat16 overflow` with DeBERTa | Use `dtype: float32` |
| Hugging Face timeout | Retry; set `HF_HOME`; use mirror if needed |
| Offline mode missing models | Download DeBERTa + MPNet first, then `TRANSFORMERS_OFFLINE=1` |
| MPNet not found | `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"` |
| Infer OOM | `max_batch_size: 1` or lower `max_new_tokens` |

---

## Tech stack

Python 3.11 · **ms-swift** · Hugging Face · PyTorch bf16 · **EvalScope** (`math_equal`) · **bert-score** · **sentence-transformers** · **matplotlib** · **vLLM** · **Docker Compose**

---

## Status

| Component | Status |
|-----------|--------|
| `src/common/`, `src/data/`, `src/train/`, `src/eval/` | Done |
| `configs/*.yaml`, `scripts/`, `tests/` (27 cases) | Done |
| `deploy/vllm/`, `docker/` + `docker-compose.yml` | Done |
| Holdout eval (base + ft + compare report) | Verified on default config |
| `.dockerignore` | Optional |

---

## License

MIT
