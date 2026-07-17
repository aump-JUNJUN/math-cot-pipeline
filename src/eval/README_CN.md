# Eval：数学 CoT 评测体系

本目录承载业务评测相关逻辑。目标不只是报一个准确率，而是把评测拆成两条线：

1. **结果评测**：最终答案对不对（Exact Match / 数学等价 acc 等）
2. **思维链评测**：推理过程是否像样（bert_score、informativeness_chain 等）

设计原则：**尽量不改 ms-swift / EvalScope 大框架**，在成型能力上接薄层——官方负责批量推理引擎，本项目负责「中间结果落盘 + 拆分 + 指标 + 对比报告」。

> 端到端流水线与快速开始见根目录 **[README_CN.md](../../README_CN.md)**。English: [src/eval/README_EN.md](README_EN.md)

---

## 项目定位（如何解读指标）

**本评测体系的首要目标是验证通道可跑通、支持多 run 对比与指标扩展**，而不是在 86 条 holdout 上追求 SOTA 或证明微调「很强」。

因此：

- `reports/metrics/*.json` 中的数字适合作为 **ft vs base 的相对比较** 与 pipeline 冒烟信号
- **不应**单独把 acc / bert_score 当作模型能力的充分证据
- 训练侧采用默认量级数据（约 778 train）、单次 LoRA、未系统调超参

更完整的定位说明见 [README_CN.md § 项目定位](../../README_CN.md#项目定位)。

---

## 参考结果（默认配置，n=86）

在 holdout `test.jsonl` 上跑通 base 与 `outputs/merged/best` 后，`baseline: base` 的对比摘要：

| 指标 | base | ft_best | Δ |
|------|------|---------|---|
| acc | 0.209 | 0.279 | **+0.070** |
| exact_match | 0.151 | 0.174 | +0.023 |
| extract_rate | 0.756 | 0.709 | −0.047 |
| bert_score | 0.738 | 0.742 | +0.004 |
| informativeness_chain | 0.902 | 0.900 | ≈ 0 |

**粗读：** 微调主要在 **acc（数学等价）** 上有可见提升；CoT 语义指标与 base 接近——符合「通道验证为主、CoT 分非优化目标」的预期。

复现：`python -m src.eval.report --config configs/compare.yaml` → `reports/metrics/compare_all.json` + `.png`。

---

## 实现进度

| 模块 | 状态 | 入口 |
|------|------|------|
| `infer.py` | ✅ | `python -m src.eval.infer --config configs/eval_ft.yaml` |
| `split.py` | ✅ | `python -m src.eval.split --config configs/eval_ft.yaml` |
| `metrics.py` | ✅ | 指标类（`BertScoreMetric` 等），由 `run_metrics` 调度 |
| `roscoe_utils.py` | ✅ | `python -m src.eval.roscoe_utils`（smoke test） |
| `run_metrics.py` | ✅ | `python -m src.eval.run_metrics --config configs/eval_ft.yaml` |
| `report.py` | ✅ | `python -m src.eval.report --config configs/compare.yaml` |
| `common/registry.py` | ✅ | `ANSWER_METRIC_REGISTRY` / `COT_METRIC_REGISTRY` |
| base + ft + compare 全链 | ✅ | 默认 yaml 下已跑通 |

---

## 和训练栈的关系

本项目微调走 **ms-swift**。官方评测入口是 `swift eval`，内部大致是：

```text
swift eval
  → 部署或接入推理服务（可选）
  → 拼 EvalScope TaskConfig
  → 批量推理（InferEngine.infer）
  → 将 completions 转为 EvalScope ModelOutput
  → EvalScope 算结果侧指标（acc 等）
```

本项目不 fork 上述路径，而是：

- 用同一套 `PtEngine.infer` 批量推理（见 `infer.py`）
- 自行落盘 predictions，再 split / 算指标 / 多 run 对比

LoRA 与基座的结合发生在 **export 阶段**（merge 成完整权重）；ft 链 `infer` 加载 `outputs/merged/best`。

### ms-swift 3.4 注意点

| 场景 | 配置 |
|------|------|
| 本地 merged 模型 | `infer.model_path` + **`infer.model_type: qwen2_5_math`**（必传） |
| base 模型（Hub） | `model_path: null`，回退到 `model_id` |

### 两个不同的 `model_type`（勿混淆）

| 配置位置 | 模型 | 用途 |
|----------|------|------|
| `infer.model_type` | `qwen2_5_math` | ms-swift 加载 Qwen 7B 推理 |
| `cot_metrics.metric_args.bert_score.model_type` | `microsoft/deberta-base-mnli` | bert_score 裁判模型 |

---

## 总流程

```text
test.jsonl（含金标 Y，推理时不喂给模型）

  【单 run：eval_ft.yaml 或 eval_base.yaml】
  → infer.py        批量推理（GPU）→ reports/predictions/{name}.jsonl
  → split.py        抽取 ŷ + pred_cot  → reports/predictions/{name}_answers.jsonl
                                       reports/predictions/{name}_cots.jsonl
  → run_metrics.py  聚合指标          → reports/metrics/{name}_answer.json
                                       reports/metrics/{name}_cot.json

  【多 run 对比：compare.yaml】
  → report.py       读各 run 的 metrics JSON
                    → reports/metrics/compare_all.json
                    → reports/metrics/compare_all.png（可选）
```

```text
                    ┌─ pred_answer / ŷ ─→ answer_metrics（exact_match / acc 等）
generated_text ─→ split ─┤
                    └─ pred_cot ────────→ cot_metrics（bert_score / informativeness_chain）
```

要点：

- **中间值**：`generated_text` 落盘到 `predictions/`，推理与打分解耦
- **金标 Y**：清洗阶段从 `solution` 抽出 `answer` / `COT`；推理时不喂给模型
- **预测 ŷ**：对 `generated_text` 跑与 data 清洗同一套 `extract` 策略
- **对比报告**：各 run 独立跑完 metrics 后，由 `compare.yaml` 汇总

### 推荐运行顺序

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate math-cot
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope
cd math-cot-pipeline

pip install -r requirements/eval.txt

# ── ft 链 ──
python -m src.eval.infer       --config configs/eval_ft.yaml
python -m src.eval.split       --config configs/eval_ft.yaml
python -m src.eval.run_metrics --config configs/eval_ft.yaml --task all

# ── base 链 ──
./scripts/run_eval_base.sh
# 或分步：infer → split → run_metrics --config configs/eval_base.yaml

# ── 对比报告（CPU）──
python -m src.eval.report --config configs/compare.yaml
```

`--task answer|cot|all` 可分开跑。CoT 首次需下载 DeBERTa（~0.5GB）与 MPNet（~0.4GB）。

> **环境**：务必在 `conda activate math-cot` 下运行；AutoDL 建议模型缓存放数据盘（`HF_HOME`）。

---

## 报告目录：`predictions/` vs `metrics/`

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

- 换指标或重算 CoT：**只重跑 `run_metrics`**，不必重新 infer
- `reports/` 在 `.gitignore` 中；对外展示可复制 `metrics/*.json` / `.png` 到 `docs/results/`

---

## 金标 Y vs 预测 ŷ

| 符号 | 含义 | 从哪来 |
|------|------|--------|
| **Y (answer)** | 标准答案 | 清洗：`solution` → `answer` |
| **Y (COT)** | 标准思维链 | 清洗：`solution` → `COT` |
| **ŷ (pred_answer)** | 模型预测答案 | split：对 `generated_text` 跑 `extract` |
| **pred_cot** | 模型思维链 | split：`cot_mode: before_answer` 时取 `\boxed` 之前 |

### CoT 格式说明（为何不做逐步 ROSCOE）

- **金标 COT**：多为编号步骤（`1. 2. 3.`）
- **pred_cot**：多为段落式（`Given` / `Next` / `Therefore`），与金标步骤结构不对齐

因此 v1 **不做** step-based faithfulness，改用整段指标：

- **`bert_score`**：pred_cot vs gold `COT`
- **`informativeness_chain`**：pred_cot vs `problem`（ROSCOE 简化）

---

## 指标设计

### 职责拆分

| 文件 | 职责 |
|------|------|
| [`metrics.py`](metrics.py) | 指标类 + `@register_*` |
| [`run_metrics.py`](run_metrics.py) | join → 实例化 → 聚合 → 写 JSON |
| [`report.py`](report.py) | 多 run 对比 JSON + PNG |
| [`roscoe_utils.py`](roscoe_utils.py) | informativeness_chain embedding |
| [`common/registry.py`](../common/registry.py) | 双注册表 |

### 三种指标调用接口

| 接口 | 适用指标 | 输入 |
|------|----------|------|
| `apply(preds, refs)` | `exact_match`, `acc`, `bert_score` | 字符串列表 pred + gold |
| `apply_from_rows(pred_rows)` | `extract_rate` | 读 `extract_ok` |
| `apply_with_context(rows, ...)` | `informativeness_chain` | `pred_cot` + `problem` |

### 结果侧（answer_metrics）

| 指标 | 说明 |
|------|------|
| `exact_match` | 规范化字符串完全一致 |
| `acc` | EvalScope `math_equal` 数学等价 |
| `extract_rate` | `extract_ok` 比例 |

### 思维链侧（cot_metrics）

| 指标 | 比较对象 | 说明 |
|------|----------|------|
| `bert_score` | pred_cot ↔ gold `COT` | DeBERTa token 级语义 F1（见下节） |
| `informativeness_chain` | pred_cot ↔ `problem` | MPNet 整句 cosine，映射到 [0,1] |

---

## bert_score 详解

### 算法概要

对每条样本（pred_cot, gold COT）：

1. DeBERTa tokenizer 分词，**超过 512 token 截断**
2. 分别编码为 token 向量（默认第 9 层 hidden）
3. **双向贪心匹配**：pred 每个 token 在 gold 中找最相似 token → Precision；反向 → Recall
4. 加权（IDF）后算 F1；86 条取平均写入 `{name}_cot.json`

### 局限（必读）

| 局限 | 说明 |
|------|------|
| **512 token 截断** | 长数学 CoT 约 90%+ 样本触顶；主要评**前半段**与金标相似度 |
| **语义 ≠ 正确** | 写法像 gold 不代表推导正确；应结合 `acc` |
| **后半段** | 重复枚举、烂尾、结论常落在截断外，对分数影响有限 |
| **库默认 batch=64** | 长 CoT + GPU 易 OOM；**必须在 yaml + `metrics.py` 传 `batch_size: 1`** |

### 配置（eval_ft.yaml / eval_base.yaml）

```yaml
cot_metrics:
  metric_args:
    bert_score:
      model_type: microsoft/deberta-base-mnli
      batch_size: 1
      device: cuda
      dtype: float32    # 勿用 bfloat16，DeBERTa + autocast 会 overflow
    informativeness_chain:
      embedding_model: all-mpnet-base-v2
```

[`metrics.py`](metrics.py) 中 `BertScoreMetric` 将上述参数传给 `bert_score.score(...)`；`dtype` 非 fp32 时在 CUDA 上用 `torch.autocast`（DeBERTa 仅推荐 fp32）。

### informativeness_chain 补充

- 模型：`all-mpnet-base-v2`（sentence-transformers）
- 每次 encode 一条文本，86 条 × 2（pred + problem），显存 <1GB
- 整句 embedding，约 **384 token** 上限，与 bert_score 截断方式不同

---

## run_metrics 输出 JSON（实测示例）

`reports/metrics/ft_answer.json`：

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

`reports/metrics/ft_cot.json`：

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

### 对比（report.py）

`compare_all.json` 片段（实测）：

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

新增实验：复制 `eval_ft.yaml` → `eval_ft_<name>.yaml`，跑完 metrics 后在 `compare.yaml` → `runs` 追加一项。

---

## 中间产物 schema

### infer：`reports/predictions/{name}.jsonl`

```json
{
  "id": "000732",
  "problem": "...",
  "generated_text": "模型完整生成... \\boxed{(0,4)}",
  "model_path": "outputs/merged/best",
  "model_id": "Qwen/Qwen2.5-Math-7B-Instruct"
}
```

### split：`_answers.jsonl` / `_cots.jsonl`

见 [`split.py`](split.py)；`pred_answer` + `extract_ok` / `pred_cot` + `extract_ok`。

---

## 配置约定

| 文件 | 用途 |
|------|------|
| [`configs/eval_ft.yaml`](../../configs/eval_ft.yaml) | 微调链 |
| [`configs/eval_base.yaml`](../../configs/eval_base.yaml) | 基座链（`model_path: null`，前缀 `base_*`） |
| [`configs/compare.yaml`](../../configs/compare.yaml) | 多 run 对比 |

单 run yaml 四大块：`infer` / `split` / `answer_metrics` / `cot_metrics`。

---

## 模块与依赖

```text
src/eval/
├── infer.py          # GPU 批量推理
├── split.py          # pred_answer / pred_cot
├── metrics.py        # 指标类（含 BertScoreMetric）
├── roscoe_utils.py   # informativeness_chain
├── run_metrics.py    # 调度 + CLI
└── report.py         # 对比 JSON + PNG
```

| 阶段 | 依赖 | 设备 |
|------|------|------|
| infer | train.txt（ms-swift） | GPU |
| split | base.txt | CPU |
| answer metrics | evalscope | CPU |
| cot metrics | bert-score, sentence-transformers | GPU 推荐（bert_score）；MPNet 可 GPU/CPU |
| report | matplotlib | CPU |

### 依赖安装

```bash
pip install -r requirements/eval.txt
```

若 resolver 卡住，分步安装：

```bash
pip install "evalscope==1.8.1" matplotlib
pip install bert-score sentence-transformers
```

首次 CoT 需下载：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

---

## 常见问题（eval 专用）

| 现象 | 处理 |
|------|------|
| bert_score CUDA OOM | `batch_size: 1`；勿依赖库默认 64 |
| `BFloat16 overflow` | `dtype: float32` |
| HF 连接超时 | `HF_ENDPOINT=https://hf-mirror.com`，`HF_HOME` 指数据盘 |
| 离线报 LocalEntryNotFound | 先在线下载 DeBERTa + MPNet，再 `TRANSFORMERS_OFFLINE=1` |
| MPNet 在 bert_score 之后失败 | 两模型独立；MPNet 需单独缓存 |
| infer OOM | `max_batch_size: 1` |
| 只想重算指标 | `run_metrics --task cot`，不 rerun infer |

---

## 扩展指标

1. 在 [`metrics.py`](metrics.py) 实现类，加 `@register_answer_metric` 或 `@register_cot_metric`
2. 在 yaml 的 `metrics:` 列表与 `metric_args` 中注册
3. 重跑 `run_metrics`（无需 infer）

需上下文的 CoT 指标：实现 `apply_with_context`，参考 `InformativenessChainMetric`。

---

## 刻意不做（第一版）

- 不 fork ms-swift `EvalModel` 作为主路径
- 不用 LLM-as-judge 评步骤正确性
- 不做 step-based ROSCOE（pred/gold CoT 结构不对齐）
- 不把 holdout 指标当作模型能力最终结论

---

## 参考

- [ms-swift](https://github.com/modelscope/ms-swift) · [EvalScope](https://github.com/modelscope/evalscope) · [ParlAI ROSCOE](https://github.com/facebookresearch/ParlAI/tree/main/projects/roscoe)
- [README_CN.md](../../README_CN.md) — 端到端流水线、项目定位、快速开始
