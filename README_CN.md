# Math CoT Pipeline

[English](README_EN.md) | **中文（完整）**

端到端数学推理流水线：**公开数据清洗 → ms-swift LoRA 微调 → 答案 + 思维链双轨评测 → 多 run 对比报告 → 可选 vLLM 部署**。

默认基座：`Qwen/Qwen2.5-Math-7B-Instruct`。训练目标是让模型输出完整 Chain-of-Thought，并以可解析的最终答案（如 `\boxed{}`）收尾。

---

## 项目定位

**本仓库的首要目标是搭建可复现的评测体系与端到端通道**，而不是在特定 holdout 上刷 SOTA 或系统调优 LoRA 超参。

我们更想展示的是：

- 数据如何从 NuminaMath-CoT 清洗为统一 schema，并划分 train / holdout test
- 如何用 ms-swift 完成 LoRA SFT、merge 与批量 infer
- 如何将生成结果落盘，并拆分为 **答案轨** 与 **思维链轨** 分别评测
- 如何通过 `registry` 注册指标、重算 metrics 而无需重跑 infer
- 如何汇总 base vs fine-tuned 等多 run 对比（JSON + 柱状图）

因此，`reports/metrics/` 中的数值应视为 **pipeline 冒烟与相对比较**（ft vs base），不宜单独作为「模型能力很强/很弱」的充分依据。详见下文 [训练曲线（参考）](#训练曲线参考) 与 [参考实验结果](#参考实验结果) 及 [指标说明与局限](#指标说明与局限)。

---

## 训练曲线（参考）

默认配置下（约 **778 train**，单次 LoRA SFT、1 epoch / 44 steps，按 `eval_loss` 选 best checkpoint，未做系统超参搜索），训练 loss 曲线如下：

![LoRA SFT 训练 loss](docs/results/training_loss.png)

产物路径：

- 仓库快照（随 git 提交）：`docs/results/training_loss.png`
- 本地复现输出：`outputs/lora/images/loss.png`（`./scripts/run_train.sh` 结束后由 `src/train/sft.py` 生成）

> 训练曲线仅作 **pipeline 可跑通与收敛参考**；最终效果以 holdout 评测（见下节）为准。

---

## 参考实验结果

在默认配置下（约 **778 train / 86 holdout test**，单次 LoRA SFT，未做系统超参搜索），完整跑通 base 与 `outputs/merged/best` 微调模型后，对比报告（`baseline: base`）如下：

| 指标 | base | ft_best | Δ (ft − base) | 说明 |
|------|------|---------|---------------|------|
| **acc** | 20.9% | 27.9% | **+7.0%** | 数学等价准确率（`math_equal`） |
| **exact_match** | 15.1% | 17.4% | +2.3% | 字符串精确匹配 |
| **extract_rate** | 75.6% | 70.9% | −4.7% | 能否从生成中抽出 `\boxed` 答案 |
| **bert_score** | 0.738 | 0.742 | +0.004 | pred CoT vs 金标 COT 语义 F1 |
| **informativeness_chain** | 0.902 | 0.900 | ≈ 0 | pred CoT vs 题目的相关性 |

对比柱状图（`baseline: base`）：

![base vs ft 指标对比](docs/results/compare_all.png)

产物路径：

- 仓库快照（随 git 提交）：`docs/results/compare_all.png`
- 本地复现输出：`reports/metrics/compare_all.json`、`reports/metrics/compare_all.png`

复现对比报告（metrics JSON 已就绪时）：

```bash
python -m src.eval.report --config configs/compare.yaml
```

---

## 项目亮点

- **数据**：基于 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)，统一 schema、去重、从 `solution` 抽取金标 `answer` 与 `COT`；**train / test 在 `clean` 阶段一次划分**（val 由 ms-swift 训练时从 train 切分）
- **微调**：ms-swift SFT + LoRA；chat `messages` 格式；bf16、best checkpoint、smoke 冒烟
- **评测**：自有 holdout `test.jsonl`：**infer → split → run_metrics → report**；答案侧 EM / acc，思维链侧 bert_score / informativeness_chain；`predictions/` 落盘可重算
- **部署**：vLLM（`deploy/vllm/` + `configs/serve.yaml`），根目录 `docker-compose.yml`（`serve` profile）

---

## 流程概览

```text
NuminaMath-CoT
  → download（HF → raw/*.jsonl）
  → clean（清洗 / 去重 / 抽 answer+COT / 划分 train+test）
  → train.jsonl / test.jsonl
                              ↓
        format（可选）→ train_messages.jsonl
                              ↓
                   ms-swift LoRA SFT → outputs/lora
                              ↓
              ┌───────────────┴────────────────┐
              ↓                                ↓
        Export merged                      Batch Infer (eval)
        → outputs/merged/best              → predictions/*.jsonl
              ↓                                ↓
         deploy/vllm (vLLM API)            Split ŷ | pred_cot
                                               ├─ Answer → EM / acc / extract_rate
                                               └─ CoT    → bert_score / informativeness_chain
                                               ↓
                                         run_metrics → metrics/*.json
                                               ↓
                                         report → compare_all.json + .png
```

公开 benchmark（GSM8K 等）如需 EvalScope / `swift eval`，可作为可选扩展，**不替代**自有 holdout test 主路径。

---

## 目录结构

```text
math-cot-pipeline/
├── configs/
│   ├── data.yaml          # 数据源、清洗、holdout test
│   ├── train.yaml         # ms-swift / LoRA / 训练 / export
│   ├── eval_ft.yaml       # 微调模型评测链
│   ├── eval_base.yaml     # 基座模型评测链
│   ├── compare.yaml       # 多 run 对比报告
│   └── serve.yaml         # vLLM 部署
├── data/raw/              # 原始下载
├── data/processed/        # train.jsonl / test.jsonl / train_messages.jsonl
├── src/
│   ├── common/            # config / io / extract / registry
│   ├── data/              # download / clean
│   ├── train/             # format / sft / export
│   └── eval/              # infer / split / run_metrics / report
├── scripts/               # run_data / run_train / run_eval_* / run_report
├── tests/
├── docker/
├── deploy/vllm/
├── outputs/               # lora / merged（gitignore 建议）
├── reports/               # predictions/ + metrics/（gitignore）
├── requirements/
├── README_CN.md
└── README_EN.md
```

---

## 快速开始

> **GPU**：训练、export、infer 建议 NVIDIA ≥24GB（32GB 更宽裕）。split、run_metrics、report 可在 CPU 上跑；CoT 的 bert_score 建议 GPU + 小 batch。

### 1. 环境

```bash
conda create -n math-cot python=3.11 -y
source /root/miniconda3/etc/profile.d/conda.sh   # 新 shell 需 source，再 activate
conda activate math-cot

pip install -r requirements/base.txt

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/train.txt
pip install -r requirements/eval.txt
```

**AutoDL / 国内镜像建议：**

```bash
export HF_HOME=/root/autodl-tmp/hf              # 大模型缓存放数据盘
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope
```

`evalscope` 依赖较重；若 `pip install -r requirements/eval.txt` 卡住，可分步安装，见 [`src/eval/README_EN.md`](src/eval/README_EN.md)。

**脚本一键编排（等价于分步 `python -m`）：**

```bash
./scripts/run_data.sh
./scripts/run_train.sh
./scripts/run_eval_ft.sh
./scripts/run_eval_base.sh
./scripts/run_report.sh
```

**单元测试：**

```bash
pip install -r requirements/dev.txt
pytest tests/ -v
```

### 2. 数据准备

```bash
cd math-cot-pipeline
python -m src.data.download
python -m src.data.clean
```

| 产出 | 说明 |
|------|------|
| `data/raw/numina_math_cot.jsonl` | 原始下载 |
| `data/processed/train.jsonl` | 微调集 |
| `data/processed/test.jsonl` | holdout 评测集（不参与训练） |
| `data/processed/manifest.json` | 清洗统计 |

默认配置示例：1000 raw → 864 保留 → **train 778 / test 86**。

**processed 样本字段：** `id`、`source`、`problem`、`solution`、`COT`（`\boxed` 前）、`answer`（金标 Y）。

### 3. 微调

```bash
python -m src.train.format          # → train_messages.jsonl
python -m src.train.sft --smoke     # 冒烟 → outputs/lora-smoke
python -m src.train.sft             # 正式 → outputs/lora
python -m src.train.export          # merge → outputs/merged/best
```

| 步骤 | 模块 | 产出 |
|------|------|------|
| format | `src/train/format.py` | `train_messages.jsonl` |
| sft | `src/train/sft.py` | `outputs/lora/checkpoint-*` |
| export | `src/train/export.py` | `outputs/merged/best/` |

训练要点（`configs/train.yaml`）：bf16、`load_best_model_at_end`、有效 batch=16（1×16 grad accum）。小数据集建议 `save_steps` / `eval_steps` = 10。

### 4. 评测

完整设计见 [`src/eval/README_EN.md`](src/eval/README_EN.md)。

#### 单 run（ft / base 各一条链）

```bash
# 微调模型（outputs/merged/best）
python -m src.eval.infer       --config configs/eval_ft.yaml
python -m src.eval.infer       --config configs/eval_ft.yaml --limit 4   # 试跑
python -m src.eval.split       --config configs/eval_ft.yaml
python -m src.eval.run_metrics --config configs/eval_ft.yaml --task all

# 基座模型（model_path: null，Hub 权重）
python -m src.eval.infer       --config configs/eval_base.yaml
python -m src.eval.split       --config configs/eval_base.yaml
python -m src.eval.run_metrics --config configs/eval_base.yaml --task all
```

`--task` 可选 `answer` | `cot` | `all`。CoT 指标依赖 `bert-score`、`sentence-transformers`，首次会下载 DeBERTa 与 MPNet。

**CoT 指标推荐配置**（`eval_ft.yaml` / `eval_base.yaml` → `cot_metrics.metric_args.bert_score`）：

```yaml
bert_score:
  model_type: microsoft/deberta-base-mnli
  batch_size: 1      # 长 CoT 勿用库默认 64，易 OOM
  device: cuda
  dtype: float32     # DeBERTa 与 bf16 autocast 不兼容，请用 fp32
informativeness_chain:
  embedding_model: all-mpnet-base-v2
```

#### 多 run 对比

```bash
python -m src.eval.report --config configs/compare.yaml
# → reports/metrics/compare_all.json + compare_all.png
```

#### 流水线产物

```text
test.jsonl
  → infer.py        → reports/predictions/{name}.jsonl
  → split.py        → reports/predictions/{name}_answers.jsonl / _cots.jsonl
  → run_metrics.py  → reports/metrics/{name}_answer.json / _cot.json
  → report.py       → compare_all.json + .png
```

**换指标或重算 CoT**：只需重跑 `run_metrics`，**不必重新 infer**。

### 5. 部署（vLLM）

```bash
python -m src.train.export   # 若尚未 merge
pip install -r requirements/serve.txt vllm
bash deploy/vllm/start.sh
```

Docker：`docker compose --profile serve up -d vllm-api`。评测 infer 仍走 `src/eval/infer.py`（PtEngine），与 vLLM API 职责分离。详见 [`deploy/vllm/README.md`](deploy/vllm/README.md)。

---

## 指标说明与局限

### 答案轨

| 指标 | 含义 |
|------|------|
| `exact_match` | 预测答案与金标字符串归一化后是否完全一致 |
| `acc` | 数学等价（EvalScope `math_equal`，更宽松） |
| `extract_rate` | 是否成功从生成文本中解析出 `\boxed` 等格式答案 |

### 思维链轨

| 指标 | 比较对象 | 含义 |
|------|----------|------|
| `bert_score` | pred_cot ↔ 金标 `COT` | DeBERTa token 级语义 F1（贪心匹配） |
| `informativeness_chain` | pred_cot ↔ `problem` | MPNet 整句 embedding 余弦相似度（ROSCOE 简化） |

**重要局限：**

1. **`bert_score` 基于 DeBERTa 512 token 截断**  
   长数学 CoT 往往超过 512 token，指标主要反映**推理前半段**与金标的语义接近程度，**不能**等价于推导正确性或整段 CoT 质量。

2. **`informativeness_chain` 使用 MPNet**，整句 embedding 亦有上下文长度上限（约 384 token），与 bert_score 截断方式不同。

3. **CoT 指标高 ≠ 答案对**；**acc 高 ≠ CoT 与金标写法一致**。应结合答案轨与 CoT 轨一起看，并理解为 pipeline 验证信号。

4. 本项目的 **LoRA 数据规模与训练轮次有限**，未针对 holdout 做深度调参；数值用于 **ft vs base 趋势** 即可。

更细的评测架构说明见 [`src/eval/README_EN.md`](src/eval/README_EN.md)。

---

## Docker 一体编排

| Profile | 典型服务 | GPU |
|---------|----------|-----|
| `cpu` | `test`、`data`、`report`、`eval-ft-metrics` | 否 |
| `repro` | `train`、`export`、`eval-ft`、`eval-base` | 是 |
| `full` | 上述合集 | 视服务而定 |
| `serve` | `vllm-api` | 是 |

```bash
docker compose build
docker compose --profile cpu run --rm test
docker compose --profile cpu run --rm data
docker compose --profile repro run --rm eval-ft
docker compose --profile serve up -d vllm-api
```

单卡请勿同时跑 infer 与 vLLM。本机不用 Docker 时见 [`scripts/README.md`](scripts/README.md)。

---

## 配置说明

| 文件 | 用途 |
|------|------|
| `configs/data.yaml` | 数据源、清洗、holdout test |
| `configs/train.yaml` | 模型、LoRA、训练、export、smoke |
| `configs/eval_ft.yaml` | 微调模型 infer / split / metrics |
| `configs/eval_base.yaml` | 基座模型，路径前缀 `base_*` |
| `configs/compare.yaml` | 多 run 对比 report |
| `configs/serve.yaml` | vLLM merged 路径与端口 |

`eval_ft.yaml` 中有两个不同的 `model_type`：`infer.model_type` 为 ms-swift 加载 Qwen；`cot_metrics...bert_score.model_type` 为 DeBERTa 裁判模型，勿混淆。

---

## 训练样本格式

```json
{
  "messages": [
    {"role": "user", "content": "<problem>"},
    {"role": "assistant", "content": "<solution>"}
  ]
}
```

`src/train/format.py` 按 `train.yaml` → `train_schema` 从 `train.jsonl` 生成 `train_messages.jsonl`。

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `conda activate` 报错 | 先 `source .../conda.sh` 再 activate |
| HuggingFace 连接超时 | `export HF_ENDPOINT=https://hf-mirror.com`；缓存用 `HF_HOME` 指向数据盘 |
| 模型已下载仍尝试联网 | 缓存完整时 `export TRANSFORMERS_OFFLINE=1` |
| CoT `bert_score` CUDA OOM | yaml 设 `batch_size: 1`；勿用库默认 64 |
| bf16 报 `BFloat16 overflow` | `dtype: float32`，DeBERTa 勿开 autocast bf16 |
| infer OOM | `max_batch_size: 1`，或降低 `max_new_tokens` |
| MPNet 找不到 | 先 `SentenceTransformer('all-mpnet-base-v2')` 下载到 `HF_HOME` |

---

## 技术栈

Python 3.11 · **ms-swift** · Hugging Face · PyTorch bf16 · **EvalScope**（`math_equal`）· **bert-score** · **sentence-transformers** · **matplotlib** · **vLLM** · **Docker Compose**

---

## 开发状态

| 模块 | 状态 |
|------|------|
| `src/common/`、`src/data/`、`src/train/`、`src/eval/` | ✅ |
| `configs/*.yaml`、`scripts/`、`tests/`（27 用例） | ✅ |
| `deploy/vllm/`、`docker/` + `docker-compose.yml` | ✅ |
| holdout 评测链（base + ft + compare 报告） | ✅ 默认配置下已跑通 |
| `.dockerignore` | 🔲 可选 |

---

## License

MIT
