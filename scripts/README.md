# scripts：薄 Shell 编排层

本目录 **不包含业务逻辑**，只把已有的 `python -m src.*` 命令按阶段串起来，方便本机复现和以后 docker 复用。

| 层 | 位置 | 职责 |
|----|------|------|
| 业务 | `src/` | 数据 / 训练 / 评测实现 |
| 参数 | `configs/` | yaml 配置 |
| 编排 | `scripts/`（本目录） | conda + 顺序执行命令 |

设计原则：

- 每个脚本 5～20 行，出错即停（`set -euo pipefail`）
- 默认从项目根目录执行；脚本内应 `cd` 到仓库根
- 需要改实验时改 **config** 或环境变量 `CONFIG`，不要改 Python 代码

---

## 前置条件

```bash
source /root/miniconda3/etc/profile.d/conda.sh   # 路径按本机调整
conda activate math-cot
cd /root/my_files/math-cot-pipeline                # 项目根
```

依赖安装见根 [README.md](../README.md) 与 [src/eval/README_EN.md](../src/eval/README_EN.md)。

---

## 脚本一览

| 脚本 | 阶段 | GPU | 说明 |
|------|------|-----|------|
| `_common.sh` | — | — | 公共：切根目录、conda（被其他脚本 `source`） |
| `run_data.sh` | 数据 | 否 | `download` → `clean` |
| `run_train.sh` | 训练 | 是 | `format` → `sft` → `export` |
| `run_train_smoke.sh` | 训练 | 是 | `sft --smoke` 冒烟 |
| `run_eval_ft.sh` | 评测 | infer 需 GPU | ft 链：`infer` → `split` → `run_metrics` |
| `run_eval_base.sh` | 评测 | infer 需 GPU | base 链：同上，配置 `eval_base.yaml` |
| `run_report.sh` | 报告 | 否 | 读 `compare.yaml`，产出 JSON + PNG |

> 命名约定：评测微调链脚本为 **`run_eval_ft.sh`**（下划线）。若本地存在 `run.eval_ft.sh`，请重命名为 `run_eval_ft.sh` 以保持一致。

---

## 用法

首次使用建议加可执行权限（也可直接用 `bash scripts/xxx.sh`）：

```bash
chmod +x scripts/*.sh
```

### 数据

```bash
./scripts/run_data.sh
```

等价于：

```bash
python -m src.data.download
python -m src.data.clean
```

### 训练

```bash
./scripts/run_train.sh
```

等价于：`format` → `sft` → `export`（正式 LoRA + merge）。

冒烟：

```bash
./scripts/run_train_smoke.sh
```

等价于：`python -m src.train.sft --smoke`。

分布式训练（torchrun / deepspeed）仍以 `configs/train.yaml` 为准；`run_train.sh` 内应调用与 `src/train/sft.py` 中 `build_launch_command()` 一致的启动方式，**不要在脚本里硬编码** `nproc` 等参数。

### 评测（微调模型）

```bash
./scripts/run_eval_ft.sh
```

默认配置：`configs/eval_ft.yaml`。若 predictions 已存在，可在脚本里跳过 `infer`，只跑 `split` + `run_metrics`。

试跑 infer（参数透传给 Python）：

```bash
./scripts/run_eval_ft.sh --limit 4
```

指定其它 eval 配置：

```bash
CONFIG=configs/eval_ft_v2.yaml ./scripts/run_eval_ft.sh
```

### 评测（基座模型）

```bash
./scripts/run_eval_base.sh
```

默认配置：`configs/eval_base.yaml`（Hub 权重，`model_path: null`）。

### 多 run 对比报告

各 run 的 `reports/metrics/*_answer.json` 与 `*_cot.json` 就绪后：

```bash
./scripts/run_report.sh
./scripts/run_report.sh --no-plot    # 只写 JSON，不出 PNG
```

默认读 `configs/compare.yaml`，产出：

- `reports/metrics/compare_all.json`
- `reports/metrics/compare_all.png`（除非 `plot_file: null` 或 `--no-plot`）

---

## 推荐执行顺序

```text
run_data.sh
    ↓
run_train.sh          （或 run_train_smoke.sh 试跑）
    ↓
run_eval_ft.sh
run_eval_base.sh      （可在另一台 GPU 机器）
    ↓
run_report.sh
```

 deliberately **没有** `run_all.sh`：各阶段耗时长、对 GPU 要求不同，分开跑更稳。

---

## `_common.sh` 约定（实现参考）

其它脚本开头建议：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"
```

`_common.sh` 内可集中处理：

- `conda activate math-cot`（或检测 `$CONDA_DEFAULT_ENV`）
- 可选：`export HF_ENDPOINT`、`MODELSCOPE_CACHE` 等

---

## 与 docker 的关系

未来 `docker-compose` 的 service 可直接调用同一套脚本，例如：

```yaml
command: ["./scripts/run_eval_ft.sh"]
```

避免在 Dockerfile 和 shell 里各写一遍命令。

---

## 相关文档

- 根 [README.md](../README.md)：端到端流程
- [src/eval/README_EN.md](../src/eval/README_EN.md) — eval details, metrics, compare config (中文: [README_CN.md](../src/eval/README_CN.md))
- 配置：[configs/eval_ft.yaml](../configs/eval_ft.yaml)、[eval_base.yaml](../configs/eval_base.yaml)、[compare.yaml](../configs/compare.yaml)
