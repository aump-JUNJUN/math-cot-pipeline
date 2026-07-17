# vLLM 部署（方案 B）

独立 vLLM 服务，加载 merge 后的完整权重，提供 OpenAI 兼容 API。

与评测链的关系：

- **评测**：`src/eval/infer.py`（ms-swift PtEngine）→ predictions → metrics → report
- **部署**：本目录 → vLLM HTTP API（给人/程序在线调用）

参数来源：`configs/serve.yaml`（与 `eval_ft.yaml` 对齐 temperature=0、max_tokens=1024）。

## 前置

```bash
# 1. merge 权重
python -m src.train.export

# 2. 确认路径与 serve.yaml 一致
ls outputs/merged/best/config.json