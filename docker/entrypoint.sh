#!/usr/bin/env bash
# pipeline 容器入口（Dockerfile ENTRYPOINT 指向本脚本）
#
# 每次 docker compose run/up 启动 pipeline 容器时：
#   1. cd 到项目根 /workspace
#   2. 设 SKIP_CONDA=1（容器内无 conda，见 scripts/_common.sh）
#   3. exec 执行 compose 传入的 command（如 bash scripts/run_data.sh）
#
# 不负责 pip 安装；依赖在 Dockerfile 的 RUN 阶段完成。

set -euo pipefail

cd "${WORKSPACE:-/workspace}"

export SKIP_CONDA="${SKIP_CONDA:-1}"

# 可选：compose 环境变量可覆盖
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

exec "$@"
