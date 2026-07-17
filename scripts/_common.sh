# scripts/_common.sh
# 被其它脚本 source，不要直接 bash _common.sh

# 1. 项目根：从 scripts/ 往上一级
_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"
cd "$ROOT"


# 2. conda（Docker 容器内设 SKIP_CONDA=1 跳过，见 docker/entrypoint.sh）
if [[ "${SKIP_CONDA:-}" == "1" ]]; then
  :
elif [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-math-cot}" ]]; then
  CONDA_ENV="${CONDA_ENV:-math-cot}"
  _CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
  if [[ -f "$_CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$_CONDA_SH"
    conda activate "$CONDA_ENV"
  else
    echo "warning: conda not found at $_CONDA_SH; using current python" >&2
  fi
fi

# 3. 可选：国内镜像 / 大模型缓存（有数据盘再开）
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
# export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope}"

