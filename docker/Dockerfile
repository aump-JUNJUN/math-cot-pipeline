# math-cot-pipeline：训练 + 评测 pipeline 镜像
# 基座：PyTorch 官方 CUDA 12.4（与 README cu124 一致）
# vLLM 不在此镜像内，见 docker-compose.yml → vllm-api

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="math-cot-pipeline"
LABEL org.opencontainers.image.description="Data / train / eval / report pipeline"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SKIP_CONDA=1 \
    WORKSPACE=/workspace

WORKDIR /workspace

# 部分 pip 包（ms-swift / datasets）可能需要 git
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先 COPY 依赖清单，利用 Docker 层缓存
COPY requirements/base.txt requirements/train.txt requirements/eval.txt requirements/

# 基座镜像已带 torch；直接装 train + eval 栈
RUN pip install --upgrade pip \
 && pip install -r requirements/train.txt \
 && pip install -r requirements/eval.txt

# 单元测试（compose 里 test service 用）
COPY requirements/dev.txt requirements/
RUN pip install -r requirements/dev.txt

# 启动脚本（compose 也会 volume 挂载整仓，这里 COPY 保证镜像可独立跑）
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 源码默认靠 volume 挂载；build 时不 COPY 全仓，加快迭代
# 若需要「无挂载单镜像」，取消下一行注释：
# COPY . .

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]