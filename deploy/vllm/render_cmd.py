#!/usr/bin/env python3
"""Read configs/serve.yaml and exec `vllm serve`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _add_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def main() -> None:
    config_path = Path(os.environ.get("SERVE_CONFIG", "configs/serve.yaml"))
    if not config_path.is_file():
        sys.exit(f"serve config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # configs/serve.yaml → project root
    root = config_path.resolve().parent.parent
    model_override = os.environ.get("MODEL_PATH", "").strip()

    model_cfg = cfg.get("model", {})
    server_cfg = cfg.get("server", {})
    vllm_cfg = cfg.get("vllm", {})

    if model_override:
        model = model_override
    else:
        raw_path = model_cfg.get("path")
        if not raw_path:
            sys.exit("model.path is required in serve.yaml (or set MODEL_PATH)")
        p = Path(raw_path)
        model = str(p if p.is_absolute() else root / p)

    if not Path(model).exists() and not str(model).startswith(("Qwen/", "meta-llama/", "google/")):
        sys.exit(
            f"model not found: {model}\n"
            "Run: python -m src.train.export  (or set MODEL_PATH=/model in Docker)"
        )

    cmd: list[str] = ["vllm", "serve", model]

    _add_flag(cmd, "--host", server_cfg.get("host"))
    _add_flag(cmd, "--port", server_cfg.get("port"))
    _add_flag(cmd, "--served-model-name", server_cfg.get("served_model_name"))

    api_key = server_cfg.get("api_key")
    if api_key:
        _add_flag(cmd, "--api-key", api_key)

    _add_flag(cmd, "--task", server_cfg.get("task"))

    if vllm_cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    _add_flag(cmd, "--dtype", vllm_cfg.get("dtype"))
    _add_flag(cmd, "--max-model-len", vllm_cfg.get("max_model_len"))
    _add_flag(cmd, "--quantization", vllm_cfg.get("quantization"))
    _add_flag(cmd, "--tensor-parallel-size", vllm_cfg.get("tensor_parallel_size"))
    _add_flag(cmd, "--gpu-memory-utilization", vllm_cfg.get("gpu_memory_utilization"))
    _add_flag(cmd, "--max-num-seqs", vllm_cfg.get("max_num_seqs"))

    if vllm_cfg.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")

    override = vllm_cfg.get("override_generation_config")
    if override:
        cmd.extend(["--override-generation-config", json.dumps(override)])

    chat_template = vllm_cfg.get("chat_template")
    if chat_template:
        ct = Path(chat_template)
        _add_flag(cmd, "--chat-template", str(ct if ct.is_absolute() else root / ct))

    print("Starting:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()