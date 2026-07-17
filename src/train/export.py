from __future__ import annotations

"""
将 LoRA adapter 合并进基座模型，产出完整权重供 vLLM 部署使用。

输入：configs/train.yaml（export 块）+ sft 产出的 checkpoint
输出：outputs/merged/（完整 HuggingFace 模型目录）

前置：python -m src.train.sft

启动：
  python -m src.train.export
  python -m src.train.export --config configs/train.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Any

from src.common.config import load_config


def _setup_runtime(config: dict[str, Any]) -> None:
    """export 可能占用较多显存；local 模式下沿用 train.yaml 的 GPU 设置。"""
    runtime = config.get("runtime", {})
    if runtime.get("backend", "local") != "local":
        return
    cuda_devices = runtime.get("cuda_visible_devices")
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)


def _normalize_checkpoint_name(name: str) -> str:
    """将 100 / checkpoint-100 统一为 checkpoint-100；best/latest 保持原样。"""
    name = str(name).strip()
    if name.lower() in ("best", "latest"):
        return name.lower()
    if name.isdigit():
        return f"checkpoint-{name}"
    if not name.startswith("checkpoint-"):
        return f"checkpoint-{name}"
    return name


def _read_best_checkpoint(adapter_root: Path) -> Path:
    """从 trainer_state.json 读取 best_model_checkpoint 路径。"""
    import json

    state_file = adapter_root / "trainer_state.json"
    if not state_file.exists():
        # ms-swift / HF Trainer 只在 checkpoint-* 子目录写 trainer_state.json
        checkpoints = sorted(
            adapter_root.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", maxsplit=1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"trainer_state.json not found under {adapter_root} "
                "and no checkpoint-* directories exist. "
                "Run `python -m src.train.sft` with load_best_model_at_end first."
            )
        state_file = checkpoints[-1] / "trainer_state.json"
        if not state_file.exists():
            raise FileNotFoundError(
                f"trainer_state.json not found under {adapter_root} or latest checkpoint. "
                "Run `python -m src.train.sft` with load_best_model_at_end first."
            )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    best = state.get("best_model_checkpoint")
    if not best:
        raise FileNotFoundError(
            f"best_model_checkpoint missing in {state_file}. "
            "Ensure eval ran during training and load_best_model_at_end is enabled."
        )

    best_path = Path(str(best)).resolve()
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint path not found: {best_path}. "
            "It may have been removed by save_total_limit."
        )
    return best_path


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or path.name.startswith("checkpoint-")


def _resolve_adapter_path(adapter_path: Path, checkpoint: str | None = None) -> Path:
    """
    解析 adapter 路径：
    - checkpoint: best → 读 trainer_state.json
    - checkpoint: latest / 未指定 → 取最新 checkpoint-*
    - checkpoint: checkpoint-100 → 指定目录
    - adapter_dir 已是 checkpoint 目录且未指定 checkpoint → 直接使用
    """
    adapter_path = adapter_path.resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    adapter_root = adapter_path.parent if _is_checkpoint_dir(adapter_path) else adapter_path
    ckpt_key = _normalize_checkpoint_name(checkpoint) if checkpoint else None

    if ckpt_key == "best":
        return _read_best_checkpoint(adapter_root)

    if ckpt_key == "latest" or ckpt_key is None:
        if _is_checkpoint_dir(adapter_path) and ckpt_key is None:
            return adapter_path
        checkpoints = sorted(
            adapter_root.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", maxsplit=1)[-1]),
        )
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(
            f"No checkpoint-* under {adapter_root}. Run `python -m src.train.sft` first."
        )

    target = adapter_root / ckpt_key
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Adapter path not found: {target}")
    if (target / "adapter_config.json").exists():
        return target

    raise FileNotFoundError(
        f"No adapter_config.json under {target}. Run `python -m src.train.sft` first."
    )


def _run_ms_swift_merge(
    model: str,
    adapters: str,
    output_dir: str,
    torch_dtype: str | None = None,
) -> None:
    """调用 ms-swift export，将 LoRA 权重 merge 进基座并写入 output_dir。"""
    try:
        from swift.llm import ExportArguments, export_main
    except ImportError:
        from swift import ExportArguments, export_main

    kwargs: dict[str, Any] = {
        "model": model,
        "adapters": adapters,
        "merge_lora": True,
        "output_dir": output_dir,
        "exist_ok": True,
    }
    if torch_dtype:
        kwargs["torch_dtype"] = torch_dtype

    export_main(ExportArguments(**kwargs))


def run_export_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """主流程：定位 checkpoint → merge LoRA → 写入 merged 目录。"""
    _setup_runtime(config)

    export_cfg = config.get("export", {})
    train_output = config.get("training_args", {}).get("output_dir", "outputs/lora")

    adapter_input = Path(export_cfg.get("adapter_dir", train_output))
    checkpoint = export_cfg.get("checkpoint")
    merged_dir = Path(export_cfg.get("merged_dir", "outputs/merged")).resolve()
    torch_dtype = export_cfg.get("torch_dtype")

    adapter_path = _resolve_adapter_path(adapter_input, checkpoint=checkpoint)
    #merged_dir.mkdir(parents=True, exist_ok=True)

    model_id = config.get("model", {}).get("model_id")
    if not model_id:
        raise ValueError("model.model_id is required in config for export.")

    _run_ms_swift_merge(
        model=model_id,
        adapters=str(adapter_path),
        output_dir=str(merged_dir),
        torch_dtype=torch_dtype,
    )

    return {
        "adapter_path": str(adapter_path),
        "checkpoint": checkpoint,
        "merged_dir": str(merged_dir),
        "base_model": config.get("model", {}).get("model_id"),
        "torch_dtype": torch_dtype,
    }


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    """从 yaml 加载 config，再调用 run_export_pipeline。"""
    config = load_config(config_path)
    return run_export_pipeline(config)


"""   CLI 入口  """

def main() -> None:
    """CLI 入口：python -m src.train.export"""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model weights")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置文件路径")
    args = parser.parse_args()

    result = run_from_config(args.config)
    print("export done:")
    print(f"  adapter: {result['adapter_path']}")
    if result.get("checkpoint"):
        print(f"  checkpoint: {result['checkpoint']}")
    print(f"  merged_dir: {result['merged_dir']}")
    print(f"  base_model: {result['base_model']}")
    if result.get("torch_dtype"):
        print(f"  torch_dtype: {result['torch_dtype']}")


if __name__ == "__main__":
    main()

