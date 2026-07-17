from __future__ import annotations

"""
读取 train_messages.jsonl + train.yaml，调用 ms-swift 跑 LoRA SFT。

前置：python -m src.train.format
产出：outputs/lora/（checkpoint + images/loss.png）

启动方式（见 configs/train.yaml → runtime.backend）：
  local:     python -m src.train.sft
  smoke:     python -m src.train.sft --smoke
  torchrun:  torchrun --nproc_per_node=N -m src.train.sft
  deepspeed: deepspeed --num_nodes=... --num_gpus=... -m src.train.sft
"""

import argparse
import os
from pathlib import Path
from typing import Any

from src.common.config import load_config

_SUPPORTED_BACKENDS = ("local", "torchrun", "deepspeed")


def _setup_runtime(config: dict[str, Any]) -> str:
    """按 backend 设置运行环境；须在 import torch / ms-swift 之前调用。"""
    runtime = config.get("runtime", {})
    backend = runtime.get("backend", "local")

    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported runtime.backend: {backend}. "
            f"Choose from {_SUPPORTED_BACKENDS}."
        )

    if backend == "local":
        cuda_devices = runtime.get("cuda_visible_devices")
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    elif backend in ("torchrun", "deepspeed"):
        # 多卡/多机：由命令行 launcher 注入 RANK / LOCAL_RANK 等，此处只补默认 master
        dist = config.get("distributed", {})
        os.environ.setdefault("MASTER_ADDR", str(dist.get("master_addr", "127.0.0.1")))
        os.environ.setdefault("MASTER_PORT", str(dist.get("master_port", 29500)))

    return backend


def build_launch_command(config: dict[str, Any]) -> str:
    """根据 config 生成推荐启动命令（供 scripts / 文档使用，不在代码内执行）。"""
    backend = config.get("runtime", {}).get("backend", "local")
    dist = config.get("distributed", {})
    smoke_suffix = " --smoke" if config.get("smoke", {}).get("enabled") else ""

    if backend == "local":
        return f"python -m src.train.sft{smoke_suffix}"

    if backend == "torchrun":
        nproc = dist.get("nproc_per_node", 1)
        nnodes = dist.get("nnodes", 1)
        if nnodes > 1:
            return (
                f"torchrun --nnodes={nnodes} --nproc_per_node={nproc} "
                f"--master_addr={dist.get('master_addr', '127.0.0.1')} "
                f"--master_port={dist.get('master_port', 29500)} "
                f"-m src.train.sft"
            )
        return f"torchrun --nproc_per_node={nproc} -m src.train.sft"

    if backend == "deepspeed":
        nnodes = dist.get("nnodes", 1)
        nproc = dist.get("nproc_per_node", 1)
        return (
            f"deepspeed --num_nodes={nnodes} --num_gpus={nproc} "
            f"-m src.train.sft"
        )

    raise ValueError(f"Unsupported runtime.backend: {backend}")


def _apply_smoke_overrides(config: dict[str, Any]) -> bool:
    """smoke 启用时，用少量 step/样本覆盖 training_args，避免污染正式训练产物。"""
    smoke = config.get("smoke", {})
    if not smoke.get("enabled"):
        return False

    ta = config.setdefault("training_args", {})
    ta["max_steps"] = smoke.get("max_steps", 10)
    ta["save_steps"] = smoke.get("save_steps", 5)
    ta["eval_steps"] = smoke.get("eval_steps", 5)
    ta["logging_steps"] = smoke.get("logging_steps", 1)
    if smoke.get("output_dir"):
        ta["output_dir"] = smoke["output_dir"]
    return True


def _build_training_args(config: dict[str, Any], output_dir: str):
    """把 yaml 的 training_args 映射成 Seq2SeqTrainingArguments。"""
    from swift.trainers import Seq2SeqTrainingArguments

    args_cfg = config["training_args"]
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(args_cfg["learning_rate"]),
        "per_device_train_batch_size": args_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": args_cfg["per_device_eval_batch_size"],
        "gradient_checkpointing": args_cfg.get("gradient_checkpointing", True),
        "weight_decay": float(args_cfg["weight_decay"]),
        "lr_scheduler_type": args_cfg["lr_scheduler_type"],
        "warmup_ratio": float(args_cfg["warmup_ratio"]),
        "gradient_accumulation_steps": args_cfg["gradient_accumulation_steps"],
        "num_train_epochs": args_cfg["num_train_epochs"],
        "save_strategy": args_cfg.get("save_strategy", "steps"),
        "save_steps": args_cfg.get("save_steps", 50),
        "eval_strategy": args_cfg.get("eval_strategy", "steps"),
        "eval_steps": args_cfg.get("eval_steps", 50),
        "save_total_limit": args_cfg.get("save_total_limit", 2),
        "metric_for_best_model": args_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": args_cfg.get("greater_is_better", False),
        "load_best_model_at_end": args_cfg.get("load_best_model_at_end", False),
        "logging_steps": args_cfg.get("logging_steps", 5),
        "logging_first_step": args_cfg.get("logging_first_step", True),
        "report_to": args_cfg.get("report_to", ["tensorboard"]),
        "dataloader_num_workers": args_cfg.get("dataloader_num_workers", 1),
        "data_seed": args_cfg.get("data_seed", 42),
    }

    if args_cfg.get("max_steps") is not None:
        kwargs["max_steps"] = args_cfg["max_steps"]

    ds_cfg = config.get("deepspeed", {})
    if ds_cfg.get("enabled"):
        config_file = ds_cfg.get("config_file")
        if not config_file:
            raise ValueError("deepspeed.enabled=true but deepspeed.config_file is missing")
        kwargs["deepspeed"] = str(config_file)
    else:
        if args_cfg.get("bf16"):
            kwargs["bf16"] = True
        if args_cfg.get("fp16"):
            kwargs["fp16"] = True

    return Seq2SeqTrainingArguments(**kwargs)


def _save_loss_plot(trainer: Any, output_dir: str) -> str | None:
    """
    方案 B：从 trainer.state.log_history 绘制 loss 曲线，保存为 PNG。

    在 trainer.train() 结束后立即调用；数据来自内存中的训练日志，
    不依赖 TensorBoard 网页。产物路径：{output_dir}/images/loss.png
    """
    import matplotlib

    # 云服务器 / 无图形界面环境也能保存图片
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = trainer.state.log_history
    train_points: list[tuple[int, float]] = []
    eval_points: list[tuple[int, float]] = []

    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_points.append((step, entry["loss"]))
        if "eval_loss" in entry:
            eval_points.append((step, entry["eval_loss"]))

    if not train_points:
        return None

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / "loss.png"

    plt.figure(figsize=(8, 5))
    train_steps, train_losses = zip(*train_points)
    plt.plot(train_steps, train_losses, label="train loss")

    if eval_points:
        eval_steps, eval_losses = zip(*eval_points)
        plt.plot(eval_steps, eval_losses, label="eval loss")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return str(out_path)


def run_sft_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """主流程：加载模型 → LoRA → 读 messages 数据 → 训练 → 保存 loss 图。"""
    backend = _setup_runtime(config)
    smoke_enabled = _apply_smoke_overrides(config)
    smoke_cfg = config.get("smoke", {})

    from peft import LoraConfig, get_peft_model
    # ms-swift 3.4+：API 在 swift.llm（旧版在顶层 swift / swift.dataset）
    from swift.llm import (
        EncodePreprocessor,
        get_model_info_meta,
        get_model_tokenizer,
        get_template,
        load_dataset,
    )
    from swift.trainers import Seq2SeqTrainer
    from swift.utils import get_logger, get_model_parameter_info, seed_everything

    data_cfg = config["data"]
    model_cfg = config["model"]
    lora_cfg = config["lora_args"]
    train_args_cfg = config["training_args"]

    seed = train_args_cfg.get("data_seed", 42)
    seed_everything(seed)
    logger = get_logger()

    model_id = model_cfg["model_id"]
    system_prompt = model_cfg["system_prompt"]
    max_length = data_cfg["max_length"]
    messages_path = Path(data_cfg["messages_file"]).resolve()
    output_dir = os.path.abspath(os.path.expanduser(train_args_cfg["output_dir"]))

    if not messages_path.exists():
        raise FileNotFoundError(
            f"Messages file not found: {messages_path}. "
            "Run `python -m src.train.format` first."
        )

    launch_cmd = build_launch_command(config)
    logger.info(f"runtime.backend: {backend}")
    logger.info(f"smoke.enabled: {smoke_enabled}")
    logger.info(f"recommended launch: {launch_cmd}")
    logger.info(f"model_id: {model_id}")
    logger.info(f"dataset: {messages_path}")
    logger.info(f"output_dir: {output_dir}")
    if train_args_cfg.get("bf16"):
        logger.info("training precision: bf16")
    elif train_args_cfg.get("fp16"):
        logger.info("training precision: fp16")
    if train_args_cfg.get("load_best_model_at_end"):
        logger.info(
            f"best checkpoint selection: {train_args_cfg.get('metric_for_best_model', 'eval_loss')}"
        )

    training_args = _build_training_args(config, output_dir)

    # 3.4+：先取 template 类型，再加载 model/tokenizer
    _, model_meta = get_model_info_meta(model_id, download_model=False)
    model, tokenizer = get_model_tokenizer(model_id)
    template = get_template(
        model_meta.template,
        tokenizer,
        default_system=system_prompt,
        max_length=max_length,
    )
    template.set_mode("train")

    target_modules = lora_cfg.get("target_modules")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"lora_config: {lora_config}")
    logger.info(f"model_parameter_info: {get_model_parameter_info(model)}")

    train_dataset, val_dataset = load_dataset(
        [str(messages_path)],
        split_dataset_ratio=data_cfg["split_dataset_ratio"],
        num_proc=data_cfg["num_proc"],
        seed=seed,
    )
    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=data_cfg["num_proc"])
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=data_cfg["num_proc"])

    if smoke_enabled and smoke_cfg.get("max_train_samples"):
        n = int(smoke_cfg["max_train_samples"])
        train_dataset = train_dataset.select(range(min(n, len(train_dataset))))
        val_n = max(4, n // 10)
        val_dataset = val_dataset.select(range(min(val_n, len(val_dataset))))
        logger.info(f"smoke subset: train={len(train_dataset)} val={len(val_dataset)}")

    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        template=template,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    last_checkpoint = trainer.state.last_model_checkpoint
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    logger.info(f"last_model_checkpoint: {last_checkpoint}")
    logger.info(f"best_model_checkpoint: {best_checkpoint}")

    # 训练结束后保存 loss 曲线图（outputs/lora/images/loss.png）
    loss_plot = _save_loss_plot(trainer, output_dir)
    if loss_plot:
        logger.info(f"loss_plot: {loss_plot}")
    else:
        logger.warning("loss_plot skipped: no train loss in log_history")

    return {
        "backend": backend,
        "smoke": smoke_enabled,
        "launch_command": launch_cmd,
        "model_id": model_id,
        "messages_file": str(messages_path),
        "output_dir": output_dir,
        "last_checkpoint": last_checkpoint,
        "best_checkpoint": best_checkpoint,
        "loss_plot": loss_plot,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
    }


def run_from_config(config_path: str | Path, smoke: bool = False) -> dict[str, Any]:
    """从 yaml 加载 config，再调用 run_sft_pipeline。"""
    config = load_config(config_path)
    if smoke:
        config.setdefault("smoke", {})["enabled"] = True
    return run_sft_pipeline(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT via ms-swift")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置文件路径")
    parser.add_argument("--smoke", action="store_true", help="快速冒烟测试（见 train.yaml smoke 块）")
    args = parser.parse_args()

    result = run_from_config(args.config, smoke=args.smoke)
    print("training done:")
    print(f"  backend: {result['backend']}")
    print(f"  smoke: {result['smoke']}")
    print(f"  launch: {result['launch_command']}")
    print(f"  last_checkpoint: {result['last_checkpoint']}")
    print(f"  best_checkpoint: {result['best_checkpoint']}")
    print(f"  loss_plot: {result['loss_plot']}")
    print(f"  output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()


