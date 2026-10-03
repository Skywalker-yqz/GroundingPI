#!/usr/bin/env python3
"""Validate an GroundingPI GAM YAML and emit the exact Swift CLI argument vector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from train.release_parameters import validate_training_fields


GAM_ROOT = Path(__file__).resolve().parents[2]
SWIFT_ROOT = GAM_ROOT / "vendor/swift"
TRANSFORMERS_ROOT = GAM_ROOT / "vendor/transformers"
EXPECTED_RUNTIME_SHA256 = {
    "configuration_groundingpi.py": "feb2c7822fbc4e26ef0793c31bf38407e5cc019f922b867fdeef7c4e17da84f1",
    "modeling_groundingpi.py": "c89476c7969ce2115ddb5fa3cace84090ebd2f24216f43c73d4eca51a42bb28b",
    "configuration_groundingpi_vision.py": "5271ee5625322634fd181d9b4cd47249abf7b97c844baf57544a557f49bffd5d",
    "modeling_groundingpi_vision.py": "fb1243c45e2be50517c867e68bc2eeee7d7cb4341f9846db2f50105ee9d95dd2",
    "processing_groundingpi.py": "fada769df58fa0d4efd9d4ec45688f6cf01e0c5433539fa9cc888f5615db5fe4",
    "image_processing_groundingpi.py": "0b81c45dbe91473c4370ad8a4f95e56defb92bbcff35f22f85518937da061db7",
}
# SHA256 of the bundled training plugin.
EXPECTED_PLUGIN_SHA256 = "6d6ec32de63a6b52821fc226071f3ab4c65a3f9e16553bd0409ddc075d6070fe"


def fail(message: str) -> None:
    raise SystemExit(f"FATAL: {message}")


def load_config(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail("YAML 顶层必须为 mapping")
    for key in ("meta", "model", "training", "runtime", "datasets"):
        if key not in obj:
            fail(f"YAML 缺少 {key}")
    validate_training_fields(obj)
    return obj


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plugin(path: Path) -> None:
    if sha256(path) != EXPECTED_PLUGIN_SHA256:
        fail("GroundingPI Swift plugin hash 漂移")


def validate(path: Path, *, expected_world_size=None, batch_size_override=None, packing_length_override=None):
    """Validate the final YAML and its bound model/data/dependency resources."""
    if batch_size_override is not None or packing_length_override is not None:
        fail("Edit batch size and packing length in the training YAML before validation")
    from train.vlm.portable_validation import validate as validate_release
    return validate_release(path, expected_world_size=expected_world_size)


def as_cli(
    path: Path,
    sample_per_input: int | None,
    batch_size_override: int | None = None,
    packing_length_override: int | None = None,
    max_steps_override: int | None = None,
) -> list[str]:
    obj = load_config(path)
    model = obj["model"]
    t = obj["training"]
    runtime = obj["runtime"]
    datasets: list[str] = []
    for item in obj["datasets"]:
        dataset_path = item["path"]
        if sample_per_input is not None:
            dataset_path = f"{dataset_path}#{sample_per_input}"
        elif item.get("sample_count") is not None:
            dataset_path = f"{dataset_path}#{int(item['sample_count'])}"
        datasets.extend([dataset_path] * int(item["repeat"]))

    args = [
        "--model", model["path"],
        "--model_type", model["type"],
        "--template", model["type"],
        "--external_plugins", runtime["external_plugin"],
        "--cached_dataset", *datasets,
        "--output_dir", model["output_dir"],
        "--add_version", "true",
        "--tuner_type", t["tuner_type"],
        "--torch_dtype", t["torch_dtype"],
        "--bf16", "true",
        "--attn_impl", t["attn_impl"],
        "--freeze_vit", str(t["freeze_vit"]).lower(),
        "--freeze_aligner", str(t["freeze_aligner"]).lower(),
        "--freeze_llm", str(t["freeze_llm"]).lower(),
        "--max_length", str(t["max_length"]),
        "--truncation_strategy", t["truncation_strategy"],
        "--packing", str(t["packing"]).lower(),
        "--packing_length", str(
            packing_length_override
            if packing_length_override is not None
            else t["packing_length"]
        ),
        "--num_train_epochs", str(t["num_train_epochs"]),
        "--per_device_train_batch_size", str(
            batch_size_override
            if batch_size_override is not None
            else t["per_device_train_batch_size"]
        ),
        "--gradient_accumulation_steps", str(t["gradient_accumulation_steps"]),
        "--learning_rate", str(t["learning_rate"]),
        "--vit_lr", str(t["vit_lr"]),
        "--aligner_lr", str(t["aligner_lr"]),
        "--warmup_ratio", str(t["warmup_ratio"]),
        "--lr_scheduler_type", str(t["lr_scheduler_type"]),
        "--lr_scheduler_kwargs", str(t["lr_scheduler_kwargs"]),
        "--weight_decay", str(t["weight_decay"]),
        "--max_grad_norm", str(t["max_grad_norm"]),
        "--gradient_checkpointing", str(t["gradient_checkpointing"]).lower(),
        "--vit_gradient_checkpointing", str(t["vit_gradient_checkpointing"]).lower(),
        "--optimizer", t["optimizer"],
        "--loss_scale", t["loss_scale"],
        "--add_non_thinking_prefix", str(t["add_non_thinking_prefix"]).lower(),
        "--use_logits_to_keep", str(t["use_logits_to_keep"]).lower(),
        "--save_strategy", str(t["save_strategy"]),
        "--save_only_model", str(t["save_only_model"]).lower(),
        "--logging_steps", str(t["logging_steps"]),
        "--deepspeed", str(t["deepspeed"]),
        "--dataset_num_proc", str(t["dataset_num_proc"]),
        "--packing_num_proc", str(t["packing_num_proc"]),
        "--dataloader_num_workers", str(t["dataloader_num_workers"]),
        "--split_dataset_ratio", str(t["split_dataset_ratio"]),
        "--dataset_shuffle", str(t.get("dataset_shuffle", True)).lower(),
        "--load_from_cache_file", str(t.get("load_from_cache_file", True)).lower(),
        "--seed", str(t["seed"]),
        "--data_seed", str(t["data_seed"]),
        "--ddp_timeout", str(t["ddp_timeout"]),
        "--report_to", str(t["report_to"]),
    ]
    for key in ("max_steps", "save_steps", "resume_from_checkpoint"):
        if key in t:
            args.extend([f"--{key}", str(t[key])])
    if max_steps_override is not None:
        if max_steps_override < 1:
            fail("max_steps_override 必须为正整数")
        if "max_steps" in t:
            pos = args.index("--max_steps")
            args[pos + 1] = str(max_steps_override)
        else:
            args.extend(["--max_steps", str(max_steps_override)])
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("config", type=Path)
    check.add_argument("--expected-world-size", type=int)
    check.add_argument("--batch-size-override", type=int)
    check.add_argument("--packing-length-override", type=int)
    check.add_argument("--report", type=Path)
    emit = sub.add_parser("args0")
    emit.add_argument("config", type=Path)
    emit.add_argument("--sample-per-input", type=int)
    emit.add_argument("--batch-size-override", type=int)
    emit.add_argument("--packing-length-override", type=int)
    emit.add_argument("--max-steps-override", type=int)
    ns = parser.parse_args()
    if ns.command == "validate":
        report = validate(
            ns.config,
            expected_world_size=ns.expected_world_size,
            batch_size_override=ns.batch_size_override,
            packing_length_override=ns.packing_length_override,
        )
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if ns.report:
            ns.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = ns.report.with_name(f".{ns.report.name}.tmp.{os.getpid()}")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, ns.report)
        sys.stdout.write(encoded)
    else:
        if ns.sample_per_input is not None and ns.sample_per_input < 1:
            fail("sample-per-input 必须为正整数")
        for arg in as_cli(
            ns.config,
            ns.sample_per_input,
            ns.batch_size_override,
            ns.packing_length_override,
            ns.max_steps_override,
        ):
            sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")


if __name__ == "__main__":
    main()
