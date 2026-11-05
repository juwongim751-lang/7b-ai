"""Conservative, explicitly heuristic model/training memory estimates."""
from __future__ import annotations


def estimate_training_memory_bytes(config, micro_batch_size=1, seq_len=512,
                                   precision="bf16", gradient_checkpointing=True,
                                   optimizer="adamw"):
    """Estimate training VRAM, not an allocation guarantee or benchmark.

    The baseline counts FP32 parameters, gradients, and Adam's two FP32
    moments (16 bytes/parameter). Activations/logits and runtime workspaces are
    rough estimates and vary by PyTorch/kernel/version. BitNet QAT master
    parameters and optimizer remain floating point, so no 1.58-bit saving is
    assumed.
    """
    if min(int(micro_batch_size), int(seq_len)) < 1:
        raise ValueError("micro_batch_size and seq_len must be positive")
    if optimizer.lower() != "adamw":
        raise ValueError("only AdamW memory estimation is currently supported")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32/fp16/bf16")
    params = int(config.calculate_parameters())
    dbytes = 4 if precision == "fp32" else 2
    base = params * 16  # FP32 parameter + grad + two Adam moments
    batch, seq, hidden, layers, vocab = (int(micro_batch_size), int(seq_len),
                                          int(config.hidden_size), int(config.num_layers),
                                          int(config.vocab_size))
    # Saved layer boundaries for checkpointing; larger factor when all layer
    # intermediates are retained. Fused kernels/workspaces are covered below.
    activation_factor = 3 if gradient_checkpointing else 12
    activations = batch * seq * hidden * layers * dbytes * activation_factor
    logits = batch * seq * vocab * 4  # cross-entropy/logit workspace in FP32
    raw = base + activations + logits
    reserve = int(raw * 0.25)  # allocator fragmentation and temporary kernels
    total = raw + reserve
    return {
        "parameters": params,
        "precision": precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "optimizer": "adamw",
        "parameter_gradient_optimizer_bytes": base,
        "activation_estimate_bytes": activations,
        "logits_estimate_bytes": logits,
        "runtime_reserve_bytes": reserve,
        "estimated_training_bytes": total,
        "estimated_training_gib": round(total / 1024**3, 3),
        "estimate_kind": "heuristic_not_measured",
        "bitnet_storage_credit_assumed": False,
        "note": "Actual peak depends on kernels, allocator, sequence length, batch and optimizer implementation; run a real probe before scaling.",
    }


def estimate_inference_weight_bytes(config, precision="bf16"):
    """Unpacked dense inference weights only; excludes KV cache/workspaces."""
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32/fp16/bf16")
    bytes_per = 4 if precision == "fp32" else 2
    total = int(config.calculate_parameters()) * bytes_per
    return {"bytes": total, "gib": round(total / 1024**3, 3),
            "precision": precision, "kv_cache_included": False,
            "bitnet_packed": False, "estimate_kind": "parameter_count_times_dtype"}
