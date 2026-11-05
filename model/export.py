"""Export MiniCoder weights without requiring FSDP at import time."""
from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Optional

import torch

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    save_file = None


class ModelExporter:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def _state_dict(self):
        model = self.model
        # FSDP is optional. Gather a full state dict only when the supplied model is FSDP.
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                    return model.state_dict()
        except (ImportError, RuntimeError, TypeError):
            pass
        return model.state_dict()

    def export(self, output_dir="output/MiniCoder-7B", prefer_safetensors=True,
               pack_bitnet=False):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        packed_layers = 0
        if pack_bitnet:
            if not bool(getattr(self.config, "use_bitnet", False)):
                raise ValueError("packed ternary export requires a BitNet-configured model")
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                if isinstance(self.model, FSDP):
                    raise NotImplementedError("packed ternary FSDP export requires a gathered full model")
            except ImportError:
                pass
            export_model = copy.deepcopy(self.model)
            from model.bitpack import convert_bitlinear_modules
            packed_layers = convert_bitlinear_modules(export_model)
            if packed_layers == 0:
                raise ValueError("no BitLinear modules were found to pack")
            raw_state = export_model.state_dict()
        else:
            raw_state = self._state_dict()
        state = {k: v.detach().cpu().contiguous() for k, v in raw_state.items()}
        tied = bool(getattr(self.config, "tie_word_embeddings", False))
        # safetensors rejects shared storage. The tied output head is represented in config.
        if tied and "lm_head.weight" in state and "embed_tokens.weight" in state:
            state.pop("lm_head.weight")
        config_dict = {
            "architectures": ["MiniCoder7B"], "model_type": "minicoder",
            "vocab_size": self.config.vocab_size, "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_layers, "num_attention_heads": self.config.num_heads,
            "num_key_value_heads": self.config.num_kv_heads, "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rope_theta": self.config.rope_theta, "tie_word_embeddings": tied,
            "use_bitnet": bool(getattr(self.config, "use_bitnet", False)),
            "activation_bits": int(getattr(self.config, "activation_bits", 8)),
            "packed_ternary": bool(pack_bitnet),
            "ternary_packing_scheme": "base3_5_trits_per_uint8_1.6_bits_per_weight" if pack_bitnet else None,
        }
        (output / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")
        if prefer_safetensors and save_file is not None:
            save_file(state, str(output / "model.safetensors"), metadata={"tied_word_embeddings": str(tied).lower()})
            weights_file = output / "model.safetensors"
        else:
            torch.save(state, output / "pytorch_model.bin")
            weights_file = output / "pytorch_model.bin"
        (output / "weights_manifest.json").write_text(json.dumps({
            "weights": weights_file.name, "format": weights_file.suffix.lstrip("."),
            "tied_word_embeddings": tied, "omitted_tied_key": "lm_head.weight" if tied else None,
            "stored_dtypes": sorted({str(t.dtype).replace("torch.", "") for t in state.values()}),
            "packed_ternary": bool(pack_bitnet),
            "packed_layers": packed_layers,
            "packed_weight_bits_per_ternary_weight": 1.6 if pack_bitnet else None,
            "packing_scheme": "five {-1,0,+1} weights encoded as a base-3 integer in one uint8; per-layer fp32 scales stored separately" if pack_bitnet else None,
            "note": ("Packed reference inference format. It unpacks ternary weights before F.linear and has no fused integer GEMM speedup." if pack_bitnet else
                     "This export stores floating-point master parameters; BitNet-style quantization is applied in forward, not packed in this file."),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Model successfully exported to {output}")
        return str(output)


if __name__ == "__main__":
    from model.config import MiniCoderConfig
    from model.transformer import MiniCoder7B
    ModelExporter(MiniCoder7B(MiniCoderConfig()), MiniCoderConfig()).export()
