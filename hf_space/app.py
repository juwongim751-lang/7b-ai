"""Inference-only Hugging Face Space entrypoint; no training or remote submissions."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from safetensors.torch import load_file

try:
    import spaces
    gpu = spaces.GPU(duration=30)
except (ImportError, AttributeError):
    def gpu(fn):
        return fn

from dataset.tokenizer import MiniCoderTokenizer
from model.config import MiniCoderConfig
from model.transformer import MiniCoder7B

MODEL_DIR = Path(os.environ.get("MINICODER_MODEL_DIR", str(Path(__file__).parent / "model"))).resolve()


@lru_cache(maxsize=1)
def load_demo():
    config_path = MODEL_DIR / "config.json"
    weights_path = MODEL_DIR / "model.safetensors"
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    for path in (config_path, weights_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing exported MiniCoder asset: {path.name}. Place model.safetensors, config.json, tokenizer.json under {MODEL_DIR}.")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = MiniCoderConfig(vocab_size=raw["vocab_size"], hidden_size=raw["hidden_size"],
                          num_layers=raw["num_hidden_layers"], num_heads=raw["num_attention_heads"],
                          num_kv_heads=raw["num_key_value_heads"], intermediate_size=raw["intermediate_size"],
                          max_position_embeddings=raw["max_position_embeddings"], rope_theta=raw.get("rope_theta", 10000.0),
                          tie_word_embeddings=raw.get("tie_word_embeddings", False),
                          use_bitnet=raw.get("use_bitnet", False), activation_bits=raw.get("activation_bits", 8),
                          precision="fp32", gradient_checkpointing=False)
    model = MiniCoder7B(cfg)
    state = load_file(str(weights_path), device="cpu")  # safetensors: no pickle execution
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"Weight keys mismatch; missing={missing}, unexpected={unexpected}")
    tokenizer = MiniCoderTokenizer()
    tokenizer.load(str(tokenizer_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


@gpu
def generate(prompt: str, max_new_tokens: int = 32, temperature: float = 0.7):
    if not prompt.strip():
        return "프롬프트를 입력하세요."
    if max_new_tokens < 1 or max_new_tokens > 64:
        raise ValueError("max_new_tokens must be in [1, 64]")
    model, tokenizer, device = load_demo()
    token_ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if input_ids.shape[1] >= model.config.max_position_embeddings:
        raise ValueError("Prompt exceeds the model context length")
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
                            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0].detach().cpu().tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(lines=5, label="Prompt"), gr.Slider(1, 64, value=32, step=1, label="New tokens"),
            gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")],
    outputs=gr.Textbox(lines=8, label="Generated text"),
    title="MiniCoder inference demo",
    description="Inference only. This Space does not pretrain, submit cloud jobs, or download private checkpoints.")

if __name__ == "__main__":
    demo.launch()
