"""
VLM model registry + generic loader.

We use the modern unified transformers API:
    AutoModelForImageTextToText + AutoProcessor + chat template with embedded image.
This path works across Qwen2.5-VL, Qwen3-VL, InternVL3 (-hf), LLaVA-OneVision,
and Aya-Vision in recent transformers, so we avoid per-family special casing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch

# Short key -> HF repo id. `gated` models need a license + HF_TOKEN.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── China · Alibaba (Qwen) ───────────────────────────────────────────
    "qwen2.5-vl-7b": {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "gated": False},
    "qwen2.5-vl-3b": {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct", "gated": False},
    "qwen3-vl-8b": {"hf_id": "Qwen/Qwen3-VL-8B-Instruct", "gated": False},

    # ── China · Shanghai AI Lab (InternVL) ───────────────────────────────
    "internvl3-8b": {"hf_id": "OpenGVLab/InternVL3-8B-hf", "gated": False},
    "internvl3-2b": {"hf_id": "OpenGVLab/InternVL3-2B-hf", "gated": False},

    # ── China · Zhipu / Z.ai (GLM-V) ─────────────────────────────────────
    # 9B reasoning VLM; bf16 ≈ 18 GB → use 8-bit (bnb) to fit 16 GB
    "glm-4.1v-9b-thinking": {"hf_id": "zai-org/GLM-4.1V-9B-Thinking", "gated": False},

    # ── China · OpenBMB / ModelBest (MiniCPM-V) ──────────────────────────
    "minicpm-v-4.5": {"hf_id": "openbmb/MiniCPM-V-4_5", "gated": False},  # 8B, Qwen3-8B + SigLIP2
    "minicpm-v-4": {"hf_id": "openbmb/MiniCPM-V-4", "gated": False},      # 4.1B, edge-friendly

    # ── China · AIDC-AI / Alibaba Intl (Ovis) ────────────────────────────
    "ovis2-8b": {"hf_id": "AIDC-AI/Ovis2-8B", "gated": False},  # custom_code

    # ── China · Moonshot AI (Kimi-VL) ────────────────────────────────────
    # MoE: ~3B active / ~16B total → weights need 4-bit to fit 16 GB; custom_code
    "kimi-vl-a3b": {"hf_id": "moonshotai/Kimi-VL-A3B-Instruct", "gated": False},

    # ── China · DeepSeek (DeepSeek-VL2) ──────────────────────────────────
    # MoE small variant (~2.8B active); custom_code
    "deepseek-vl2-small": {"hf_id": "deepseek-ai/deepseek-vl2-small", "gated": False},

    # ── US · Meta (Llama Vision) ─────────────────────────────────────────
    # 11B; bf16 ≈ 22 GB → 4-bit required for 16 GB
    "llama-3.2-11b-vision": {"hf_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "gated": True},

    # ── US · Microsoft (Phi) ─────────────────────────────────────────────
    "phi-4-multimodal": {"hf_id": "microsoft/Phi-4-multimodal-instruct", "gated": False},  # 5.6B
    "phi-3.5-vision": {"hf_id": "microsoft/Phi-3.5-vision-instruct", "gated": False},        # 4.2B

    # ── US · Allen Institute for AI (Molmo) ──────────────────────────────
    "molmo2-8b": {"hf_id": "allenai/Molmo2-8B", "gated": False},          # 8B, Qwen3-8B + SigLIP2
    "molmo-7b-d": {"hf_id": "allenai/Molmo-7B-D-0924", "gated": False},   # 7B, fully open data

    # ── US · IBM (Granite Vision) ────────────────────────────────────────
    "granite-vision-3.3-2b": {"hf_id": "ibm-granite/granite-vision-3.3-2b", "gated": False},

    # ── US · Google (Gemma 3, multimodal) ────────────────────────────────
    "gemma-3-4b": {"hf_id": "google/gemma-3-4b-it", "gated": True},
    "gemma-3-12b": {"hf_id": "google/gemma-3-12b-it", "gated": True},  # 12B → 4-bit for 16 GB

    # ── US/Canada · Cohere Labs (Aya Vision) ─────────────────────────────
    "aya-vision-8b": {"hf_id": "CohereLabs/aya-vision-8b", "gated": True},

    # ── Europe · Mistral AI / France (Pixtral) ───────────────────────────
    # 12B; bf16 ≈ 24 GB → 4-bit required for 16 GB
    "pixtral-12b": {"hf_id": "mistral-community/pixtral-12b", "gated": False},

    # ── Europe · Hugging Face / France (SmolVLM) ─────────────────────────
    "smolvlm2-2.2b": {"hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "gated": False},

    # ── India · Krutrim / Ola (Chitrarth) ────────────────────────────────
    # ~7B (Krutrim-1 + SigLIP); custom_code, Krutrim Community License
    "chitrarth": {"hf_id": "krutrim-ai-labs/Chitrarth", "gated": False},

    # ── Indonesia/SEA · AI Singapore (SEA-LION VL) ───────────────────────
    "sea-lion-v4-8b-vl": {"hf_id": "aisingapore/Qwen-SEA-LION-v4-8B-VL", "gated": False},
    "sea-lion-v4-4b-vl": {"hf_id": "aisingapore/Gemma-SEA-LION-v4-4B-VL", "gated": False},

    # ── Open source / community (LLaVA, Moondream) ───────────────────────
    "llava-onevision-7b": {
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "gated": False,
    },
    "moondream2": {"hf_id": "vikhyatk/moondream2", "gated": False},  # ~1.9B, fully open
}

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class LoadedModel:
    key: str
    hf_id: str
    model: Any
    processor: Any
    dtype: Any

    def num_layers(self) -> int:
        cfg = self.model.config
        for attr in ("text_config", "llm_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None and hasattr(sub, "num_hidden_layers"):
                return sub.num_hidden_layers
        return getattr(cfg, "num_hidden_layers", -1)


def load_model(
    key: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_impl: str | None = None,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> LoadedModel:
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Known: {list(MODEL_REGISTRY)}")
    hf_id = MODEL_REGISTRY[key]["hf_id"]
    torch_dtype = DTYPES[dtype]

    common = dict(trust_remote_code=True, cache_dir=cache_dir)
    if hf_token:
        common["token"] = hf_token

    processor = AutoProcessor.from_pretrained(hf_id, **common)

    # transformers >=5 renamed `torch_dtype` -> `dtype`; older uses torch_dtype.
    import transformers as _tf
    _dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_kwargs = {
        _dtype_key: torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        **common,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **model_kwargs)
    except Exception:
        # Some custom-code repos still resolve only through AutoModelForCausalLM.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    model.eval()
    return LoadedModel(key=key, hf_id=hf_id, model=model, processor=processor, dtype=torch_dtype)


def build_inputs(lm: LoadedModel, image: Any, prompt: str):
    """Build model inputs from a single (image, prompt) using the chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    proc = lm.processor
    # Preferred modern path: template tokenizes + handles the image in one call.
    try:
        inputs = proc.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
    except Exception:
        # Fallback: render text template, then run processor over text+image.
        text = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = proc(text=[text], images=[image], return_tensors="pt")
        return inputs


def get_tokenizer(lm: LoadedModel):
    """Return the underlying text tokenizer for a VLM processor."""
    tok = getattr(lm.processor, "tokenizer", None)
    return tok if tok is not None else lm.processor


def letter_token_ids(lm: LoadedModel, letters=("A", "B", "C", "D")) -> dict[str, list[int]]:
    """Map each answer letter to the set of first-token ids that could realize it.

    Tokenizers encode "A", " A", "a", " a" differently, and which variant a model
    emits as its first answer token depends on the chat template. We collect the
    first-token id of every variant so logit-scoring can take the max logit over
    all plausible realizations of each letter (robust across all model families).
    """
    tok = get_tokenizer(lm)
    out: dict[str, list[int]] = {}
    for L in letters:
        ids: set[int] = set()
        for variant in (L, " " + L, L.lower(), " " + L.lower()):
            try:
                enc = tok.encode(variant, add_special_tokens=False)
            except TypeError:
                enc = tok.encode(variant)
            if enc:
                ids.add(int(enc[0]))
        out[L] = sorted(ids)
    return out


def free_model(lm: LoadedModel) -> None:
    import gc

    del lm.model
    del lm.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
