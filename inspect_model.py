#!/usr/bin/env python3
"""
Safetensors Model Inspector
Identifies architecture, components, and properties of .safetensors model files.
"""

import json
import argparse
import re
import struct
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Safetensors header reader
# ---------------------------------------------------------------------------

def read_safetensors_header(filepath: str):
    """Read safetensors header without loading tensor data."""
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("File too small to be a valid safetensors file")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 200_000_000:
            raise ValueError(f"Header size ({header_size}) seems unreasonably large")
        header = json.loads(f.read(header_size))

    metadata = header.pop("__metadata__", {})
    tensor_info = header  # remaining entries are tensor definitions
    return metadata, tensor_info, file_size


# ---------------------------------------------------------------------------
# Tensor analysis helpers
# ---------------------------------------------------------------------------

DTYPE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8,
    "F8_E4M3": 8, "F8_E5M2": 8,
}

DTYPE_FRIENDLY = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
    "F8_E4M3": "float8 (E4M3)", "F8_E5M2": "float8 (E5M2)",
}

FINGERPRINTS = [
    "distilled_guidance_layer", "individual_token_refiner",
    "double_stream_modulation", "joint_blocks", "context_block",
    "x_block", "caption_projection", "txt_norm", "mlp_t5",
    "cap_embedder", "head.modulation", "head_modulation",
    "adaln_single", "patchify_proj", "double_blocks", "single_blocks",
    "img_attn", "txt_attn", "input_blocks", "output_blocks",
    "middle_block", "label_emb", "conditioner", "diffusion_model",
    "lora_te_", "lora_te1_", "lora_te2_", "lora_unet_",
    "cross_attn", "self_attn", "blocks", "model.layers",
    "embed_tokens", "guidance_in", "attn2", "noise_refiner",
    "transformer_blocks",
]


def analyze_tensors(tensor_info: dict):
    """Return dtype counts, total parameter count, and per-tensor shapes."""
    dtypes = Counter()
    total_params = 0
    shapes = {}

    for name, info in tensor_info.items():
        dtype = info.get("dtype", "unknown")
        tensor_shape = info.get("shape", [])
        dtypes[dtype] += 1
        shapes[name] = tensor_shape
        params = 1
        for dim in tensor_shape:
            params *= dim
        total_params += params

    return dtypes, total_params, shapes


# ---------------------------------------------------------------------------
# Component detection
# ---------------------------------------------------------------------------

def detect_components(keys: list[str]):
    components = {
        "unet": False,
        "transformer": False,
        "vae": False,
        "text_encoder": False,
        "text_encoder_2": False,
        "text_encoders": {},  # name -> tensor count for named text encoders
        "lora": False,
    }

    for k in keys:
        if k.startswith("model.diffusion_model."):
            components["unet"] = True
        if ("double_blocks." in k or "single_blocks." in k or
                "single_transformer_blocks." in k or k.startswith("transformer.") or
                k.startswith("model.double_layers.") or
                k.startswith("model.single_layers.")):
            components["transformer"] = True
        if k.startswith("first_stage_model."):
            components["vae"] = True
        if (k.startswith("vae.") or
            (k.startswith("encoder.") and "text" not in k) or
            (k.startswith("decoder.") and "text" not in k)):
            components["vae"] = True
        # Standard text encoder prefixes
        if (k.startswith("cond_stage_model.") or k.startswith("text_encoder.") or
                k.startswith("conditioner.embedders.")):
            components["text_encoder"] = True
        if k.startswith("text_encoder_2."):
            components["text_encoder_2"] = True
        # Generic text_encoders.{name}.* pattern (e.g. text_encoders.clip_l,
        # text_encoders.clip_g, text_encoders.t5xxl, text_encoders.qwen3_4b)
        if k.startswith("text_encoders."):
            parts = k.split(".")
            if len(parts) >= 2:
                enc_name = parts[1]
                components["text_encoders"][enc_name] = \
                    components["text_encoders"].get(enc_name, 0) + 1
                components["text_encoder"] = True
        if ("lora_up" in k or "lora_down" in k or "lora_A" in k or
                "lora_B" in k or ".lora." in k or k.startswith("lora_")):
            components["lora"] = True
        # LyCORIS family adapters may not use lora_up/lora_down keys.
        if (k.startswith("lycoris_") or "lokr_" in k or "loha_" in k or
                "hada_" in k or "dora_" in k or "glora" in k):
            components["lora"] = True

    return components


def _tensor_component_bucket(key: str) -> str | None:
    """Map a tensor key to a high-level model component bucket."""
    if key.startswith("model.diffusion_model."):
        return "unet"
    if ("double_blocks." in key or "single_blocks." in key or
            "single_transformer_blocks." in key or key.startswith("transformer.") or
            key.startswith("model.double_layers.") or
            key.startswith("model.single_layers.")):
        return "transformer"
    if key.startswith("first_stage_model."):
        return "vae"
    if (key.startswith("vae.") or
            (key.startswith("encoder.") and "text" not in key) or
            (key.startswith("decoder.") and "text" not in key)):
        return "vae"
    if key.startswith("text_encoder_2."):
        return "text_encoder_2"
    if (key.startswith("cond_stage_model.") or key.startswith("text_encoder.") or
            key.startswith("conditioner.embedders.") or key.startswith("text_encoders.")):
        return "text_encoder"
    return None


def _summarize_dtype_mix(dtype_counts: Counter, total_tensors: int) -> str:
    """Summarize a dtype counter using the same outlier tolerance as global precision."""
    if not dtype_counts or total_tensors <= 0:
        return "-"
    if len(dtype_counts) == 1:
        only = next(iter(dtype_counts.keys()))
        return DTYPE_FRIENDLY.get(only, only)

    dominant_dtype, dominant_count = dtype_counts.most_common(1)[0]
    dominant_pct = dominant_count / total_tensors * 100
    if dominant_pct >= 99.0:
        return DTYPE_FRIENDLY.get(dominant_dtype, dominant_dtype)
    return "Mixed (" + ", ".join(
        DTYPE_FRIENDLY.get(d, d) for d, _ in dtype_counts.most_common()
    ) + ")"


def analyze_component_precisions(tensor_info: dict) -> dict[str, Counter]:
    """Build per-component dtype counters from tensor keys."""
    component_dtypes: dict[str, Counter] = {}
    for name, info in tensor_info.items():
        bucket = _tensor_component_bucket(name)
        if not bucket:
            continue
        dtype = info.get("dtype", "unknown")
        if bucket not in component_dtypes:
            component_dtypes[bucket] = Counter()
        component_dtypes[bucket][dtype] += 1
    return component_dtypes


def build_component_precision_summary(component_dtypes: dict[str, Counter]) -> str:
    """Render per-component precision summary for mixed checkpoints."""
    ordered_labels = [
        ("unet", "UNet"),
        ("transformer", "Transformer"),
        ("vae", "VAE"),
        ("text_encoder", "Text Encoder"),
        ("text_encoder_2", "Text Encoder 2"),
    ]
    parts = []
    for key, label in ordered_labels:
        dtype_counts = component_dtypes.get(key)
        if not dtype_counts:
            continue
        summary = _summarize_dtype_mix(dtype_counts, sum(dtype_counts.values()))
        parts.append(f"{label}: {summary}")
    return " | ".join(parts)


def build_component_precision_map(component_dtypes: dict[str, Counter]) -> dict[str, str]:
    """Return per-component precision summaries keyed by component id."""
    ordered_keys = [
        "unet",
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
    ]
    out: dict[str, str] = {}
    for key in ordered_keys:
        dtype_counts = component_dtypes.get(key)
        if not dtype_counts:
            continue
        out[key] = _summarize_dtype_mix(dtype_counts, sum(dtype_counts.values()))
    return out


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
#
# Detection uses unique "fingerprint" keys per architecture, ordered from
# most specific to least specific.  Works for both full checkpoints and
# LoRAs (key substrings are preserved in all common LoRA formats).
#
# Primary detection keys (from ComfyUI model_detection.py & diffusers):
#   Chroma          → distilled_guidance_layer
#   HunyuanVideo    → individual_token_refiner
#   Flux 2 Klein    → double_stream_modulation
#   SD 3 / 3.5      → joint_blocks + context_block / x_block
#   HiDream         → caption_projection
#   Qwen Image/Edit → txt_norm  (top-level, not inside other blocks)
#   Hunyuan     → mlp_t5
#   Z-Image         → cap_embedder  (hidden_dim 3840 distinguishes from Lumina2)
#   Wan             → head.modulation  (full model) / cross_attn+self_attn in blocks (LoRA)
#   LTX / LTX 2    → adaln_single
#   Flux.1          → double_blocks + img_attn
#   SDXL            → input_blocks + label_emb / lora_te2_
#   SD 1.5          → input_blocks + context_dim 768 / lora_te_
# ---------------------------------------------------------------------------


def _max_block_index(keys: list[str], block_name: str) -> int:
    mx = -1
    for k in keys:
        if block_name in k:
            try:
                idx = int(k.split(block_name)[1].split(".")[0])
                mx = max(mx, idx)
            except (ValueError, IndexError):
                pass
    return mx


def _collect_lora_up_dims(keys, shapes):
    """Collect the output dimensions of lora_up / lora_B weights.
    These correspond to the hidden dimension of the target layer."""
    dims = set()
    for k in keys:
        if ("lora_up" in k or "lora_B" in k) and k.endswith(".weight"):
            s = shapes.get(k, [])
            if len(s) >= 2:
                dims.add(s[0])
    return dims


def _detect_lora_rank(keys, shapes):
    """Detect the LoRA rank from the weight shapes."""
    ranks = []
    for k in keys:
        if ("lora_down" in k or "lora_A" in k) and k.endswith(".weight"):
            s = shapes.get(k, [])
            if len(s) >= 2:
                ranks.append(s[0])
    if not ranks:
        return None
    return Counter(ranks).most_common(1)[0][0]


def detect_adapter_type(keys: list[str], metadata: dict) -> str | None:
    """Detect adapter family/type when file contains adapter weights."""
    if not keys:
        return None

    key_blob = "\n".join(keys).lower()
    lyco_cfg = str(metadata.get("lycoris_config", "")).lower()
    ss_network_module = str(metadata.get("ss_network_module", "")).lower()
    ss_network_args = str(metadata.get("ss_network_args", "")).lower()
    algo_lokr = bool(re.search(r'"algo"\s*:\s*"lokr"', lyco_cfg + " " + ss_network_args))
    algo_loha = bool(re.search(r'"algo"\s*:\s*"loha"', lyco_cfg + " " + ss_network_args))
    algo_dora = bool(re.search(r'"algo"\s*:\s*"dora"', lyco_cfg + " " + ss_network_args))
    algo_glora = bool(re.search(r'"algo"\s*:\s*"glora"', lyco_cfg + " " + ss_network_args))

    # Specific algorithms first
    if ("lokr_w1" in key_blob or "lokr_w2" in key_blob or algo_lokr):
        return "LoKr"
    if ("hada_w1_a" in key_blob or "hada_w1_b" in key_blob or
            "hada_w2_a" in key_blob or "hada_w2_b" in key_blob or algo_loha):
        return "LoHa"
    if "dora_scale" in key_blob or algo_dora:
        return "DoRA"
    # GLoRA commonly stores factorized weights as a1/a2/b1/b2 and algo in ss_network_args.
    has_glora_factorized = all(tok in key_blob for tok in (".a1", ".a2", ".b1", ".b2"))
    if ("glora" in key_blob or algo_glora or has_glora_factorized):
        return "GLoRA"

    # Generic LyCORIS container (non-LoRA variants, unknown exact algo)
    if ("lycoris_" in key_blob or "lycoris" in lyco_cfg or
            "lycoris" in ss_network_module):
        return "LyCORIS"

    # Standard LoRA formats
    if ("lora_up" in key_blob or "lora_down" in key_blob or
            "lora_a" in key_blob or "lora_b" in key_blob or ".lora." in key_blob):
        return "LoRA"

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_architecture(keys: list[str], shapes: dict, total_params: int,
                        components: dict, metadata: dict):
    """Detect model architecture. Returns (arch_name, details_dict)."""
    details = {}

    # Surface useful metadata
    spec_arch = metadata.get("modelspec.architecture", "")
    if spec_arch:
        details["metadata_architecture"] = spec_arch
    ss_base = metadata.get("ss_base_model_version", "")
    if ss_base:
        details["training_base_model"] = ss_base

    # LoRA rank (compute early, attach to details)
    if components["lora"]:
        details["lora_rank"] = _detect_lora_rank(keys, shapes)
        adapter_type = detect_adapter_type(keys, metadata)
        if adapter_type:
            details["adapter_type"] = adapter_type

    # 1. Try metadata-based detection (most reliable, especially for LoRAs)
    meta_result = _detect_from_metadata(metadata)
    if meta_result:
        return meta_result, details

    # 2. Key-pattern detection
    key_blob = "\n".join(keys)
    return _detect_from_keys(
        keys, key_blob, shapes, total_params, components, metadata, details
    )


# ---------------------------------------------------------------------------
# Metadata-based detection
# ---------------------------------------------------------------------------

def _build_metadata_blob(metadata: dict):
    """Build normalized metadata text plus key fields for variant detection."""
    spec = metadata.get("modelspec.architecture", "").lower()
    ss = metadata.get("ss_base_model_version", "").lower()
    title = metadata.get("modelspec.title", "").lower()
    desc = metadata.get("modelspec.description", "").lower()
    output_name = metadata.get("ss_output_name", "").lower()
    sd_model = metadata.get("ss_sd_model_name", "").lower()

    # Also scan all short metadata values for clues.
    all_meta = f"{spec} {ss} {title} {desc} {output_name} {sd_model}"
    for _, mv in metadata.items():
        v = str(mv).lower()
        if len(v) < 200:  # skip huge JSON blobs
            all_meta += " " + v

    # Additionally, parse sshs_cp0/sshs_cp1 for key training-name hints
    # commonly embedded as large JSON strings.
    sshs_meta = ""
    for cpk in ("sshs_cp0", "sshs_cp1"):
        cpv = metadata.get(cpk, "")
        if cpv and len(cpv) > 200:
            extracted = []

            # Try to decode nested JSON (common in sshs_cp* fields).
            decoded = cpv
            for _ in range(2):
                try:
                    parsed = json.loads(decoded)
                except Exception:
                    break
                if isinstance(parsed, str):
                    decoded = parsed
                    continue
                if isinstance(parsed, dict):
                    for field in (
                        "ss_sd_model_name",
                        "ss_output_name",
                        "modelspec.title",
                        "modelspec.description",
                        "ss_base_model_version",
                    ):
                        fv = parsed.get(field)
                        if fv:
                            extracted.append(str(fv).lower())
                    break

            # Fallback: regex scan raw text, handling both quoted and escaped-quoted keys.
            if not extracted:
                cpv_l = str(cpv).lower()
                for field in (
                    "ss_sd_model_name",
                    "ss_output_name",
                    "modelspec.title",
                    "modelspec.description",
                    "ss_base_model_version",
                ):
                    pat = rf'(?:\\?"{re.escape(field)}\\?"\s*:\s*\\?"([^"\\]+))'
                    m = re.search(pat, cpv_l)
                    if m:
                        extracted.append(m.group(1).lower())

            if extracted:
                sshs_meta += " " + " ".join(extracted)
    all_meta += " " + sshs_meta
    return {
        "all_meta": all_meta,
        "sshs_meta": sshs_meta,
        "spec": spec,
        "ss": ss,
        "output_name": output_name,
        "sd_model": sd_model,
    }


def _detect_from_metadata(metadata: dict):
    """Try to identify architecture purely from safetensors __metadata__."""
    meta = _build_metadata_blob(metadata)
    all_meta = meta["all_meta"]
    sshs_meta = meta["sshs_meta"]
    ss = meta["ss"]
    output_name = meta["output_name"]
    sd_model = meta["sd_model"]

    if not all_meta.strip():
        return None

    # Order matters: check specific variants before generic families

    # Chroma
    if "chroma" in all_meta:
        return "Chroma"
    if "auraflow" in all_meta or "aura_flow" in all_meta or "aura flow" in all_meta:
        return "Aura Flow"

    # Flux variants (specific before generic)
    if "kontext" in all_meta:
        return "Flux Kontext"
    if "krea" in all_meta:
        return "Flux Krea"
    # Flux 2 Klein: must have "klein" explicitly
    if "klein" in all_meta:
        return "Flux 2 Klein"
    # Flux 2: "flux2" without "klein"
    if "flux2" in all_meta or "flux.2" in all_meta or "flux 2" in all_meta:
        return "Flux 2"
    if "flux" in all_meta:
        if "schnell" in all_meta:
            return "Flux.1 Schnell"
        return "Flux.1 Dev"

    # LTX / LNX (before HiDream; some LTX variants can contain caption_projection)
    if "lnx" in all_meta:
        return "LNX Video"
    if "ltx" in all_meta:
        if "ltxvideo2" in all_meta or "ltx2" in all_meta or "ltx-2" in all_meta:
            return "LTX 2"
        return "LTX"

    # HiDream
    if "hidream" in all_meta:
        return "HiDream"

    # SD3 variants (check 3.5 before 3)
    if ("sd3.5" in all_meta or "sd35" in all_meta or
            "stable-diffusion-3.5" in all_meta or "stable_diffusion_3_5" in all_meta or
            "3-5-large" in all_meta or "3-5-medium" in all_meta or
            "stable-diffusion-3-3-5" in all_meta):
        return "SD3.5"
    if ("sd3" in all_meta or "stable-diffusion-v3" in all_meta or
            "stable_diffusion_3" in all_meta or "stable-diffusion-3" in all_meta):
        # Make sure we don't false-match on "stable-diffusion-3-3-5" (already caught above)
        return "SD3"

    # SDXL variants (specific forks before generic SDXL)
    # NAI: check sshs_meta for "illustrious" from training checkpoint name
    if "noob" in all_meta or "nai" in all_meta.split():
        return "NAI"
    if (("pony" in all_meta and "v7" in all_meta) or
            "ponyv7" in all_meta or "pony v7" in all_meta):
        return "Pony7"
    if "pony" in all_meta or "pdxl" in all_meta:
        return "PDXL"
    if "illustrious" in all_meta or "ilxl" in all_meta or "noobai" in all_meta:
        return "ILXL"
    # Also check sshs_meta specifically for illustrious (NAI trained on illustrious)
    if "illustrious" in sshs_meta:
        return "ILXL"
    if "sdxl" in all_meta or "sd_xl" in all_meta:
        return "SDXL"
    if ("v1-5" in all_meta or "v1_5" in all_meta or
            "stable-diffusion-v1" in all_meta or ss == "sd_1.5" or
            "sd 1.5" in all_meta or "sd1.5" in all_meta):
        return "SD 1.5"

    # Wan: check before Hunyuan because Wan metadata can include
    # implementation URLs containing "HunyuanVideo".
    if "wan" in all_meta:
        wan_ver = "Wan"
        # Version detection: check for 2.2 in sd_model_name (more reliable)
        if "wan2.2" in all_meta or "wan 2.2" in all_meta or "2.2" in sd_model:
            wan_ver = "Wan 2.2"
        elif "wan2.1" in all_meta or "wan 2.1" in all_meta or "2.1" in sd_model:
            wan_ver = "Wan 2.1"
        # HIGH/LOW noise pass detection
        if "high_noise" in all_meta or "high noise" in all_meta:
            return f"{wan_ver} HIGH"
        if "low_noise" in all_meta or "low noise" in all_meta:
            return f"{wan_ver} LOW"
        if "i2v" in all_meta or "image2video" in all_meta:
            return f"{wan_ver} I2V"
        if "t2v" in all_meta or "text2video" in all_meta:
            return f"{wan_ver} T2V"
        return wan_ver

    # Hunyuan
    if "hunyuan" in all_meta:
        if "video" in all_meta:
            return "HunyuanVideo"
        return "Hunyuan"

    # Z-Image
    if "z-image" in all_meta or "zimage" in all_meta or "z_image" in all_meta:
        return _zimage_label(metadata, all_meta)

    # Qwen
    if "qwen" in all_meta:
        if ("edit" in all_meta or "qwen_edit" in all_meta or
                "qwen-edit" in all_meta or "image_edit" in all_meta):
            return "Qwen Edit"
        return None

    return None


# ---------------------------------------------------------------------------
# Key-pattern detection  (ordered most-specific → least-specific)
# ---------------------------------------------------------------------------

def _zimage_label(metadata=None, key_blob=""):
    """Return canonical Z-Image label from behavior hints (CFG vs no-CFG)."""
    all_hints = key_blob.lower()
    if metadata:
        meta = _build_metadata_blob(metadata)
        all_hints = f"{all_hints} {meta['all_meta']}"

    # Variant logic intentionally avoids model-name matching.
    # Official behavior split: Turbo uses no CFG, Base uses CFG.
    has_cfg_true = bool(re.search(
        r'(?:use_cfg|do_cfg|classifier[_ ]?free[_ ]?guidance)\s*["=: ]+\s*true',
        all_hints,
    ))
    has_cfg_false = bool(re.search(
        r'(?:use_cfg|do_cfg|classifier[_ ]?free[_ ]?guidance)\s*["=: ]+\s*false',
        all_hints,
    ))

    # Prefer Turbo when there is any explicit no-CFG signal.
    if has_cfg_false:
        return "Z-Image Turbo"
    if has_cfg_true:
        return "Z-Image Base"

    # Ambiguous fallback: this helper is only called from confirmed Z-Image paths.
    return "Z-Image Turbo"


def _detect_from_keys(keys, key_blob, shapes, total_params, components, metadata, details):

    # ── Chroma (Flux-based, unique distilled_guidance_layer) ──────────
    if "distilled_guidance_layer" in key_blob:
        return "Chroma", details

    # ── HunyuanVideo (has individual_token_refiner, also double_blocks) ──
    if "individual_token_refiner" in key_blob:
        return "HunyuanVideo", details

    # ── Flux 2 Klein (unique double_stream_modulation) ────────────────
    if "double_stream_modulation" in key_blob:
        db = _max_block_index(keys, "double_blocks.") + 1
        sb = _max_block_index(keys, "single_blocks.") + 1
        if db > 0:
            details["double_blocks"] = db
        if sb > 0:
            details["single_blocks"] = sb
        meta_blob = _build_metadata_blob(metadata)["all_meta"]
        if "klein" in meta_blob:
            return "Flux 2 Klein", details
        return "Flux 2 Dev", details

    # ── SD3 / SD3.5 (joint_blocks + context_block / x_block) ─────────
    #    Also matches kohya format: lora_unet_joint_blocks_*_context_block_*
    if "joint_blocks" in key_blob and ("context_block" in key_blob or "x_block" in key_blob):
        if "x_block" in key_blob and "attn2" in key_blob:
            return "SD3.5", details
        return "SD3", details
    # SD3 LoRA in diffusers format: transformer.joint_blocks.*
    if "joint_blocks" in key_blob:
        return "SD3", details

    # ── LTX / LTX 2 (adaln_single or transformer_blocks + patchify) ──
    if "adaln_single" in key_blob or "patchify_proj" in key_blob:
        if "audio_adaln_single" in key_blob:
            return "LTX 2", details
        return "LTX", details

    # ── Aura Flow (checkpoint key layout: model.single_layers/double_layers)
    if ("model.single_layers." in key_blob and
            "model.double_layers." in key_blob and
            "model.cond_seq_linear" in key_blob):
        return "Aura Flow", details

    # ── HiDream (unique caption_projection) ───────────────────────────
    if ("caption_projection" in key_blob and
            "adaln_single" not in key_blob and
            "patchify_proj" not in key_blob):
        return "HiDream", details

    # ── Qwen Image / Qwen Edit (unique add_k_proj / add_q_proj) ─────
    #    Qwen uses transformer_blocks with add_k_proj/add_q_proj which
    #    is unique — no other architecture uses these key names.
    if "add_k_proj" in key_blob or "add_q_proj" in key_blob:
        meta_blob = _build_metadata_blob(metadata)["all_meta"]
        if ("qwen_edit" in meta_blob or "qwen-edit" in meta_blob or
                "image_edit" in meta_blob or " edit " in f" {meta_blob} "):
            return "Qwen Edit", details
        # Full merged checkpoints with qwen text encoder + VAE are edit models.
        if (not components.get("lora") and
                "model.diffusion_model." in key_blob and
                "text_encoders.qwen" in key_blob and
                "vae." in key_blob):
            return "Qwen Edit", details
        if "img_mod." not in key_blob and "txt_mod." not in key_blob:
            return "Qwen Edit", details
        # Edit adapters are commonly saved under transformer.transformer_blocks
        # (without diffusion_model prefix) and mixed lora.down/lora.up keys.
        if "transformer.transformer_blocks" in key_blob and "diffusion_model" not in key_blob:
            return "Qwen Edit", details
        return "Qwen Image", details
    # Also check for top-level txt_norm (full Qwen checkpoints)
    if "txt_norm" in key_blob and "txt_in" not in key_blob:
        # Full Qwen Edit checkpoints expose model.diffusion_model.* + txt_norm.
        if "model.diffusion_model." in key_blob and not components.get("lora"):
            return "Qwen Edit", details
        return "Qwen Image", details
    if any(k.startswith("txt_norm") for k in keys):
        if "model.diffusion_model." in key_blob and not components.get("lora"):
            return "Qwen Edit", details
        return "Qwen Image", details

    # ── Hunyuan (unique mlp_t5 projection) ────────────────────────────
    if "mlp_t5" in key_blob:
        return "Hunyuan", details

    # ── Z-Image (unique cap_embedder, hidden_dim 3840) ───────────────
    if "cap_embedder" in key_blob:
        return _detect_zimage_variant(keys, shapes, metadata, details)

    # ── Z-Image LoRA (diffusion_model.layers.{N}.attention) ──────────
    #    Z-Image LoRAs use diffusion_model.layers (not transformer_blocks,
    #    not input_blocks). Hidden dim 3840.
    if "diffusion_model.layers." in key_blob and "attention" in key_blob:
        return _detect_zimage_variant(keys, shapes, metadata, details)
    if "auraflow" in key_blob or "aura_flow" in key_blob:
        return "Aura Flow", details

    # ── Wan (unique head.modulation for full models) ──────────────────
    if "head.modulation" in key_blob or "head_modulation" in key_blob:
        return _detect_wan_variant(keys, shapes, total_params, metadata, details)

    # ── LTX v1 LoRA (diffusion_model.transformer_blocks with attn1/attn2)
    #    LTX uses attn1+attn2 inside transformer_blocks, which is distinct
    #    from SD (which uses input_blocks/output_blocks) and from Qwen
    #    (which uses add_k_proj, already caught above).
    if "diffusion_model.transformer_blocks" in key_blob:
        if "attn1" in key_blob or "attn2" in key_blob:
            return "LTX", details

    # ── HunyuanVideo LoRA (double_blocks with specific naming) ────────
    #    Must check BEFORE Flux detection.
    if "double_blocks" in key_blob or "single_blocks" in key_blob:
        has_video_dims = any(len(shapes.get(k, [])) == 5 for k in keys)
        has_hunyuanvideo_lora_layout = (
            "transformer.double_blocks." in key_blob and
            ("img_attn_qkv" in key_blob or "txt_attn_qkv" in key_blob) and
            ("img_mod.linear" in key_blob or "txt_mod.linear" in key_blob)
        )
        if has_video_dims or has_hunyuanvideo_lora_layout:
            return "HunyuanVideo", details

    # ── Flux Kontext / Flux diffusers-format LoRAs ───────────────────
    #    Kontext and other diffusers-trained Flux LoRAs use
    #    transformer.transformer_blocks / transformer.single_transformer_blocks
    #    (note: "single_transformer_blocks" not "single_blocks")
    if "single_transformer_blocks" in key_blob:
        return _detect_flux_variant(keys, key_blob, shapes, total_params, details)
    if ("transformer.transformer_blocks" in key_blob and
            "double_blocks" not in key_blob and
            "joint_blocks" not in key_blob and
            "add_k_proj" not in key_blob):
        # Diffusers-format Flux LoRA (transformer.transformer_blocks = double_blocks equivalent)
        return _detect_flux_variant(keys, key_blob, shapes, total_params, details)

    # ── Flux.1 family (double_blocks / single_blocks + img_attn) ─────
    if "double_blocks" in key_blob or "single_blocks" in key_blob:
        return _detect_flux_variant(keys, key_blob, shapes, total_params, details)

    # ── LTX LoRA fallback (transformer_blocks without other markers) ──
    #    LTX uses transformer_blocks.{N} like PixArt, but has specific
    #    cross-attention patterns.  Check before SD/SDXL.
    if "transformer_blocks" in key_blob:
        # If no input_blocks/output_blocks/diffusion_model, this is likely
        # a transformer-based model (LTX, PixArt, etc), not SD.
        if ("input_blocks" not in key_blob and "output_blocks" not in key_blob
                and "diffusion_model" not in key_blob
                # Avoid false positives for SD/SDXL LoRA + LyCORIS formats.
                and "lora_unet_" not in key_blob
                and "lora_te_" not in key_blob
                and "lora_te1_" not in key_blob
                and "lora_te2_" not in key_blob
                and "down_blocks" not in key_blob
                and "up_blocks" not in key_blob
                and "mid_block" not in key_blob
                and "lycoris" not in key_blob):
            return "LTX", details

    # ── SD 1.5 / SDXL (input_blocks / diffusion_model) ───────────────
    if ("input_blocks" in key_blob or "output_blocks" in key_blob or
            "middle_block" in key_blob):
        return _detect_sd_variant(
            keys, key_blob, shapes, total_params, components, metadata, details
        )
    # Diffusers UNet checkpoints
    if ("down_blocks" in key_blob and "up_blocks" in key_blob and
            ("mid_block" in key_blob or "conv_in" in key_blob)):
        return _detect_sd_variant(
            keys, key_blob, shapes, total_params, components, metadata, details
        )
    # diffusion_model prefix — but NOT diffusion_model.transformer_blocks
    # (already caught as LTX) and NOT diffusion_model.layers (Z-Image)
    if "diffusion_model" in key_blob:
        if ("diffusion_model.transformer_blocks" not in key_blob and
                "diffusion_model.layers." not in key_blob):
            return _detect_sd_variant(
                keys, key_blob, shapes, total_params, components, metadata, details
            )

    # ── SDXL LoRA (kohya lora_te2_ = dual text encoder = SDXL) ───────
    if "lora_te2_" in key_blob:
        return _detect_sdxl_pony_ilxl(keys, metadata, details), details
    if "lora_te1_" in key_blob:
        return _detect_sdxl_pony_ilxl(keys, metadata, details), details

    # ── SD 1.5 LoRA (kohya lora_te_ single text encoder) ─────────────
    if "lora_te_" in key_blob and "lora_te1_" not in key_blob:
        return "SD 1.5", details

    # ── Wan LoRA fallback (blocks + cross_attn + self_attn) ──────────
    if ("cross_attn" in key_blob and "self_attn" in key_blob and
            "blocks" in key_blob):
        return _detect_wan_variant(keys, shapes, total_params, metadata, details)

    # ── Ambiguous blocks.{N} — use dimension analysis ────────────────
    if "blocks" in key_blob and ("attn" in key_blob or "self_attn" in key_blob):
        return _detect_from_dims(keys, shapes, total_params, metadata, details)

    # ── Qwen LLM (standalone text encoder: model.layers + embed_tokens)
    if ("model.layers" in key_blob and "self_attn" in key_blob and
            "embed_tokens" in key_blob):
        return "Qwen (text encoder)", details

    if components.get("unet"):
        if ("conditioner." in key_blob or
                "conditioner.embedders.1" in key_blob or
                "text_encoder_2" in key_blob or
                "clip_g" in key_blob):
            return _detect_sdxl_pony_ilxl(keys, metadata, details), details
        return "SD 1.5", details

    # ── Standalone components ─────────────────────────────────────────
    if components.get("vae") and not components.get("unet") and not components.get("transformer"):
        vae_prefixes = ("first_stage_model.", "vae.", "encoder.", "decoder.")
        non_vae = [k for k in keys if not any(k.startswith(pref) for pref in vae_prefixes)]
        if len(non_vae) <= max(8, len(keys) // 50):
            return "VAE (standalone)", details
    if components.get("text_encoder") and not components.get("unet") and not components.get("transformer"):
        return "Text Encoder (standalone)", details

    return "Unknown", details


# ---------------------------------------------------------------------------
# Architecture-specific sub-detectors
# ---------------------------------------------------------------------------

def _detect_flux_variant(keys, key_blob, shapes, total_params, details):
    """Distinguish Flux.1 Dev / Schnell / Kontext and count blocks."""
    # Support both kohya format (double_blocks.N) and diffusers format
    # (transformer.transformer_blocks.N / transformer.single_transformer_blocks.N)
    db = _max_block_index(keys, "double_blocks.") + 1
    sb = _max_block_index(keys, "single_blocks.") + 1

    # Diffusers format uses transformer_blocks for double and
    # single_transformer_blocks for single
    if db <= 0:
        db = _max_block_index(keys, "transformer_blocks.") + 1
    if sb <= 0:
        sb = _max_block_index(keys, "single_transformer_blocks.") + 1

    if db > 0:
        details["double_blocks"] = db
    if sb > 0:
        details["single_blocks"] = sb

    has_guidance = "guidance_in" in key_blob

    # LoRAs won't target all blocks, so block count isn't reliable for
    # distinguishing Dev/Schnell. For LoRAs, default to "Flux.1 Dev".
    is_lora = "lora_A" in key_blob or "lora_B" in key_blob or "lora_down" in key_blob

    # Standard Flux.1: 19 double, 38 single
    if db == 19 and sb == 38:
        details["guidance_input"] = has_guidance
        if has_guidance:
            return "Flux.1 Dev", details
        return "Flux.1 Schnell", details

    # For LoRAs, we can't reliably count blocks
    if is_lora:
        details["guidance_input"] = has_guidance
        return "Flux.1 Dev", details

    # Smaller block counts could be a distilled / lite variant
    if 0 < db < 19 or 0 < sb < 38:
        details["guidance_input"] = has_guidance
        return "Flux (compact variant)", details

    # Larger
    details["guidance_input"] = has_guidance
    return "Flux.1 Dev", details


def _detect_sd_variant(keys, key_blob, shapes, total_params, components, metadata, details):
    """Distinguish SD 1.5 vs SDXL (full checkpoints and LoRAs)."""
    is_sdxl = ("label_emb" in key_blob or "conditioner" in key_blob or
               "lora_te2_" in key_blob or "lora_te1_" in key_blob)

    if not is_sdxl:
        # Dimension check: SDXL cross-attn context_dim = 2048
        up_dims = _collect_lora_up_dims(keys, shapes)
        if any(d == 2048 or d == 2816 for d in up_dims):
            is_sdxl = True

    if is_sdxl:
        arch = _detect_sdxl_pony_ilxl(keys, metadata, details)
    else:
        arch = "SD 1.5"

    # Checkpoint sub-type (only for non-LoRA)
    if not components.get("lora"):
        if components.get("vae") and components.get("text_encoder"):
            details["model_type"] = "Checkpoint"
        elif components.get("vae"):
            details["model_type"] = "Checkpoint (UNet + VAE)"
        elif components.get("text_encoder"):
            details["model_type"] = "Checkpoint (UNet + Text Enc)"
        else:
            details["model_type"] = "UNet only"

    return arch, details


def _detect_sdxl_pony_ilxl(keys, metadata, details):
    """Try to distinguish PDXL / ILXL / NAI from generic SDXL.
    These are architecturally identical — metadata is checked first in
    _detect_from_metadata.  If we reach here there was no metadata match,
    so return plain SDXL."""
    meta_blob = _build_metadata_blob(metadata)["all_meta"]
    if (("pony" in meta_blob and "v7" in meta_blob) or
            "ponyv7" in meta_blob or "pony v7" in meta_blob):
        return "Pony7"
    if "pony" in meta_blob or "pdxl" in meta_blob:
        return "PDXL"
    if "illustrious" in meta_blob or "ilxl" in meta_blob or "noobai" in meta_blob:
        return "ILXL"
    if "noob" in meta_blob or " nai " in f" {meta_blob} ":
        return "NAI"
    return "SDXL"


def _detect_zimage_variant(keys, shapes, metadata, details):
    """Distinguish Z-Image from Lumina 2 (both use cap_embedder).
    Also handles Z-Image LoRAs with diffusion_model.layers pattern."""
    # Z-Image hidden_dim = 3840 (30 heads × 128).
    # Some checkpoints can have ambiguous intermediate dims; do not classify
    # as Lumina2 from dims alone to avoid false positives.
    key_blob = "\n".join(keys)
    has_lumina_marker = "lumina" in key_blob

    # Structural split from observed checkpoints:
    # - Turbo: wrapped transformer keys under model.diffusion_model.*
    # - Base:  flat transformer keys under layers.{N}.*
    has_wrapped_diffusion = any(k.startswith("model.diffusion_model.") for k in keys)
    has_flat_layers = any(re.match(r"^layers\.\d+\.", k) for k in keys)
    if has_wrapped_diffusion:
        details["zimage_layout"] = "wrapped_diffusion_model"
        return "Z-Image Turbo", details
    if has_flat_layers and not has_wrapped_diffusion:
        details["zimage_layout"] = "flat_layers"
        return "Z-Image Base", details

    # Check cap_embedder or any large weight dim.
    for k in keys:
        if "cap_embedder" in k and k.endswith(".weight"):
            s = shapes.get(k, [])
            if s:
                max_dim = max(s)
                if max_dim >= 3840:
                    return _zimage_label(metadata, key_blob), details
                if 2280 <= max_dim <= 2320:
                    return "Lumina 2", details
                if max_dim >= 2304 and max_dim < 3840 and has_lumina_marker:
                    return "Lumina 2", details

    # For LoRAs, check dimension of lora_up weights
    up_dims = _collect_lora_up_dims(keys, shapes)
    if any(3800 <= d <= 3900 for d in up_dims):
        return _zimage_label(metadata, key_blob), details
    if any(2280 <= d <= 2320 for d in up_dims):
        return "Lumina 2", details

    # Explicit Lumina markers
    if "lumina" in key_blob:
        return "Lumina 2", details

    # Canonical fallback handled by _zimage_label (Turbo unless explicit Base).
    return _zimage_label(metadata, key_blob), details


def _detect_wan_variant(keys, shapes, total_params, metadata, details):
    """Detect Wan variant (2.1 vs 2.2), HIGH/LOW pass, and size."""
    # 5D tensors (3D conv) indicate video diffusion
    has_3d = any(len(shapes.get(k, [])) == 5 for k in keys)
    if has_3d:
        details["has_3d_conv"] = True

    # Variant sub-keys (from ComfyUI detection)
    key_blob = "\n".join(keys)
    if "vace_patch_embedding" in key_blob:
        details["wan_variant"] = "VACE"
    elif "control_adapter" in key_blob:
        details["wan_variant"] = "Camera"
    elif "img_emb" in key_blob:
        details["wan_variant"] = "I2V"

    # Size-based sub-classification
    if total_params > 0:
        if total_params < 2_000_000_000:
            details["variant_note"] = "1.3B"
        elif total_params < 10_000_000_000:
            details["variant_note"] = "Large"
        else:
            details["variant_note"] = "14B"

    # Wan 2.2 HIGH/LOW pass detection:
    # HIGH pass models target higher-level blocks (larger indices),
    # LOW pass models target lower-level blocks (smaller indices).
    # Detect from block index ranges in LoRA keys.
    block_indices = set()
    for k in keys:
        if "blocks" in k:
            try:
                # Extract block index: blocks.N. or blocks_N_
                for sep in ["blocks.", "blocks_"]:
                    if sep in k:
                        idx_str = k.split(sep)[1].split(".")[0].split("_")[0]
                        block_indices.add(int(idx_str))
            except (ValueError, IndexError):
                pass

    if block_indices:
        max_idx = max(block_indices)
        min_idx = min(block_indices)
        details["block_range"] = f"{min_idx}-{max_idx}"
    meta = _build_metadata_blob(metadata)
    all_meta = meta["all_meta"]
    sd_model = meta["sd_model"]
    spec = meta["spec"]

    wan_ver = "Wan"
    ver_source = f"{sd_model} {spec}"
    if "wan2.2" in ver_source or "wan 2.2" in ver_source or "2.2" in ver_source:
        wan_ver = "Wan 2.2"
    elif "wan2.1" in ver_source or "wan 2.1" in ver_source or "2.1" in ver_source:
        wan_ver = "Wan 2.1"

    has_i2v = (
        "img_emb" in key_blob or " i2v " in f" {all_meta} " or "image2video" in all_meta
    )
    has_t2v = (" t2v " in f" {all_meta} " or "text2video" in all_meta)
    is_lora = any(
        ("lora_" in k or ".lora." in k or "lycoris_" in k)
        for k in keys
    )
    if has_i2v:
        return f"{wan_ver} I2V", details
    if has_t2v or not is_lora:
        return f"{wan_ver} T2V", details
    return wan_ver, details


def _detect_from_dims(keys, shapes, total_params, metadata, details):
    """Last-resort dimension-based detection for ambiguous blocks.{N} models."""
    up_dims = _collect_lora_up_dims(keys, shapes)
    if not up_dims:
        # Check non-LoRA weight dims
        for k in keys:
            if k.endswith(".weight") and "blocks" in k:
                s = shapes.get(k, [])
                if len(s) >= 2:
                    up_dims.add(max(s))

    # Z-Image: hidden_dim ~3840
    if any(3800 <= d <= 3900 for d in up_dims):
        return _zimage_label(metadata, "\n".join(keys)), details
    # Hunyuan: hidden_size ~1408
    if any(1400 <= d <= 1420 for d in up_dims):
        return "Hunyuan", details
    # Wan 1.3B: dim ~1536
    if any(1520 <= d <= 1550 for d in up_dims):
        return "Wan", details
    # Wan 14B: dim ~5120
    if any(5100 <= d <= 5150 for d in up_dims):
        return "Wan", details

    return "Unknown", details


# ---------------------------------------------------------------------------
# Model type classification
# ---------------------------------------------------------------------------

def classify_model_type(components: dict, arch: str):
    """Classify: checkpoint, single component, or LoRA."""
    if components["lora"]:
        return "LoRA"

    has_backbone = components["unet"] or components["transformer"]
    has_aux = components["vae"] or components["text_encoder"] or components["text_encoder_2"]

    if has_backbone and has_aux:
        return "Checkpoint"
    if has_backbone:
        return "Backbone"
    if components["vae"] and not has_backbone:
        return "VAE"
    if (components["text_encoder"] or components["text_encoder_2"]) and not has_backbone:
        return "Text Encoder"

    return "Unknown"


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _friendly_encoder_name(raw_name: str) -> str:
    """Map raw text encoder key names to friendly display names."""
    mapping = {
        "clip_l": "CLIP-L (OpenAI ViT-L/14)",
        "clip_g": "CLIP-G (OpenCLIP ViT-bigG)",
        "clip": "CLIP",
        "t5xxl": "T5-XXL",
        "t5": "T5",
        "qwen3_4b": "Qwen3 4B",
        "qwen2_vl": "Qwen2-VL",
        "qwen": "Qwen",
    }
    return mapping.get(raw_name.lower(), raw_name)


def _safe_json_loads(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_training_meta(metadata: dict) -> dict:
    meta = {}

    def first(*keys):
        for k in keys:
            v = metadata.get(k)
            if v not in (None, "", "None"):
                return v
        return None

    train_images = first("ss_num_train_images")
    if train_images is None:
        ds_dirs = _safe_json_loads(metadata.get("ss_dataset_dirs"))
        if isinstance(ds_dirs, dict):
            train_images = sum(
                int(v.get("img_count", 0))
                for v in ds_dirs.values()
                if isinstance(v, dict)
            ) or None
    if train_images is not None:
        meta["train_images"] = str(train_images)

    epochs = first("ss_epoch", "ss_num_epochs")
    if epochs is not None:
        meta["epochs"] = str(epochs)

    steps = first("ss_steps", "ss_max_train_steps")
    if steps is not None:
        meta["steps"] = str(steps)

    resolution = first("ss_resolution", "modelspec.resolution")
    if resolution is not None:
        meta["resolution"] = str(resolution)

    clip_skip = first("ss_clip_skip")
    if clip_skip is not None:
        meta["clip_skip"] = str(clip_skip)

    lr_scheduler = first("ss_lr_scheduler")
    if lr_scheduler is not None:
        meta["lr_scheduler"] = str(lr_scheduler)

    mixed_precision = first("ss_mixed_precision")
    if mixed_precision is not None:
        meta["mixed_precision"] = str(mixed_precision)

    seed = first("ss_seed")
    if seed is not None:
        meta["seed"] = str(seed)

    network_module = first("ss_network_module")
    if network_module is not None:
        meta["network_module"] = str(network_module)

    started = metadata.get("ss_training_started_at")
    finished = metadata.get("ss_training_finished_at")
    if started not in (None, "", "None") and finished not in (None, "", "None"):
        try:
            duration = float(finished) - float(started)
            if duration >= 0:
                meta["training_duration_sec"] = str(int(duration))
        except Exception:
            pass

    software = _safe_json_loads(metadata.get("software"))
    if isinstance(software, dict):
        name = software.get("name")
        ver = software.get("version")
        if name and ver:
            meta["software"] = f"{name} {ver}"
        elif name:
            meta["software"] = str(name)

    training_info = _safe_json_loads(metadata.get("training_info"))
    if isinstance(training_info, dict):
        if training_info.get("step") is not None:
            meta["ti_step"] = str(training_info.get("step"))
        if training_info.get("epoch") is not None:
            meta["ti_epoch"] = str(training_info.get("epoch"))

    tag_freq = _safe_json_loads(metadata.get("ss_tag_frequency"))
    if isinstance(tag_freq, dict):
        tag_count = 0
        for v in tag_freq.values():
            if isinstance(v, dict):
                tag_count += len(v)
        if tag_count:
            meta["tag_count"] = str(tag_count)

    return meta


def _apply_filename_alias_detection(arch: str, filepath: str) -> str:
    """Optional alias detection from filename for SDXL-family off-model names."""
    stem = Path(filepath).stem.lower()
    # Normalize for robust matching across separators/order.
    # "Qwen-Edit", "qwen_edit", "edit qwen" all become token-compatible.
    collapsed = re.sub(r"[^a-z0-9]+", "", stem)
    tokens = set(re.findall(r"[a-z0-9]+", stem))

    def has_token(value: str) -> bool:
        return value in tokens

    def has_all(*values: str) -> bool:
        return all(v in tokens for v in values)

    # These aliases are opt-in and only used as fallback when enabled.
    if ("ilxl" in collapsed or
            "illustrious" in collapsed or
            has_token("illu")):
        return "ILXL"
    if ("pony7" in collapsed or "ponyv7" in collapsed or
            has_all("pony", "v7")):
        return "Pony7"
    if "pdxl" in collapsed or has_token("pony"):
        return "PDXL"
    if has_token("nai"):
        return "NAI"
    if ("qwenedit" in collapsed or "editqwen" in collapsed or
            has_all("qwen", "edit")):
        return "Qwen Edit"
    return arch


def inspect_file(filepath: str, options: dict | None = None) -> dict:
    """
    Analyze a .safetensors file and return structured results.

    Returns a dict with keys:
        filepath, filename, file_size, file_size_friendly,
        tensor_count, total_params, total_params_friendly,
        architecture, arch_details, model_type,
        components, named_text_encoders,
        dtypes (list of {dtype, friendly, bits, count, pct}),
        precision_summary, metadata
    """
    options = options or {}
    allow_filename_alias_detection = bool(options.get("allow_filename_alias_detection", False))

    metadata, tensor_info, file_size = read_safetensors_header(filepath)
    keys = sorted(tensor_info.keys())
    dtypes, total_params, shapes = analyze_tensors(tensor_info)
    components = detect_components(keys)
    arch, arch_details = detect_architecture(keys, shapes, total_params, components, metadata)
    if allow_filename_alias_detection:
        arch = _apply_filename_alias_detection(arch, filepath)
    model_type = classify_model_type(components, arch)
    adapter_type = detect_adapter_type(keys, metadata)
    training_meta = _extract_training_meta(metadata)

    # Build dtype list
    dtype_list = []
    for dtype, count in dtypes.most_common():
        dtype_list.append({
            "dtype": dtype,
            "friendly": DTYPE_FRIENDLY.get(dtype, dtype),
            "bits": DTYPE_BITS.get(dtype, 0),
            "count": count,
            "pct": round(count / len(tensor_info) * 100, 1),
        })

    # Precision summary
    precision_summary = _summarize_dtype_mix(dtypes, len(tensor_info))
    component_dtypes = analyze_component_precisions(tensor_info)
    component_precision_summary = build_component_precision_summary(component_dtypes)
    component_precisions = build_component_precision_map(component_dtypes)
    if precision_summary.startswith("Mixed (") and component_precision_summary:
        precision_display = component_precision_summary
    else:
        precision_display = precision_summary

    # Component booleans (exclude the dict-type text_encoders from simple flags)
    comp_flags = {k: v for k, v in components.items() if k != "text_encoders"}

    # Named text encoders
    named_enc = {}
    for enc_name, enc_count in components.get("text_encoders", {}).items():
        named_enc[_friendly_encoder_name(enc_name)] = enc_count

    # LoRA rank
    lora_rank = arch_details.pop("lora_rank", None)
    detected_adapter = arch_details.pop("adapter_type", None) or adapter_type

    # Extra metadata we can surface
    extra = {}
    if metadata.get("ss_network_dim"):
        extra["training_rank"] = metadata["ss_network_dim"]
    if metadata.get("ss_network_alpha"):
        extra["training_alpha"] = metadata["ss_network_alpha"]
    if metadata.get("ss_lr"):
        extra["learning_rate"] = metadata["ss_lr"]
    if metadata.get("ss_optimizer"):
        extra["optimizer"] = metadata["ss_optimizer"]
    if metadata.get("ss_training_comment"):
        extra["training_comment"] = metadata["ss_training_comment"]
    if metadata.get("ss_output_name"):
        extra["output_name"] = metadata["ss_output_name"]
    if metadata.get("modelspec.title"):
        extra["model_title"] = metadata["modelspec.title"]
    if metadata.get("modelspec.description"):
        extra["model_description"] = metadata["modelspec.description"]
    if metadata.get("modelspec.author"):
        extra["author"] = metadata["modelspec.author"]
    if detected_adapter:
        extra["adapter_type"] = detected_adapter
    for mk, mv in training_meta.items():
        if mk not in extra:
            extra[mk] = mv

    return {
        "filepath": filepath,
        "filename": Path(filepath).name,
        "file_size": file_size,
        "file_size_friendly": format_size(file_size),
        "tensor_count": len(tensor_info),
        "total_params": total_params,
        "total_params_friendly": format_params(total_params),
        "architecture": arch,
        "arch_details": arch_details,
        "model_type": model_type,
        "components": comp_flags,
        "named_text_encoders": named_enc,
        "lora_rank": lora_rank,
        "adapter_type": detected_adapter,
        "training_meta": training_meta,
        "dtypes": dtype_list,
        "precision_summary": precision_summary,
        "component_precision_summary": component_precision_summary,
        "component_precisions": component_precisions,
        "precision_display": precision_display,
        "metadata": metadata,
        "extra": extra,
    }


def print_report(filepath: str, metadata: dict, tensor_info: dict, file_size: int):
    keys = sorted(tensor_info.keys())
    dtypes, total_params, shapes = analyze_tensors(tensor_info)
    components = detect_components(keys)
    arch, arch_details = detect_architecture(keys, shapes, total_params, components, metadata)
    model_type = classify_model_type(components, arch)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SAFETENSORS MODEL INSPECTOR")
    print(sep)

    # File info
    print(f"\n  File:           {Path(filepath).name}")
    print(f"  Path:           {filepath}")
    print(f"  File size:      {format_size(file_size)}")
    print(f"  Tensor count:   {len(tensor_info)}")
    print(f"  Parameters:     {format_params(total_params)} ({total_params:,})")

    # Architecture
    print(f"\n  Architecture:   {arch}")
    print(f"  Model type:     {model_type}")

    if arch_details:
        for k, v in arch_details.items():
            label = k.replace("_", " ").title()
            print(f"  {label + ':':<18}{v}")

    # Components
    print(f"\n  Components detected:")
    comp_labels = {
        "unet": "UNet",
        "transformer": "Transformer / DiT",
        "vae": "VAE",
        "text_encoder": "Text Encoder (CLIP / T5)",
        "text_encoder_2": "Text Encoder 2",
        "lora": "LoRA layers",
    }
    any_found = False
    for key, label in comp_labels.items():
        if key == "text_encoders":
            continue  # handled separately below
        if components[key]:
            print(f"    [x] {label}")
            any_found = True
    # Show named text encoders discovered via text_encoders.{name}.*
    if components["text_encoders"]:
        any_found = True
        for enc_name, enc_count in sorted(components["text_encoders"].items()):
            # Friendly name mapping for common encoder names
            friendly = _friendly_encoder_name(enc_name)
            print(f"    [x] Text Encoder: {friendly} ({enc_count} tensors)")
    if not any_found:
        print(f"    (none of the standard components detected)")

    # Precision
    print(f"\n  Precision / dtype distribution:")
    for dtype, count in dtypes.most_common():
        friendly = DTYPE_FRIENDLY.get(dtype, dtype)
        bits = DTYPE_BITS.get(dtype, "?")
        pct = count / len(tensor_info) * 100
        print(f"    {friendly:<20} {count:>6} tensors  ({pct:.1f}%)  [{bits}-bit]")

    if len(dtypes) == 1:
        only_dtype = list(dtypes.keys())[0]
        print(f"  >> Uniform precision: {DTYPE_FRIENDLY.get(only_dtype, only_dtype)}")
    elif len(dtypes) > 1:
        # Check if it's truly mixed or just a handful of outliers
        dominant_count = dtypes.most_common(1)[0][1]
        dominant_dtype = dtypes.most_common(1)[0][0]
        dominant_pct = dominant_count / len(tensor_info) * 100
        if dominant_pct >= 99.0:
            print(f"  >> Effectively {DTYPE_FRIENDLY.get(dominant_dtype, dominant_dtype)}"
                  f" ({dominant_pct:.1f}%, {len(tensor_info) - dominant_count}"
                  f" outlier tensor(s) in other dtype)")
        else:
            print(f"  >> Mixed precision model")

    # Safetensors metadata
    if metadata:
        print(f"\n  Embedded metadata:")
        for mk, mv in sorted(metadata.items()):
            val_str = str(mv)
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            print(f"    {mk}: {val_str}")

    # Top-level key prefixes (structural overview)
    print(f"\n  Top-level key prefixes (first 20):")
    prefixes = Counter()
    for k in keys:
        prefix = k.split(".")[0]
        if len(k.split(".")) > 1:
            prefix += "." + k.split(".")[1]
        prefixes[prefix] += 1
    for prefix, count in prefixes.most_common(20):
        print(f"    {prefix:<45} {count:>5} tensors")

    print(f"\n{sep}\n")


def generate_modelinfo_dump(filepath: str) -> str:
    """Generate detailed .modelinfo text dump for a single safetensors file."""
    metadata, tensor_info, file_size = read_safetensors_header(filepath)
    keys = sorted(tensor_info.keys())
    _, total_params, shapes = analyze_tensors(tensor_info)

    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"  FILE: {Path(filepath).name}")
    lines.append(f"  Path: {filepath}")
    lines.append(f"  Keys: {len(keys)}    Params: {total_params:,}    Size: {file_size:,} bytes")
    lines.append(sep)

    if metadata:
        lines.append("\n  __metadata__:")
        for mk in sorted(metadata.keys()):
            val = str(metadata[mk])
            if len(val) > 200:
                val = val[:197] + "..."
            lines.append(f"    {mk}: {val}")

    lines.append("\n  Fingerprint substring scan:")
    blob = "\n".join(keys)
    for fp in FINGERPRINTS:
        if fp in blob:
            count = sum(1 for k in keys if fp in k)
            lines.append(f"    [HIT]  {fp:<40} ({count} keys)")

    up_dims = _collect_lora_up_dims(keys, shapes)
    if up_dims:
        lines.append(f"\n  LoRA up dims (target layer sizes): {sorted(up_dims)}")

    lines.append("\n  Top key prefixes (depth 2):")
    prefixes = Counter()
    for k in keys:
        parts = k.split(".")
        p = parts[0]
        if len(parts) > 1:
            p += "." + parts[1]
        prefixes[p] += 1
    for p, c in prefixes.most_common(25):
        lines.append(f"    {p:<55} {c:>5}")

    lines.append(f"\n  All tensor keys ({len(keys)}):")
    for k in keys:
        s = shapes.get(k, [])
        d = tensor_info[k].get("dtype", "?")
        lines.append(f"    {k}  {s}  [{d}]")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _iter_safetensors_paths(targets: Iterable[str], recursive: bool) -> list[str]:
    found = []
    seen = set()
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() == ".safetensors":
                s = str(p.resolve())
                if s not in seen:
                    seen.add(s)
                    found.append(s)
            continue
        if p.is_dir():
            it = p.rglob("*.safetensors") if recursive else p.glob("*.safetensors")
            for fp in it:
                s = str(fp.resolve())
                if s not in seen:
                    seen.add(s)
                    found.append(s)
            continue
    return found


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_model.py",
        description="Inspect .safetensors model files from file and folder targets.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="File(s) or folder(s) to inspect",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subfolders for directory targets",
    )
    parser.add_argument(
        "--allow-filename-alias-detection",
        action="store_true",
        help="Allow filename token fallback aliases for SDXL-family names",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output (single object for one file, list for many)",
    )
    parser.add_argument(
        "--dump-keys",
        action="store_true",
        help="Print detailed key-dump report text for each file",
    )
    parser.add_argument(
        "--write-modelinfo",
        action="store_true",
        help="Write .modelinfo files next to each inspected model",
    )
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    paths = _iter_safetensors_paths(args.targets, args.recursive)
    if not paths:
        print("No .safetensors files found in provided targets.", file=sys.stderr)
        return 1

    if args.dump_keys:
        for fp in paths:
            try:
                print(generate_modelinfo_dump(fp))
            except Exception as e:
                print(f"[ERROR] {fp}: {e}", file=sys.stderr)
        return 0

    results = []
    for fp in paths:
        try:
            info = inspect_file(
                fp,
                options={
                    "allow_filename_alias_detection": args.allow_filename_alias_detection
                },
            )
            results.append(info)
            if args.write_modelinfo:
                out_path = fp + ".modelinfo"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(generate_modelinfo_dump(fp))
        except Exception as e:
            print(f"[ERROR] {fp}: {e}", file=sys.stderr)

    if not results:
        return 1

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0], indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    for fp in paths:
        try:
            metadata, tensor_info, file_size = read_safetensors_header(fp)
            print_report(fp, metadata, tensor_info, file_size)
        except Exception as e:
            print(f"[ERROR] {fp}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
