from __future__ import annotations

from typing import Any, Iterable, List, Optional


def is_vision_module_name(name: str) -> bool:
    """Return whether a qualified module name belongs to a vision tower."""

    parts = str(name).split(".")
    return "vision_tower" in parts or "vision_model" in parts


def resolve_text_lora_target_modules(
    model: Any,
    requested_suffixes: Iterable[str],
) -> List[str]:
    """Resolve PEFT targets to exact, non-vision module names.

    PEFT list targets normally use suffix matching.  On Gemma 3 that also
    matches q/k/v projections inside the vision tower.  Merely freezing those
    adapters is insufficient: PRISM rebases the trained text adapters from r
    to 2r before saving, leaving the frozen vision adapters at r and producing
    an internally inconsistent adapter.  Passing exact text module names keeps
    vision adapters out of the model from the start.
    """

    suffixes = [str(value).strip() for value in requested_suffixes if str(value).strip()]
    if not suffixes:
        raise ValueError("requested_suffixes must not be empty")
    found_by_suffix = {suffix: 0 for suffix in suffixes}
    resolved: List[str] = []
    for name, _module in model.named_modules():
        if not name or is_vision_module_name(name):
            continue
        for suffix in suffixes:
            if name == suffix or name.endswith(f".{suffix}"):
                resolved.append(name)
                found_by_suffix[suffix] += 1
                break
    missing = [suffix for suffix, count in found_by_suffix.items() if count == 0]
    if missing:
        raise ValueError(
            "No non-vision modules matched requested LoRA targets: "
            + ", ".join(missing)
        )
    return resolved


def is_multimodal_causal_lm_config(config: Any) -> bool:
    """Return whether a causal-LM checkpoint needs a multimodal auto class.

    Gemma 3 checkpoints at 4B and above use ``Gemma3Config`` and include a
    vision tower even for text-only fine-tuning.  Loading those checkpoints via
    the text-only auto class is version-dependent and fails on several supported
    Transformers releases, so select the architecture explicitly.
    """

    model_type = str(getattr(config, "model_type", "")).lower()
    return model_type == "gemma3" and getattr(config, "vision_config", None) is not None


def _multimodal_model_class():
    import transformers

    for name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "Gemma3ForConditionalGeneration",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "This checkpoint is multimodal, but the installed Transformers version "
        "does not provide a compatible model class. Gemma 3 requires "
        "transformers>=4.50."
    )


def load_base_model(
    model_id: str,
    *,
    revision: Optional[str] = None,
    torch_dtype=None,
    device_map=None,
    trust_remote_code: bool = True,
    **kwargs,
):
    """Load a text-generating base model with stable Gemma 3 handling."""

    from transformers import AutoConfig, AutoModelForCausalLM

    common = {
        "revision": revision,
        "trust_remote_code": bool(trust_remote_code),
    }
    # Older releases may not accept an explicit None revision in every helper.
    common = {key: value for key, value in common.items() if value is not None}
    config = AutoConfig.from_pretrained(model_id, **common)
    model_cls = _multimodal_model_class() if is_multimodal_causal_lm_config(config) else AutoModelForCausalLM

    load_kwargs = dict(common)
    load_kwargs.update(kwargs)
    load_kwargs["config"] = config
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    try:
        return model_cls.from_pretrained(model_id, **load_kwargs)
    except TypeError as exc:
        # Transformers 5 renamed torch_dtype to dtype.  Retry only for that
        # compatibility boundary; preserve all other loading failures.
        if "torch_dtype" not in load_kwargs:
            raise
        retry = dict(load_kwargs)
        retry["dtype"] = retry.pop("torch_dtype")
        try:
            return model_cls.from_pretrained(model_id, **retry)
        except TypeError:
            raise exc


def resolved_model_revision(model: Any) -> Optional[str]:
    """Best-effort Hugging Face commit recorded on a loaded model config."""

    config = getattr(model, "config", None)
    value = getattr(config, "_commit_hash", None)
    return str(value) if value else None
