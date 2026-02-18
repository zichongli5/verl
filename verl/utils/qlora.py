# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
from typing import Any

import torch


def _to_dict_if_possible(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    to_dict_fn = getattr(value, "to_dict", None)
    if callable(to_dict_fn):
        try:
            output = to_dict_fn()
            if isinstance(output, dict):
                return output
        except Exception:
            return None
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return None


def get_quantization_config(model_config: Any) -> Any | None:
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get("quantization_config", None)
    return getattr(model_config, "quantization_config", None)


def get_quantization_config_dict(model_config: Any) -> dict[str, Any] | None:
    return _to_dict_if_possible(get_quantization_config(model_config))


def is_quantized_model_config(model_config: Any) -> bool:
    quantization_config = get_quantization_config(model_config)
    if quantization_config is None:
        return False
    quantization_config_dict = _to_dict_if_possible(quantization_config)
    if quantization_config_dict is None:
        return True
    return len(quantization_config_dict) > 0


def is_qlora_mode(is_lora_enabled: bool, model_config: Any) -> bool:
    return bool(is_lora_enabled and is_quantized_model_config(model_config))


def get_quantization_model_init_kwargs(model_config: Any) -> dict[str, Any]:
    quantization_config = get_quantization_config(model_config)
    if quantization_config is None:
        return {}
    quantization_config_dict = _to_dict_if_possible(quantization_config)
    if quantization_config_dict is not None and len(quantization_config_dict) == 0:
        return {}
    # HF can store quantization_config as a plain dict on AutoConfig.
    # from_pretrained expects a QuantizationConfigMixin instance here when the model
    # config resolves to a concrete quantization config class (e.g. CompressedTensorsConfig).
    # Normalize dict -> class instance to avoid class-mismatch errors.
    if isinstance(quantization_config, dict):
        try:
            from transformers.quantizers.auto import AutoQuantizationConfig

            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
        except Exception:
            # Best effort: keep original behavior if transformers quantization APIs
            # are unavailable or the dict is from an unsupported/custom method.
            pass
    return {"quantization_config": quantization_config}


def maybe_prepare_model_for_qlora(
    module,
    *,
    enable_gradient_checkpointing: bool,
    logger: logging.Logger | None = None,
):
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        if logger is not None:
            logger.warning(
                "QLoRA detected but `prepare_model_for_kbit_training` is unavailable in the installed PEFT version."
            )
        return module, False

    kwargs = {}
    signature = inspect.signature(prepare_model_for_kbit_training)
    if "use_gradient_checkpointing" in signature.parameters:
        kwargs["use_gradient_checkpointing"] = enable_gradient_checkpointing
    if "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    try:
        module = prepare_model_for_kbit_training(module, **kwargs)
    except TypeError:
        kwargs.pop("gradient_checkpointing_kwargs", None)
        module = prepare_model_for_kbit_training(module, **kwargs)
    return module, True


def is_peft_lora_injection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "target module" in message and ("not supported" in message or "not found" in message):
        return True
    if "target modules" in message and ("not found" in message or "could not be found" in message):
        return True
    return False


def is_missing_weight_attr_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "has no attribute" in message and "weight" in message


def _infer_weight_proxy_tensor(module) -> torch.Tensor | torch.nn.Parameter | None:
    for attr_name in ("weight", "qweight", "w_q", "kernel"):
        attr_value = getattr(module, attr_name, None)
        if isinstance(attr_value, (torch.Tensor, torch.nn.Parameter)):
            return attr_value

    for name, param in module.named_parameters(recurse=False):
        if "weight" in name and isinstance(param, torch.nn.Parameter):
            return param

    for name, buffer in module.named_buffers(recurse=False):
        if "weight" in name and isinstance(buffer, torch.Tensor):
            return buffer

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is None or out_features is None:
        in_features = getattr(module, "infeatures", None)
        out_features = getattr(module, "outfeatures", None)

    if in_features is None or out_features is None:
        return None

    in_features = int(in_features)
    out_features = int(out_features)

    dtype = None
    for _, param in module.named_parameters(recurse=False):
        dtype = param.dtype
        break
    if dtype is None:
        for _, buffer in module.named_buffers(recurse=False):
            dtype = buffer.dtype
            break
    if dtype is None:
        dtype = torch.float16

    return torch.empty((out_features, in_features), device="meta", dtype=dtype)


def attach_weight_proxies_for_qlora(module, logger: logging.Logger | None = None) -> int:
    patched = 0
    for _, child in module.named_modules():
        if hasattr(child, "weight"):
            continue
        weight_proxy = _infer_weight_proxy_tensor(child)
        if weight_proxy is None:
            continue
        try:
            setattr(child, "weight", weight_proxy)
            patched += 1
        except Exception:
            continue

    if patched > 0 and logger is not None:
        logger.info("QLoRA: attached weight proxy to %d modules for PEFT compatibility.", patched)
    return patched


def validate_rollout_load_format_for_qlora(load_format: str) -> None:
    if load_format is not None and "dummy" in str(load_format):
        raise ValueError(
            "QLoRA requires a preloaded rollout base model. "
            "`rollout.load_format` cannot contain `dummy`; use a real load format like `auto` or `safetensors`."
        )
