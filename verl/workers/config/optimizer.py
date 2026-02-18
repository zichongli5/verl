# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import MISSING, OmegaConf

from verl.base_config import BaseConfig

__all__ = ["OptimizerConfig", "FSDPOptimizerConfig", "McoreOptimizerConfig", "build_optimizer"]


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


class CompositeOptimizer(torch.optim.Optimizer):
    """A thin wrapper that steps multiple optimizers together."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if len(optimizers) == 0:
            raise ValueError("`optimizers` must not be empty.")
        self.optimizers = optimizers
        # `Optimizer` requires a valid parameter list for initialization.
        super().__init__(optimizers[0].param_groups, defaults={})
        self._rebuild_views()

    def _rebuild_views(self):
        self.param_groups = []
        self.state = {}
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)
            self.state.update(optimizer.state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizers[0].step(closure=closure)
        for optimizer in self.optimizers[1:]:
            optimizer.step()
        self._rebuild_views()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "composite_optimizer": True,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        # Backward compatibility: allow loading a single optimizer state dict.
        if isinstance(state_dict, dict) and state_dict.get("composite_optimizer", False):
            optimizer_state_dicts = state_dict.get("optimizers", [])
            if len(optimizer_state_dicts) != len(self.optimizers):
                raise ValueError(
                    f"Expected {len(self.optimizers)} optimizer state dicts, got {len(optimizer_state_dicts)}."
                )
            for optimizer, optimizer_state_dict in zip(self.optimizers, optimizer_state_dicts, strict=True):
                optimizer.load_state_dict(optimizer_state_dict)
        else:
            self.optimizers[0].load_state_dict(state_dict)
        self._rebuild_views()


def _to_python_dict(obj):
    if obj is None:
        return None
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return dict(obj)


def _build_optimizer_args(
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    override_config: Optional[dict],
):
    optimizer_args = {"lr": lr, "weight_decay": weight_decay}
    optimizer_name_lower = optimizer_name.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = tuple(betas)
    if override_config is not None:
        optimizer_args.update(override_config)
    return optimizer_args


def _load_optimizer_cls(optimizer_impl: str, optimizer_name: str):
    import importlib

    try:
        module = importlib.import_module(optimizer_impl)
        optimizer_cls = getattr(module, optimizer_name)
        return optimizer_cls, module
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{optimizer_name}' not found in module '{optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e


def _match_patterns(name: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def _split_muon_and_non_muon_params(named_parameters: list[tuple[str, torch.nn.Parameter]], group_cfg: dict):
    include_patterns = list(group_cfg.get("muon_include_patterns", []))
    exclude_patterns = list(group_cfg.get("muon_exclude_patterns", []))

    muon_params = []
    non_muon_params = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        use_muon = param.ndim >= 2
        if use_muon and include_patterns:
            use_muon = _match_patterns(name, include_patterns)
        if use_muon and exclude_patterns and _match_patterns(name, exclude_patterns):
            use_muon = False

        if use_muon:
            muon_params.append(param)
        else:
            non_muon_params.append(param)

    return muon_params, non_muon_params


def _normalize_named_parameters(parameters):
    named_parameters = []
    for index, item in enumerate(parameters):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            name, param = item
        else:
            name, param = f"param_{index}", item
        named_parameters.append((name, param))
    return named_parameters


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    named_parameters = _normalize_named_parameters(parameters)
    optimizer_override = _to_python_dict(config.override_optimizer_config) or {}
    group_cfg = _to_python_dict(optimizer_override.pop("optimizer_group_config", None))

    optimizer_name = config.optimizer
    optimizer_impl = config.optimizer_impl
    optimizer_name_lower = optimizer_name.lower()

    # Muon only supports 2D tensors. If user asks for split mode, optimize 2D tensors
    # with Muon and the rest with a fallback optimizer (AdamW by default).
    if optimizer_name_lower == "muon" and group_cfg is not None:
        mode = group_cfg.get("mode", "muon_2d_adamw")
        if mode != "muon_2d_adamw":
            raise ValueError(
                f"Unsupported optimizer_group_config.mode={mode!r}. Only 'muon_2d_adamw' is supported currently."
            )

        muon_params, non_muon_params = _split_muon_and_non_muon_params(named_parameters, group_cfg)
        if len(muon_params) == 0:
            n_trainable = sum(1 for _, param in named_parameters if param.requires_grad)
            n_trainable_2d = sum(1 for _, param in named_parameters if param.requires_grad and param.ndim == 2)
            raise ValueError(
                "No parameters selected for Muon. "
                f"trainable={n_trainable}, trainable_2d={n_trainable_2d}. "
                "Adjust include/exclude patterns or model selection. "
                "If using FSDP with flattened params, set "
                "`actor_rollout_ref.actor.fsdp_config.use_orig_params=True`."
            )

        muon_group_overrides = _to_python_dict(group_cfg.get("muon_group_overrides", {})) or {}
        muon_group = [{"params": muon_params, **muon_group_overrides}]

        muon_optimizer_args = _build_optimizer_args(
            optimizer_name=optimizer_name,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            override_config=optimizer_override,
        )
        muon_optimizer_cls, _ = _load_optimizer_cls(optimizer_impl=optimizer_impl, optimizer_name=optimizer_name)
        muon_optimizer = muon_optimizer_cls(muon_group, **muon_optimizer_args)

        if len(non_muon_params) == 0:
            return muon_optimizer

        non_muon_optimizer_name = group_cfg.get("non_muon_optimizer", "AdamW")
        non_muon_optimizer_impl = group_cfg.get("non_muon_optimizer_impl", "torch.optim")
        non_muon_override = _to_python_dict(group_cfg.get("non_muon_override_optimizer_config", {})) or {}
        non_muon_betas = tuple(group_cfg.get("non_muon_betas", config.betas))
        non_muon_lr = float(group_cfg.get("non_muon_lr", config.lr))
        non_muon_weight_decay = float(group_cfg.get("non_muon_weight_decay", config.weight_decay))
        non_muon_group_overrides = _to_python_dict(group_cfg.get("non_muon_group_overrides", {})) or {}
        non_muon_group = [{"params": non_muon_params, **non_muon_group_overrides}]

        non_muon_optimizer_args = _build_optimizer_args(
            optimizer_name=non_muon_optimizer_name,
            lr=non_muon_lr,
            weight_decay=non_muon_weight_decay,
            betas=non_muon_betas,
            override_config=non_muon_override,
        )
        non_muon_optimizer_cls, _ = _load_optimizer_cls(
            optimizer_impl=non_muon_optimizer_impl, optimizer_name=non_muon_optimizer_name
        )
        non_muon_optimizer = non_muon_optimizer_cls(non_muon_group, **non_muon_optimizer_args)

        return CompositeOptimizer([muon_optimizer, non_muon_optimizer])

    base_parameters = [param for _, param in named_parameters]
    if optimizer_name_lower == "muon":
        invalid_muon_params = [name for name, param in named_parameters if param.requires_grad and param.ndim != 2]
        if invalid_muon_params:
            raise ValueError(
                "Muon only supports 2D parameters. Configure split mode with:\n"
                "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.mode=muon_2d_adamw"
            )

    optimizer_args = _build_optimizer_args(
        optimizer_name=optimizer_name,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        override_config=optimizer_override,
    )
    optimizer_cls, _ = _load_optimizer_cls(optimizer_impl=optimizer_impl, optimizer_name=optimizer_name)
    return optimizer_cls(base_parameters, **optimizer_args)
