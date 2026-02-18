#!/usr/bin/env bash
set -xeuo pipefail

# Muon settings
muon_lr=${MUON_LR:-1e-6}
muon_momentum=${MUON_MOMENTUM:-0.95}
muon_nesterov=${MUON_NESTEROV:-true}
muon_ns_steps=${MUON_NS_STEPS:-5}

# Parameter grouping settings for Muon split.
muon_include_patterns=${MUON_INCLUDE_PATTERNS:-"[]"}
muon_exclude_patterns=${MUON_EXCLUDE_PATTERNS:-'["embed_tokens","lm_head"]'}

# Keep non-Muon fallback defaults aligned with train_qwen2.5_7B_base_dapo.sh.
non_muon_optimizer=${NON_MUON_OPTIMIZER:-"AdamW"}
non_muon_optimizer_impl=${NON_MUON_OPTIMIZER_IMPL:-"torch.optim"}
non_muon_lr=${NON_MUON_LR:-1e-6}
non_muon_weight_decay=${NON_MUON_WEIGHT_DECAY:-0.1}
non_muon_betas=${NON_MUON_BETAS:-"[0.9,0.999]"}

ACTOR_OPTIMIZER=${ACTOR_OPTIMIZER:-"Muon"} \
ACTOR_OPTIMIZER_IMPL=${ACTOR_OPTIMIZER_IMPL:-"torch.optim"} \
bash recipe/zli/train_qwen2.5_7B_base_dapo.sh \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.optim.lr="${muon_lr}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.momentum="${muon_momentum}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.nesterov="${muon_nesterov}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.ns_steps="${muon_ns_steps}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.adjust_lr_fn="original" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.mode=muon_2d_adamw \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.muon_include_patterns="${muon_include_patterns}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.muon_exclude_patterns="${muon_exclude_patterns}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.non_muon_optimizer="${non_muon_optimizer}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.non_muon_optimizer_impl="${non_muon_optimizer_impl}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.non_muon_lr="${non_muon_lr}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.non_muon_weight_decay="${non_muon_weight_decay}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_group_config.non_muon_betas="${non_muon_betas}" \
    "$@"
