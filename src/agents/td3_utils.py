"""Shared TD3 pieces for the classical and hybrid agents."""
import torch


def smoothed_target_actions(
    actor_target: torch.nn.Module,
    next_states: torch.Tensor,
    noise_std: float,
    noise_clip: float,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Target-policy actions with TD3 target policy smoothing.

    Adds clipped Gaussian noise to the target actor's actions and clamps the
    result to the action bounds (Fujimoto et al. 2018). With ``noise_std`` of
    0 this is the plain target actor, as in the v1 code.
    """
    actions = actor_target(next_states)
    if noise_std > 0.0:
        noise = (torch.randn_like(actions) * noise_std).clamp(-noise_clip, noise_clip)
        actions = (actions + noise).clamp(low, high)
    return actions
