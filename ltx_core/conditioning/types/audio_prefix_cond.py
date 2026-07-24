import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import AudioLatentTools, LatentTools
from ltx_core.types import LatentState


class AudioConditionByPrefixLatent(ConditioningItem):
    """Replace the first T audio latent tokens with provided clean latents."""

    def __init__(self, latent: torch.Tensor, strength: float | torch.Tensor = 1.0):
        if latent.dim() != 4:
            raise ValueError(f"latent must be [B, C, T, F], got {tuple(latent.shape)}")
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        if not isinstance(latent_tools, AudioLatentTools):
            raise TypeError(
                "AudioConditionByPrefixLatent requires AudioLatentTools, got "
                f"{type(latent_tools).__name__}"
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        t_prefix = min(tokens.shape[1], latent_state.latent.shape[1])
        if t_prefix <= 0:
            return latent_state

        updated = latent_state.clone()
        cast_tokens = tokens[:, :t_prefix].to(updated.latent.dtype)
        updated.latent[:, :t_prefix] = cast_tokens
        updated.clean_latent[:, :t_prefix] = cast_tokens

        if isinstance(self.strength, torch.Tensor):
            strength = self.strength.to(
                device=updated.denoise_mask.device,
                dtype=updated.denoise_mask.dtype,
            )
            if strength.dim() == 1:
                strength = strength.view(1, -1, 1)
            if strength.shape[1] != t_prefix:
                raise ValueError(
                    f"strength tensor second dim must be {t_prefix}, got {strength.shape[1]}"
                )
            updated.denoise_mask[:, :t_prefix] = 1.0 - strength[:, :t_prefix]
        else:
            updated.denoise_mask[:, :t_prefix] = 1.0 - float(self.strength)

        return updated
