import torch

from ltx_core.components.patchifiers import AudioLatentShape
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import AudioLatentTools
from ltx_core.types import LatentState


class AudioConditionBySilentTailToken(ConditioningItem):
    """Append silent (zero-filled) audio tokens to the tail of an audio state.

    Mirrors training-side ii2v behavior where, when the end-frame of the video
    is conditioned, a single silent audio latent token is appended at the tail
    so the audio timeline extends by one latent step. Keeping this alignment at
    inference time preserves train/infer consistency for keyframe-interpolation
    style generation (first + last frame).

    Args:
        num_tokens: Number of silent tokens to append (default 1 to match the
            training strategy ``_prepare_audio_inputs`` logic).
        strength: Conditioning strength in ``[0, 1]``; 1.0 means the token is
            fully frozen (denoise_mask=0), mirroring training where appended
            silent tokens carry ``loss_mask=False``.
    """

    def __init__(self, num_tokens: int = 1, strength: float = 1.0):
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        self.num_tokens = int(num_tokens)
        self.strength = float(strength)

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: AudioLatentTools,
    ) -> LatentState:
        if not isinstance(latent_tools, AudioLatentTools):
            raise TypeError(
                "AudioConditionBySilentTailToken requires AudioLatentTools, got "
                f"{type(latent_tools).__name__}"
            )

        device = latent_state.latent.device
        dtype = latent_state.latent.dtype
        batch_size, _, token_dim = latent_state.latent.shape

        extra_latent = torch.zeros(
            batch_size, self.num_tokens, token_dim, device=device, dtype=dtype
        )

        extra_mask_shape = list(latent_state.denoise_mask.shape)
        extra_mask_shape[1] = self.num_tokens
        extra_denoise = torch.full(
            extra_mask_shape,
            fill_value=1.0 - self.strength,
            device=device,
            dtype=latent_state.denoise_mask.dtype,
        )

        base_shape = latent_tools.target_shape
        extended_shape = AudioLatentShape(
            batch=base_shape.batch,
            channels=base_shape.channels,
            frames=base_shape.frames + self.num_tokens,
            mel_bins=base_shape.mel_bins,
        )
        extended_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=extended_shape, device=device,
        )
        tail_positions = extended_coords[:, :, -self.num_tokens:, :].to(latent_state.positions.dtype)
        if tail_positions.shape[0] != batch_size:
            tail_positions = tail_positions.expand(batch_size, *tail_positions.shape[1:])

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=self.num_tokens,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, extra_latent], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, extra_denoise], dim=1),
            positions=torch.cat([latent_state.positions, tail_positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, extra_latent], dim=1),
            attention_mask=new_attention_mask,
        )
