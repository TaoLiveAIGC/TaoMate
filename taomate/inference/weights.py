import torch
import torch.nn as nn

from taomate.runtime_support.models.text_encoder import create_text_encoder_wrapper
from taomate.runtime_support.models.vae import create_vae_wrappers

class TextConditioner(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, prompt: str) -> dict[str, torch.Tensor]:
        return self.encoder([prompt])

def load_decoder_bundles(
    checkpoint_path: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module]:
    return create_vae_wrappers(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        dtype=dtype,
    )

def load_text_conditioner(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> TextConditioner:
    encoder = create_text_encoder_wrapper(
        checkpoint_path=checkpoint_path,
        gemma_path=gemma_path,
        device=device,
        dtype=dtype,
        place_on_device=True,
    )
    return TextConditioner(encoder.eval())
