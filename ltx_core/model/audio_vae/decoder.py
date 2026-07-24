import torch
import torch.nn.functional as F
from einops import rearrange

from ltx_core.model.audio_vae.causal_conv_2d import CausalConv2d
from ltx_core.model.audio_vae.ops import PerChannelStatistics
from ltx_core.model.audio_vae.resnet import ResnetBlock
from ltx_core.model.audio_vae.upsample import build_upsampling_path
from ltx_core.model.common.normalization import PixelNorm

def build_mid_block() -> torch.nn.Module:
    mid = torch.nn.Module()
    mid.block_1 = ResnetBlock(512, 512)
    mid.attn_1 = torch.nn.Identity()
    mid.block_2 = ResnetBlock(512, 512)
    return mid

class AudioDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.per_channel_statistics = PerChannelStatistics(latent_channels=128)
        self.conv_in = CausalConv2d(8, 512, 3)
        self.non_linearity = torch.nn.SiLU()
        self.mid = build_mid_block()
        self.up, final_channels = build_upsampling_path()
        self.norm_out = PixelNorm(eps=1e-6)
        self.conv_out = CausalConv2d(final_channels, 2, 3)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, mel_bins = sample.shape
        sample = rearrange(sample, "b c t f -> b t (c f)")
        sample = self.per_channel_statistics.un_normalize(sample)
        sample = rearrange(sample, "b t (c f) -> b c t f", c=channels, f=mel_bins)

        features = self.conv_in(sample)
        features = self.mid.block_1(features)
        features = self.mid.attn_1(features)
        features = self.mid.block_2(features)
        for level in reversed(range(3)):
            stage = self.up[level]
            for block in stage.block:
                features = block(features)
            if level:
                features = stage.upsample(features)
        output = self.conv_out(self.non_linearity(self.norm_out(features)))

        target_frames = max(frames * 4 - 3, 1)
        output = output[:, :2, : min(output.shape[2], target_frames), :64]
        time_padding = target_frames - output.shape[2]
        frequency_padding = 64 - output.shape[3]
        if time_padding > 0 or frequency_padding > 0:
            output = F.pad(output, (0, max(frequency_padding, 0), 0, max(time_padding, 0)))
        return output[:, :2, :target_frames, :64]
