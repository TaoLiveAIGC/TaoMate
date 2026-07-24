"""Streaming Video Decoder for LTX-2 VAE.

Implements conv_cache-based streaming temporal decode, following the same
approach as goku's fengdavae_v3 (CogVideoX architecture).

Core mechanism
--------------
Each ``CausalConv3d`` layer caches the last ``kernel_size - 1`` temporal
frames between batches.  ``DepthToSpaceUpsample`` conditionally skips
the first-frame removal on non-first batches.  Together these produce
**mathematically identical** results to a full non-streaming decode
(with PixelNorm; minor normalisation-statistic differences may appear
when GroupNorm is used).

This file does **not** modify any existing LTX-2 source.  It uses
runtime monkey-patching to add streaming capabilities to the stock
``VideoDecoder``.

Usage example::

    from ltx_core.model.video_vae.streaming_decode import StreamingVideoDecoder

    streaming = StreamingVideoDecoder(decoder, num_latent_frames_batch_size=2)
    for chunk in streaming.streaming_decode(latent):
        process(chunk)          # chunk: [f, h, w, c] uint8 on CPU

    # or with spatial tiling + temporal streaming:
    for chunk in streaming.tiled_streaming_decode(latent, tile_h, tile_w):
        process(chunk)
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from ltx_core.model.video_vae.convolution import CausalConv3d
from ltx_core.model.video_vae.ops import unpatchify
from ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample
from ltx_core.model.video_vae.video_vae import VideoDecoder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# StreamingVideoDecoder
# ═══════════════════════════════════════════════════════════════════════════


class StreamingVideoDecoder:
    """Wrap :class:`VideoDecoder` with conv-cache-based temporal streaming.

    Following the goku fengdavae_v3 pattern:

    * ``CausalConv3d`` layers cache the last ``(kernel_size - 1)`` frames
      between temporal batches (replacing the first-frame-repeat padding).
    * ``DepthToSpaceUpsample`` removes the first frame **only** on the very
      first batch (matching the non-streaming behaviour).
    * The decoder is monkey-patched **only** inside a
      :meth:`streaming_context` and restored on exit.

    Parameters
    ----------
    decoder : VideoDecoder
        The stock LTX-2 decoder instance.
    num_latent_frames_batch_size : int
        Number of latent frames per temporal batch.  ``2`` is recommended
        (matches the value used by goku's CogVideoX VAE).

    Notes
    -----
    * With ``PixelNorm`` (the LTX-2 default) the streaming output is
      **bit-exact** vs. full decode.  With ``GroupNorm`` there may be
      imperceptible normalisation-statistic differences because GroupNorm
      computes statistics across the temporal dimension.
    * Spatial noise injection (``inject_noise``) uses a fresh random sample
      per batch, so passing a deterministic ``generator`` will yield
      slightly different noise patterns compared to full decode.  The
      structural output (from learned weights) is unaffected.
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        num_latent_frames_batch_size: int = 2,
    ) -> None:
        self.decoder = decoder
        self.num_latent_frames_batch_size = num_latent_frames_batch_size

        if not decoder.causal:
            logger.warning(
                "VideoDecoder.causal is False, but streaming decode REQUIRES "
                "causal (left-only) padding for conv_cache to work.  Causal "
                "mode will be forced to True during streaming.  Output will "
                "be mathematically equivalent to full decode with causal=True "
                "(minor temporal context shift vs causal=False)."
            )

        # --- internal state ---
        self._conv_cache_store: Dict[int, Optional[torch.Tensor]] = {}
        self._original_forwards: Dict[int, Any] = {}
        self._is_first_batch: bool = True
        self._patched: bool = False

    # ── Patching / Unpatching ───────────────────────────────────────────

    def _patch(self) -> None:
        """Monkey-patch CausalConv3d & DepthToSpaceUpsample for caching."""
        if self._patched:
            return
        for module in self.decoder.modules():
            if isinstance(module, CausalConv3d):
                self._patch_causal_conv(module)
            elif isinstance(module, DepthToSpaceUpsample) and module.stride[0] == 2:
                self._patch_depth_to_space(module)
        self._patched = True

    def _unpatch(self) -> None:
        """Restore every patched module's original ``forward``."""
        if not self._patched:
            return
        for module in self.decoder.modules():
            mid = id(module)
            if mid in self._original_forwards:
                module.forward = self._original_forwards[mid]
        self._original_forwards.clear()
        self._conv_cache_store.clear()
        self._is_first_batch = True
        self._patched = False

    # ── CausalConv3d cache patch ────────────────────────────────────────

    def _patch_causal_conv(self, module: CausalConv3d) -> None:
        """Replace ``CausalConv3d.forward`` with a cache-aware version.

        On the first call for a given module (no cache), behaviour is
        identical to the original (repeat first frame).  On subsequent
        calls the cached ``(kernel_size - 1)`` frames replace the
        first-frame repeat, providing real temporal context.
        """
        mid = id(module)
        self._original_forwards[mid] = module.forward

        store = self._conv_cache_store
        tk = module.time_kernel_size
        conv: nn.Conv3d = module.conv

        def cached_forward(x: torch.Tensor, causal: bool = True) -> torch.Tensor:
            if causal and tk > 1:
                if mid in store and store[mid] is not None:
                    cached = store[mid].to(device=x.device, dtype=x.dtype)
                    x_padded = torch.cat([cached, x], dim=2)
                else:
                    pad = x[:, :, :1, :, :].repeat(1, 1, tk - 1, 1, 1)
                    x_padded = torch.cat([pad, x], dim=2)
                # Save the last (tk-1) frames as cache for the next batch.
                store[mid] = x_padded[:, :, -(tk - 1) :].clone()
                return conv(x_padded)
            else:
                # Non-causal: symmetric padding, no caching.
                if not causal:
                    half = (tk - 1) // 2
                    first_pad = x[:, :, :1, :, :].repeat(1, 1, half, 1, 1)
                    last_pad = x[:, :, -1:, :, :].repeat(1, 1, half, 1, 1)
                    x = torch.concatenate((first_pad, x, last_pad), dim=2)
                return conv(x)

        module.forward = cached_forward  # type: ignore[method-assign]

    # ── DepthToSpaceUpsample first-frame patch ──────────────────────────

    def _patch_depth_to_space(self, module: DepthToSpaceUpsample) -> None:
        """Patch ``DepthToSpaceUpsample.forward`` for streaming.

        The stock implementation unconditionally removes the first temporal
        frame after depth-to-space rearrangement (``x[:, :, 1:, ...]``).
        In streaming mode this removal must happen **only** on the first
        batch; subsequent batches receive real temporal context from the
        conv cache, so the first rearranged frame is legitimate content.
        """
        mid = id(module)
        self._original_forwards[mid] = module.forward

        streaming = self
        stride = module.stride
        residual = module.residual
        out_ch_factor = module.out_channels_reduction_factor

        def patched_forward(x: torch.Tensor, causal: bool = True) -> torch.Tensor:
            x_in: Optional[torch.Tensor] = None
            if residual:
                x_in = rearrange(
                    x,
                    "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                    p1=stride[0],
                    p2=stride[1],
                    p3=stride[2],
                )
                num_repeat = math.prod(stride) // out_ch_factor
                x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
                if streaming._is_first_batch:
                    x_in = x_in[:, :, 1:, :, :]

            x = module.conv(x, causal=causal)
            x = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=stride[0],
                p2=stride[1],
                p3=stride[2],
            )
            if streaming._is_first_batch:
                x = x[:, :, 1:, :, :]

            if residual and x_in is not None:
                x = x + x_in

            return x

        module.forward = patched_forward  # type: ignore[method-assign]

    # ── Cache management ────────────────────────────────────────────────

    def reset_cache(self) -> None:
        """Clear all conv caches for a new decode session."""
        self._conv_cache_store.clear()
        self._is_first_batch = True

    @contextmanager
    def streaming_context(self):
        """Context manager: patch on enter, unpatch on exit."""
        self._patch()
        self.reset_cache()
        try:
            yield self
        finally:
            self._unpatch()

    # ── Internal: run decoder blocks ────────────────────────────────────

    def _forward_decoder_blocks(
        self,
        sample: torch.Tensor,
        scaled_timestep: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Run the decoder's internal blocks on a temporal sub-batch.

        This replicates the logic of ``VideoDecoder.forward`` but **without**
        the noise injection and denormalisation steps (those are done once on
        the full latent tensor before streaming begins).

        .. important::

           **Causal mode is always forced to True** regardless of
           ``decoder.causal``.  The conv-cache mechanism requires causal
           (left-only) padding so that each batch receives real temporal
           context from the previous batch's cache.  With symmetric padding
           (``causal=False``), the cache is never populated and each batch
           is decoded independently, producing boundary artifacts.

           When the decoder was originally configured with ``causal=False``,
           the streaming output will be mathematically equivalent to a full
           decode with ``causal=True`` — a minor shift in temporal context
           (each frame sees 2 past frames instead of 1 past + 1 future)
           that is visually imperceptible.
        """
        batch_size = sample.shape[0]
        dec = self.decoder

        # Force causal=True: conv_cache streaming REQUIRES causal (left-only)
        # padding.  With causal=False the cache is never used and each batch
        # gets independent symmetric padding → broken boundary frames.
        # See goku fengdavae_v3 which always uses causal mode for streaming.
        causal = True

        sample = dec.conv_in(sample, causal=causal)

        for up_block in dec.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(
                    sample,
                    causal=causal,
                    timestep=scaled_timestep if dec.timestep_conditioning else None,
                    generator=generator,
                )
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=causal, generator=generator)
            else:
                sample = up_block(sample, causal=causal)

        sample = dec.conv_norm_out(sample)

        if dec.timestep_conditioning:
            embedded_timestep = dec.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = dec.last_scale_shift_table[
                None, ..., None, None, None
            ].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = dec.conv_act(sample)
        sample = dec.conv_out(sample, causal=causal)
        sample = unpatchify(sample, patch_size_hw=dec.patch_size, patch_size_t=1)

        return sample

    # ── Block-level streaming decode ─────────────────────────────────────

    def _preprocess_latent(
        self,
        latent: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply noise injection + un-normalisation; compute scaled_timestep.

        Returns ``(preprocessed_latent, scaled_timestep)``.
        """
        batch_size = latent.shape[0]
        device = latent.device
        dtype = latent.dtype

        if self.decoder.timestep_conditioning:
            noise = (
                torch.randn(
                    latent.size(),
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                * self.decoder.decode_noise_scale
            )
            latent = noise + (1.0 - self.decoder.decode_noise_scale) * latent

        latent = self.decoder.per_channel_statistics.un_normalize(latent)

        scaled_timestep: Optional[torch.Tensor] = None
        if self.decoder.timestep_conditioning:
            timestep = torch.full(
                (batch_size,),
                self.decoder.decode_timestep,
                device=device,
                dtype=dtype,
            )
            scaled_timestep = timestep * self.decoder.timestep_scale_multiplier.to(
                latent
            )

        return latent, scaled_timestep

    def decode_block(
        self,
        block_latent: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Decode a single block's latent within an active streaming context.

        This is the block-level entry point for **true streaming**: call
        this once per AV block, right after the generator produces the
        block's latent.  The conv_cache accumulated from previous blocks
        provides temporal context automatically.

        Must be called inside a :meth:`streaming_context`.

        Parameters
        ----------
        block_latent : torch.Tensor
            ``[B, C, F_block, H, W]`` — one block's latent on GPU.
        generator : torch.Generator, optional
            RNG for deterministic noise injection (optional).

        Returns
        -------
        torch.Tensor
            ``[f, h, w, c]`` uint8 pixel frames on **CPU**.
        """
        # Pre-process this block's latent (noise + un-normalise)
        latent, scaled_timestep = self._preprocess_latent(block_latent, generator)

        # Run through decoder blocks with conv_cache
        decoded = self._forward_decoder_blocks(latent, scaled_timestep, generator)

        # Mark first batch as done (affects DepthToSpaceUpsample)
        self._is_first_batch = False

        # Convert to uint8 [f, h, w, c] on CPU
        frames = (
            ((decoded + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
        ).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")

        result = frames.cpu()
        del decoded, latent
        torch.cuda.empty_cache()

        return result

    # ── Public full-latent streaming decode ────────────────────────────

    def streaming_decode(
        self,
        latent: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[torch.Tensor]:
        """Streaming decode with conv-cache temporal batching.

        Produces output **mathematically identical** to non-streaming
        ``VideoDecoder.forward`` (see class docstring for caveats).

        Parameters
        ----------
        latent : torch.Tensor
            Latent tensor ``[B, C, F', H', W']`` on GPU.
        generator : torch.Generator, optional
            Random generator for deterministic noise injection.

        Yields
        ------
        torch.Tensor
            Decoded video chunk ``[f, h, w, c]`` uint8 on CPU.

            Frame count per chunk:

            * First batch (``B_lat`` latent frames):
              ``8 * B_lat − 7`` pixel frames
            * Subsequent batches (``B_lat`` latent frames):
              ``8 * B_lat`` pixel frames

            where ``8`` is the VAE temporal downscale factor.
        """
        batch_size, num_channels, num_frames, height, width = latent.shape
        frame_batch_size = self.num_latent_frames_batch_size
        device = latent.device
        dtype = latent.dtype

        # ── Pre-processing (once, on full latent) ──

        # 1. Noise injection (full tensor for deterministic noise pattern)
        if self.decoder.timestep_conditioning:
            noise = (
                torch.randn(
                    latent.size(),
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                * self.decoder.decode_noise_scale
            )
            latent = noise + (1.0 - self.decoder.decode_noise_scale) * latent

        # 2. Denormalise latents (per-channel, element-wise)
        latent = self.decoder.per_channel_statistics.un_normalize(latent)

        # 3. Timestep embedding (constant across batches)
        scaled_timestep: Optional[torch.Tensor] = None
        if self.decoder.timestep_conditioning:
            timestep = torch.full(
                (batch_size,),
                self.decoder.decode_timestep,
                device=device,
                dtype=dtype,
            )
            scaled_timestep = timestep * self.decoder.timestep_scale_multiplier.to(
                latent
            )

        # ── Temporal streaming ──

        num_batches = max(num_frames // frame_batch_size, 1)

        logger.info(
            "Streaming decode: %d latent frames, batch_size=%d, %d batches",
            num_frames,
            frame_batch_size,
            num_batches,
        )

        with self.streaming_context():
            for i in range(num_batches):
                # Remaining frames are folded into the FIRST batch (goku convention).
                remaining = num_frames % frame_batch_size
                start = frame_batch_size * i + (0 if i == 0 else remaining)
                end = frame_batch_size * (i + 1) + remaining

                z_batch = latent[:, :, start:end]

                decoded = self._forward_decoder_blocks(
                    z_batch, scaled_timestep, generator
                )

                # Mark first batch as done (affects DepthToSpaceUpsample).
                self._is_first_batch = False

                # Convert to uint8 [f, h, w, c] on CPU.
                frames = (
                    ((decoded + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
                ).to(torch.uint8)
                frames = rearrange(frames[0], "c f h w -> f h w c")

                yield frames.cpu()

                del decoded, z_batch
                torch.cuda.empty_cache()

    def streaming_decode_raw(
        self,
        latent: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[torch.Tensor]:
        """Like :meth:`streaming_decode` but yields raw float ``[B, C, F, H, W]``.

        Useful when the caller needs custom post-processing (e.g. keeping
        float precision for further compositing).
        """
        batch_size, num_channels, num_frames, height, width = latent.shape
        frame_batch_size = self.num_latent_frames_batch_size
        device = latent.device
        dtype = latent.dtype

        if self.decoder.timestep_conditioning:
            noise = (
                torch.randn(
                    latent.size(),
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                * self.decoder.decode_noise_scale
            )
            latent = noise + (1.0 - self.decoder.decode_noise_scale) * latent

        latent = self.decoder.per_channel_statistics.un_normalize(latent)

        scaled_timestep: Optional[torch.Tensor] = None
        if self.decoder.timestep_conditioning:
            timestep = torch.full(
                (batch_size,),
                self.decoder.decode_timestep,
                device=device,
                dtype=dtype,
            )
            scaled_timestep = timestep * self.decoder.timestep_scale_multiplier.to(
                latent
            )

        num_batches = max(num_frames // frame_batch_size, 1)

        with self.streaming_context():
            for i in range(num_batches):
                remaining = num_frames % frame_batch_size
                start = frame_batch_size * i + (0 if i == 0 else remaining)
                end = frame_batch_size * (i + 1) + remaining

                z_batch = latent[:, :, start:end]
                decoded = self._forward_decoder_blocks(
                    z_batch, scaled_timestep, generator
                )
                self._is_first_batch = False

                yield decoded

                del z_batch

    # ── Spatial tiling + temporal streaming ──────────────────────────────

    def tiled_streaming_decode(
        self,
        latent: torch.Tensor,
        tile_height: int,
        tile_width: int,
        overlap_height: Optional[int] = None,
        overlap_width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[torch.Tensor]:
        """Decode with **spatial tiling** and **temporal streaming**.

        Follows goku's ``tiled_decode`` pattern: outer loops iterate over
        spatial tiles; within each tile, temporal batches are streamed with
        conv-cache.  Adjacent tiles are alpha-blended to avoid seams.

        Parameters
        ----------
        latent : torch.Tensor
            ``[B, C, F', H', W']`` on GPU.
        tile_height, tile_width : int
            Spatial tile size **in latent space** (not pixels).
        overlap_height, overlap_width : int, optional
            Overlap between adjacent tiles in latent space.  Defaults to
            ``tile_height // 6`` and ``tile_width // 5`` respectively
            (matching goku's default factors).
        generator : torch.Generator, optional
            Random generator for deterministic noise.

        Yields
        ------
        torch.Tensor
            Decoded video chunk ``[f, h, w, c]`` uint8 on CPU.
            One chunk per temporal batch (all spatial tiles concatenated).
        """
        batch_size, num_channels, num_frames, height, width = latent.shape
        frame_batch_size = self.num_latent_frames_batch_size
        device = latent.device
        dtype = latent.dtype

        if overlap_height is None:
            overlap_height = max(1, tile_height // 6)
        if overlap_width is None:
            overlap_width = max(1, tile_width // 5)

        # Spatial scale factor (latent → pixel)
        scale_h = self.decoder.video_downscale_factors.height
        scale_w = self.decoder.video_downscale_factors.width

        tile_pixel_h = tile_height * scale_h
        tile_pixel_w = tile_width * scale_w
        blend_h = overlap_height * scale_h
        blend_w = overlap_width * scale_w
        row_limit_h = tile_pixel_h - blend_h
        row_limit_w = tile_pixel_w - blend_w

        step_h = tile_height - overlap_height
        step_w = tile_width - overlap_width

        # ── Pre-processing (once, full latent) ──

        if self.decoder.timestep_conditioning:
            noise = (
                torch.randn(
                    latent.size(),
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                * self.decoder.decode_noise_scale
            )
            latent = noise + (1.0 - self.decoder.decode_noise_scale) * latent

        latent = self.decoder.per_channel_statistics.un_normalize(latent)

        scaled_timestep: Optional[torch.Tensor] = None
        if self.decoder.timestep_conditioning:
            timestep = torch.full(
                (batch_size,),
                self.decoder.decode_timestep,
                device=device,
                dtype=dtype,
            )
            scaled_timestep = timestep * self.decoder.timestep_scale_multiplier.to(
                latent
            )

        num_batches = max(num_frames // frame_batch_size, 1)

        logger.info(
            "Tiled streaming decode: %d latent frames, batch=%d, "
            "tile=%dx%d, overlap=%dx%d, %d batches",
            num_frames,
            frame_batch_size,
            tile_height,
            tile_width,
            overlap_height,
            overlap_width,
            num_batches,
        )

        # ── Outer: spatial tiles, inner: temporal streaming ──

        # Collect decoded tiles per spatial position (each is a list of
        # temporal chunks).  After all temporal batches are done we blend
        # the spatial tiles and yield one chunk per temporal batch.
        #
        # rows[row_idx][col_idx] = concatenated temporal tensor [B,C,T,H,W]
        rows: List[List[torch.Tensor]] = []

        for i_row in range(0, height, step_h):
            row: List[torch.Tensor] = []
            for j_col in range(0, width, step_w):
                # Reset conv cache for each spatial tile (independent context)
                self._patch()
                self.reset_cache()

                time_chunks: List[torch.Tensor] = []

                for k in range(num_batches):
                    remaining = num_frames % frame_batch_size
                    start_t = frame_batch_size * k + (0 if k == 0 else remaining)
                    end_t = frame_batch_size * (k + 1) + remaining

                    tile = latent[
                        :,
                        :,
                        start_t:end_t,
                        i_row : i_row + tile_height,
                        j_col : j_col + tile_width,
                    ]

                    decoded_tile = self._forward_decoder_blocks(
                        tile, scaled_timestep, generator
                    )
                    self._is_first_batch = False

                    time_chunks.append(decoded_tile)

                    del tile

                # Concatenate all temporal chunks for this spatial tile
                row.append(torch.cat(time_chunks, dim=2))
                del time_chunks

                self._unpatch()

            rows.append(row)

        # ── Spatial blending (goku blend_v / blend_h pattern) ──

        result_rows: List[torch.Tensor] = []
        for i, row_tiles in enumerate(rows):
            result_row: List[torch.Tensor] = []
            for j, tile in enumerate(row_tiles):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = _blend_h(row_tiles[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :row_limit_h, :row_limit_w])
            result_rows.append(torch.cat(result_row, dim=4))

        full_decoded = torch.cat(result_rows, dim=3)  # [B, C, T_total, H, W]

        # Yield in temporal chunks (matching streaming_decode convention)
        offset = 0
        for k in range(num_batches):
            remaining = num_frames % frame_batch_size
            n_lat = (
                frame_batch_size + remaining if k == 0 else frame_batch_size
            )
            # First batch: 8*n_lat - 7 pixel frames; subsequent: 8*n_lat
            if k == 0:
                n_pix = 8 * n_lat - 7
            else:
                n_pix = 8 * n_lat

            chunk = full_decoded[:, :, offset : offset + n_pix]
            offset += n_pix

            frames = (
                ((chunk + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
            ).to(torch.uint8)
            frames = rearrange(frames[0], "c f h w -> f h w c")

            yield frames.cpu()

            del chunk

        del full_decoded, rows, result_rows


# ═══════════════════════════════════════════════════════════════════════════
# Spatial blending helpers (matching goku's blend_v / blend_h)
# ═══════════════════════════════════════════════════════════════════════════


def _blend_v(
    a: torch.Tensor, b: torch.Tensor, blend_extent: int
) -> torch.Tensor:
    """Vertical (height) blending between adjacent spatial tiles."""
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for y in range(blend_extent):
        alpha = y / blend_extent
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - alpha) + b[:, :, :, y, :] * alpha
    return b


def _blend_h(
    a: torch.Tensor, b: torch.Tensor, blend_extent: int
) -> torch.Tensor:
    """Horizontal (width) blending between adjacent spatial tiles."""
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    for x in range(blend_extent):
        alpha = x / blend_extent
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - alpha) + b[:, :, :, :, x] * alpha
    return b
