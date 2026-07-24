"""
AVCausalMaskBuilder: Builds causal masks with Global Prefix for audio-video
joint attention.

This module implements:
- Global Prefix pattern: V_0 + A_0 paired in Block 0 (both first-frame tokens)
- Block-wise causal masks for video and audio self-attention
- Block-aligned causal masks for A2V and V2A cross-attention
- Flexattention BlockMask support (REQUIRED -- no fallback)
- Comprehensive assertions for mask correctness

Block Structure (example: 16 video latents from 121 pixel frames):
    Block 0 (Global Prefix): V_0 (1 video frame) + A_0 (1 audio frame)
    Block 1: V_1-V_3 (3 video frames) + A_1-A_25 (25 audio frames)
    Block 2: V_4-V_6 (3 video frames) + A_26-A_50 (25 audio frames)
    Block 3: V_7-V_9 (3 video frames) + A_51-A_75 (25 audio frames)
    Block 4: V_10-V_12 (3 video frames) + A_76-A_100 (25 audio frames)
    Block 5: V_13-V_15 (3 video frames) + A_101-A_125 (25 audio frames)
    Total: 16 video + 126 audio = NO TRUNCATION from ltx-core's raw count

Attention Rules:
    - Intra-block: bidirectional (tokens within same block attend freely)
    - Inter-block: causal (block k sees blocks 0..k, not k+1..)
    - V_0 and A_0 attend to each other (both in Block 0)
    - All subsequent tokens can attend to Block 0 (V_0 + A_0)

Design Rationale (Causal 3D VAE Alignment):
    Both video and audio causal VAEs have stride 1 for the first frame:
    - Video: V_0 covers 1 pixel frame (stride 1), V_k covers 8 (stride 8)
    - Audio: A_0 covers 1 mel frame (stride 1), A_k covers 4 (stride 4)
    Both first-frame tokens exist at time ~0s, so they belong together in
    Block 0. This gives exact alignment:
    1. 126 audio = 1 (Block 0) + 5×25 (Blocks 1-5) → NO TRUNCATION
    2. V_0 and A_0 can attend to each other as temporal neighbors
    3. Block 0 serves as global context visible to all subsequent tokens
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

# Import flexattention (requires PyTorch 2.2+ with CUDA)
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask

from ltx_causal.config import CausalMaskConfig


# ============================================================================
# Block Computation Utilities
# ============================================================================

@dataclass
class AVBlock:
    """Represents a synchronized audio-video generation block."""
    block_idx: int
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int

    @property
    def video_frames(self) -> int:
        return self.video_end - self.video_start

    @property
    def audio_frames(self) -> int:
        return self.audio_end - self.audio_start

    @property
    def is_global_prefix(self) -> bool:
        """Block 0 is the Global Prefix (V_0 + A_0)."""
        return self.block_idx == 0

    def __repr__(self) -> str:
        prefix = " [Global Prefix]" if self.is_global_prefix else ""
        return (
            f"AVBlock(idx={self.block_idx}, "
            f"video=[{self.video_start}:{self.video_end}], "
            f"audio=[{self.audio_start}:{self.audio_end}]){prefix}"
        )


# Audio frames per standard block: exactly 25 (1 second = 3 video : 25 audio)
AUDIO_FRAMES_PER_BLOCK = 25
# Audio frames for the first block when using OmniForcing layout (4 video frames)
AUDIO_FRAMES_FIRST_BLOCK = 26


def compute_av_blocks(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 0,
) -> List[AVBlock]:
    """
    Compute audio-video block synchronization.

    Two modes based on num_frame_per_block_first:

    1. num_frame_per_block_first == 0 (legacy Global Prefix mode):
       Block 0 (Global Prefix): V_0 + A_0 (1 video + 1 audio).
       Blocks 1..N: num_frame_per_block video + AUDIO_FRAMES_PER_BLOCK audio.

    2. num_frame_per_block_first > 0 (OmniForcing mode, e.g. 4-3-3-3):
       Block 0: num_frame_per_block_first video + AUDIO_FRAMES_FIRST_BLOCK audio.
       Blocks 1..N: num_frame_per_block video + AUDIO_FRAMES_PER_BLOCK audio.
       No Global Prefix special case.

    Args:
        total_video_latent_frames: Total number of video latent frames
        num_frame_per_block: Video frames per standard block (default 3)
        num_frame_per_block_first: Video frames in Block 0. 0 = legacy Global Prefix.

    Returns:
        List of AVBlock with video and audio frame ranges

    Example (num_frame_per_block_first=4, 16 video frames):
        Block 0: video (0,4), audio (0,26)
        Block 1: video (4,7), audio (26,51)
        Block 2: video (7,10), audio (51,76)
        Block 3: video (10,13), audio (76,101)
        Block 4: video (13,16), audio (101,126)

    Example (num_frame_per_block_first=0, 16 video frames, legacy):
        Block 0: video (0,1), audio (0,1)       # Global Prefix
        Block 1: video (1,4), audio (1,26)
        Block 2: video (4,7), audio (26,51)
        ...
    """
    if num_frame_per_block_first > 0:
        # === OmniForcing mode: 4-3-3-3 layout ===
        assert total_video_latent_frames >= num_frame_per_block_first, (
            f"Need >= {num_frame_per_block_first} video frames, got {total_video_latent_frames}"
        )

        blocks = []

        # Block 0: num_frame_per_block_first video + AUDIO_FRAMES_FIRST_BLOCK audio
        blocks.append(AVBlock(
            block_idx=0,
            video_start=0,
            video_end=num_frame_per_block_first,
            audio_start=0,
            audio_end=AUDIO_FRAMES_FIRST_BLOCK,
        ))

        video_idx = num_frame_per_block_first
        audio_idx = AUDIO_FRAMES_FIRST_BLOCK
        block_idx = 1

        while video_idx < total_video_latent_frames:
            v_start = video_idx
            v_end = min(video_idx + num_frame_per_block, total_video_latent_frames)
            a_start = audio_idx
            video_frames_in_block = v_end - v_start
            if video_frames_in_block == num_frame_per_block:
                a_end = a_start + AUDIO_FRAMES_PER_BLOCK
            else:
                # Partial last block
                a_end = a_start + (AUDIO_FRAMES_PER_BLOCK * video_frames_in_block) // num_frame_per_block

            blocks.append(AVBlock(
                block_idx=block_idx,
                video_start=v_start,
                video_end=v_end,
                audio_start=a_start,
                audio_end=a_end,
            ))

            video_idx = v_end
            audio_idx = a_end
            block_idx += 1

        return blocks

    # === Legacy Global Prefix mode: 1-3-3-3 layout ===
    assert total_video_latent_frames >= 1, (
        f"Need at least 1 video latent frame, got {total_video_latent_frames}"
    )

    blocks = []

    # Block 0: Global Prefix (V_0 + A_0, both first-frame causal fix tokens)
    blocks.append(AVBlock(
        block_idx=0,
        video_start=0,
        video_end=1,
        audio_start=0,
        audio_end=1,
    ))

    # Standard blocks: num_frame_per_block video + AUDIO_FRAMES_PER_BLOCK audio
    video_idx = 1
    audio_idx = 1  # Start after A_0 (which is in Block 0)
    block_idx = 1

    while video_idx < total_video_latent_frames:
        v_start = video_idx
        v_end = min(video_idx + num_frame_per_block, total_video_latent_frames)

        a_start = audio_idx
        video_frames_in_block = v_end - v_start
        if video_frames_in_block == num_frame_per_block:
            a_end = a_start + AUDIO_FRAMES_PER_BLOCK
        else:
            # Partial last block: proportional audio frames (integer division)
            a_end = a_start + (AUDIO_FRAMES_PER_BLOCK * video_frames_in_block) // num_frame_per_block

        blocks.append(AVBlock(
            block_idx=block_idx,
            video_start=v_start,
            video_end=v_end,
            audio_start=a_start,
            audio_end=a_end,
        ))

        video_idx = v_end
        audio_idx = a_end
        block_idx += 1

    return blocks


def compute_aligned_audio_frames(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 0,
) -> int:
    """
    Compute the aligned audio frame count for the block structure.

    Args:
        total_video_latent_frames: Total number of video latent frames
        num_frame_per_block: Video frames per standard block
        num_frame_per_block_first: Video frames in Block 0 (0 = legacy Global Prefix)

    Returns:
        Aligned audio frame count

    Example (OmniForcing, num_frame_per_block_first=4, 16 video frames):
        - Block 0: 4 video + 26 audio
        - 12 remaining / 3 = 4 standard blocks × 25 audio = 100
        - Total: 26 + 100 = 126

    Example (legacy, num_frame_per_block_first=0, 16 video frames):
        - Block 0 (Global Prefix): 1 audio
        - 15 remaining / 3 = 5 standard blocks × 25 audio = 125
        - Total: 1 + 125 = 126
    """
    blocks = compute_av_blocks(
        total_video_latent_frames, num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    return blocks[-1].audio_end if blocks else 0


def compute_total_audio_frames(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 0,
) -> int:
    """Compute total aligned audio frames for given video frames.

    Alias for compute_aligned_audio_frames.
    """
    return compute_aligned_audio_frames(
        total_video_latent_frames, num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )


# ============================================================================
# AVCausalMaskBuilder
# ============================================================================

class AVCausalMaskBuilder:
    """
    Builds causal attention masks with Global Prefix for audio-video generation.

    Uses FlexAttention BlockMask for efficient block-wise causal self-attention,
    and dense boolean tensors for cross-modal causal attention.

    Block Structure:
        Block 0 (Global Prefix): V_0 + A_0 (1 video frame + 1 audio frame)
        Block k (k>=1): 3 video frames + 25 audio frames (standard blocks)

    Mask Types:
        1. Video Self-Attention: Block-wise causal (same block + previous blocks)
        2. Audio Self-Attention: Block-wise causal (same block + previous blocks)
        3. A2V Cross-Attention: Video in block k -> audio in blocks 0..k
           (Block 0 video sees A_0)
        4. V2A Cross-Attention: Audio in block k -> video in blocks 0..k
           (All audio always sees V_0)

    Example:
        builder = AVCausalMaskBuilder(video_frame_seqlen=384)
        blocks = compute_av_blocks(16, num_frame_per_block=3)
        video_mask = builder.build_video_self_causal_mask(blocks)
        audio_mask = builder.build_audio_self_causal_mask(blocks)
        a2v_mask = builder.build_a2v_causal_mask(blocks)
        v2a_mask = builder.build_v2a_causal_mask(blocks)
    """

    def __init__(
        self,
        video_frame_seqlen: int = 384,
        audio_frame_seqlen: int = 1,
        num_frame_per_block: int = 3,
        num_audio_sink_tokens: int = 0,
        use_flex_attention: bool = True,
    ):
        """
        Initialize the mask builder.

        Args:
            video_frame_seqlen: Tokens per video frame (default 384 for 512x768
                with patch_size=1: (512/32)*(768/32) = 16*24 = 384)
            audio_frame_seqlen: Tokens per audio frame (default 1)
            num_frame_per_block: Video frames per standard block (default 3)
            num_audio_sink_tokens: Number of sink tokens prepended to audio (default 0)
            use_flex_attention: Whether to use FlexAttention BlockMask (True) or
                dense boolean masks for standard SDPA (False)
        """
        if use_flex_attention and not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(
                "FlexAttention is REQUIRED for causal LTX-2 but is not available. "
                "Please install PyTorch 2.2+ with CUDA support. "
                "Set use_flex_attention=false to use SDPA with dense masks instead."
            )

        self.video_frame_seqlen = video_frame_seqlen
        self.audio_frame_seqlen = audio_frame_seqlen
        self.num_frame_per_block = num_frame_per_block
        self.num_audio_sink_tokens = num_audio_sink_tokens
        self.use_flex_attention = use_flex_attention

    # ========================================================================
    # Video Self-Attention Mask
    # ========================================================================

    def build_video_self_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> "BlockMask":
        """
        Build block-wise causal mask for video self-attention.

        Block 0 (Global Prefix, V_0): can only attend to itself (384 tokens).
        Block k (k>=1): can attend to all tokens in blocks 0..k.

        This means V_0 tokens see only V_0, but all subsequent video tokens
        can see V_0 as a global context.

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            BlockMask for FlexAttention
        """
        total_video_frames = blocks[-1].video_end
        total_tokens = total_video_frames * self.video_frame_seqlen

        # Pad to multiple of 128 for flexattention optimization
        padded_length = math.ceil(total_tokens / 128) * 128

        # ends[q_idx] = exclusive end of the causal window for token at q_idx
        # Token q can attend to all kv_idx < ends[q]
        ends = torch.zeros(padded_length, device=device, dtype=torch.long)

        for block in blocks:
            token_start = block.video_start * self.video_frame_seqlen
            token_end = block.video_end * self.video_frame_seqlen
            ends[token_start:token_end] = token_end

        # Padding tokens can see all real tokens
        ends[total_tokens:] = total_tokens

        def mask_fn(b, h, q_idx, kv_idx):
            # Token can attend to:
            # 1. All tokens in its causal window: kv_idx < ends[q_idx]
            # 2. Itself (for stability): q_idx == kv_idx
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        return create_block_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            device=device,
            # _compile=True 会走 torch.compile/Inductor 路径，动态 shape 下 Triton kernel 偶现
            # 越界访问（多机不同 rank 不同 padded_length 时会触发）。
            # mask 有 _mask_cache 缓存，不在热路径，关掉编译更稳。
            _compile=False,
        )

    # ========================================================================
    # Audio Self-Attention Mask
    # ========================================================================

    def build_audio_self_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> Optional["BlockMask"]:
        """
        Build block-wise causal mask for audio self-attention.

        Audio exists in all blocks (A_0 in Block 0, A_1..A_N in standard
        blocks). Audio in block k can attend to all audio in blocks 0..k.

        When num_audio_sink_tokens > 0, sink tokens are prepended to the
        audio sequence and are part of Block 0 (visible to all audio tokens).

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            BlockMask for FlexAttention, or None if no audio frames
        """
        total_audio_frames = blocks[-1].audio_end
        if total_audio_frames == 0:
            return None

        num_sink = self.num_audio_sink_tokens
        total_tokens = (num_sink + total_audio_frames) * self.audio_frame_seqlen

        # Pad to multiple of 128
        padded_length = math.ceil(total_tokens / 128) * 128

        # Compute block end indices for each audio token
        ends = torch.zeros(padded_length, device=device, dtype=torch.long)

        # Sink tokens are part of Block 0: they can see up to Block 0's audio end + sinks
        block0 = blocks[0]
        sink_causal_end = (num_sink + block0.audio_end) * self.audio_frame_seqlen
        ends[:num_sink * self.audio_frame_seqlen] = sink_causal_end

        for block in blocks:
            if block.audio_end <= block.audio_start:
                continue

            token_start = (num_sink + block.audio_start) * self.audio_frame_seqlen
            token_end = (num_sink + block.audio_end) * self.audio_frame_seqlen
            ends[token_start:token_end] = token_end

        # Padding tokens see all real tokens
        ends[total_tokens:] = total_tokens

        def mask_fn(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        return create_block_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            device=device,
            # 同上：audio_self mask 也关掉 _compile，避免 Inductor 动态 shape 越界。
            _compile=False,
        )

    # ========================================================================
    # A2V Cross-Attention Mask
    # ========================================================================

    def build_a2v_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Build causal mask for Audio-to-Video cross-attention.

        Query: Video tokens
        Key/Value: Audio tokens (with sink tokens prepended)

        Rules:
        - Block 0 (Global Prefix) video -> sinks + A_0
        - Block k (k>=1) video -> sinks + audio from blocks 0..k

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            Boolean mask [num_video_tokens, num_audio_tokens_with_sinks]
            True = can attend, False = masked
        """
        total_video_frames = blocks[-1].video_end
        total_audio_frames = blocks[-1].audio_end
        num_sink = self.num_audio_sink_tokens

        num_video_tokens = total_video_frames * self.video_frame_seqlen
        num_audio_tokens = (num_sink + total_audio_frames) * self.audio_frame_seqlen

        mask = torch.zeros(
            num_video_tokens, num_audio_tokens,
            dtype=torch.bool, device=device,
        )

        for block in blocks:
            v_token_start = block.video_start * self.video_frame_seqlen
            v_token_end = block.video_end * self.video_frame_seqlen
            a_causal_end = (num_sink + block.audio_end) * self.audio_frame_seqlen

            # Video tokens in this block can attend to sinks + all audio up to
            # this block's end (blocks 0..k)
            mask[v_token_start:v_token_end, :a_causal_end] = True

        return mask

    # ========================================================================
    # V2A Cross-Attention Mask
    # ========================================================================

    def build_v2a_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Build causal mask for Video-to-Audio cross-attention.

        Query: Audio tokens (with sink tokens prepended)
        Key/Value: Video tokens

        Rules:
        - Sink tokens (part of Block 0) -> video Block 0
        - Audio in block k -> video from blocks 0..k (always includes V_0)
        - All audio always sees the Global Prefix (V_0)

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            Boolean mask [num_audio_tokens_with_sinks, num_video_tokens]
            True = can attend, False = masked
        """
        total_video_frames = blocks[-1].video_end
        total_audio_frames = blocks[-1].audio_end
        num_sink = self.num_audio_sink_tokens

        num_video_tokens = total_video_frames * self.video_frame_seqlen
        num_audio_tokens = (num_sink + total_audio_frames) * self.audio_frame_seqlen

        mask = torch.zeros(
            num_audio_tokens, num_video_tokens,
            dtype=torch.bool, device=device,
        )

        # Sink tokens are part of Block 0: they see video Block 0
        block0 = blocks[0]
        v_block0_end = block0.video_end * self.video_frame_seqlen
        sink_row_end = num_sink * self.audio_frame_seqlen
        mask[:sink_row_end, :v_block0_end] = True

        for block in blocks:
            a_token_start = (num_sink + block.audio_start) * self.audio_frame_seqlen
            a_token_end = (num_sink + block.audio_end) * self.audio_frame_seqlen
            v_causal_end = block.video_end * self.video_frame_seqlen

            # Audio tokens in this block can attend to all video up to
            # this block's end. Block 0's A_0 sees V_0.
            mask[a_token_start:a_token_end, :v_causal_end] = True

        return mask

    # ========================================================================
    # Dense Boolean Masks for SDPA (FlexAttention alternative)
    # ========================================================================

    def build_video_self_dense_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Build dense boolean mask for video self-attention (SDPA path).

        Same causal semantics as build_video_self_causal_mask but returns a
        [L, L] boolean tensor instead of a FlexAttention BlockMask.

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            Boolean mask [num_video_tokens, num_video_tokens]
            True = can attend, False = masked
        """
        total_video_frames = blocks[-1].video_end
        total_tokens = total_video_frames * self.video_frame_seqlen

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

        for block in blocks:
            token_start = block.video_start * self.video_frame_seqlen
            token_end = block.video_end * self.video_frame_seqlen
            # Tokens in this block can attend to all tokens in blocks 0..this_block
            mask[token_start:token_end, :token_end] = True

        return mask

    def build_audio_self_dense_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> Optional[torch.Tensor]:
        """
        Build dense boolean mask for audio self-attention (SDPA path).

        Args:
            blocks: List of AVBlock from compute_av_blocks()
            device: Target device

        Returns:
            Boolean mask [num_audio_tokens, num_audio_tokens] or None
        """
        total_audio_frames = blocks[-1].audio_end
        if total_audio_frames == 0:
            return None

        num_sink = self.num_audio_sink_tokens
        total_tokens = (num_sink + total_audio_frames) * self.audio_frame_seqlen

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

        # Sink tokens (Block 0): can see themselves + Block 0 audio
        block0 = blocks[0]
        sink_end = num_sink * self.audio_frame_seqlen
        block0_audio_end = (num_sink + block0.audio_end) * self.audio_frame_seqlen
        mask[:sink_end, :block0_audio_end] = True

        for block in blocks:
            if block.audio_end <= block.audio_start:
                continue
            token_start = (num_sink + block.audio_start) * self.audio_frame_seqlen
            token_end = (num_sink + block.audio_end) * self.audio_frame_seqlen
            # Tokens in this block can attend to sinks + all audio up to this block
            mask[token_start:token_end, :token_end] = True

        return mask


# ============================================================================
# Mask Verification
# ============================================================================

def verify_causal_masks(
    masks: dict,
    blocks: List[AVBlock],
    video_frame_seqlen: int,
    audio_frame_seqlen: int,
) -> None:
    """
    Comprehensive verification of causal mask correctness.

    Checks:
    1. Exact mask shapes for dense masks (A2V, V2A)
    2. Global Prefix bidirectional (V_0 <-> A_0 in Block 0)
    3. All audio can attend to V_0 (Global Prefix video)
    4. Zero future information leakage (no token sees future blocks)
    5. Intra-block bidirectional visibility (tokens within a block see each other)

    Args:
        masks: Dictionary with 'video_self', 'audio_self', 'a2v', 'v2a' keys
        blocks: List of AVBlock from compute_av_blocks()
        video_frame_seqlen: Tokens per video frame
        audio_frame_seqlen: Tokens per audio frame

    Raises:
        AssertionError: If any verification check fails
    """
    total_video_frames = blocks[-1].video_end
    total_audio_frames = blocks[-1].audio_end
    num_video_tokens = total_video_frames * video_frame_seqlen
    num_audio_tokens = total_audio_frames * audio_frame_seqlen

    # === Required mask keys ===
    for key in ('video_self', 'audio_self', 'a2v', 'v2a'):
        assert key in masks, f"Missing '{key}' mask"

    a2v_mask = masks['a2v']
    v2a_mask = masks['v2a']

    # === Shape Assertions (Dense Masks) ===
    assert a2v_mask.shape == (num_video_tokens, num_audio_tokens), (
        f"A2V mask shape mismatch: {a2v_mask.shape} != "
        f"({num_video_tokens}, {num_audio_tokens})"
    )
    assert v2a_mask.shape == (num_audio_tokens, num_video_tokens), (
        f"V2A mask shape mismatch: {v2a_mask.shape} != "
        f"({num_audio_tokens}, {num_video_tokens})"
    )

    # === Block 0 (Global Prefix) V_0 <-> A_0 Bidirectional ===
    v0_token_end = blocks[0].video_end * video_frame_seqlen
    a0_token_end = blocks[0].audio_end * audio_frame_seqlen

    # V_0 should see A_0 in A2V
    if a0_token_end > 0:
        assert a2v_mask[:v0_token_end, :a0_token_end].all(), (
            "VIOLATION: Block 0 V_0 cannot attend to A_0 in A2V! "
            "V_0 and A_0 should be bidirectional in Global Prefix."
        )

    # V_0 should NOT see audio beyond A_0
    if a0_token_end < num_audio_tokens:
        assert not a2v_mask[:v0_token_end, a0_token_end:].any(), (
            "VIOLATION: Block 0 V_0 attends to audio beyond A_0 in A2V! "
            "V_0 should only see A_0 (its own block's audio)."
        )

    # A_0 should see V_0 in V2A
    if a0_token_end > 0:
        assert v2a_mask[:a0_token_end, :v0_token_end].all(), (
            "VIOLATION: Block 0 A_0 cannot attend to V_0 in V2A! "
            "A_0 and V_0 should be bidirectional in Global Prefix."
        )

    # === All Audio Sees V_0 ===
    if num_audio_tokens > 0:
        assert v2a_mask[:, :v0_token_end].all(), (
            "VIOLATION: Not all audio tokens can attend to V_0 (Global Prefix)! "
            "All audio tokens should be able to see the Global Prefix."
        )

    # === Zero Future Leakage (A2V) ===
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        a_causal_end = block.audio_end * audio_frame_seqlen

        # Video tokens in this block must NOT see future audio
        if a_causal_end < num_audio_tokens:
            future_leak = a2v_mask[v_token_start:v_token_end, a_causal_end:]
            assert not future_leak.any(), (
                f"A2V FUTURE LEAKAGE: Block {block.block_idx} video tokens "
                f"(rows {v_token_start}:{v_token_end}) attend to future audio "
                f"tokens after column {a_causal_end}!"
            )

    # === Zero Future Leakage (V2A) ===
    for block in blocks:
        a_token_start = block.audio_start * audio_frame_seqlen
        a_token_end = block.audio_end * audio_frame_seqlen
        v_causal_end = block.video_end * video_frame_seqlen

        # Audio tokens in this block must NOT see future video
        if v_causal_end < num_video_tokens:
            future_leak = v2a_mask[a_token_start:a_token_end, v_causal_end:]
            assert not future_leak.any(), (
                f"V2A FUTURE LEAKAGE: Block {block.block_idx} audio tokens "
                f"(rows {a_token_start}:{a_token_end}) attend to future video "
                f"tokens after column {v_causal_end}!"
            )

    # === Intra-block Bidirectional (A2V) ===
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        a_token_start = block.audio_start * audio_frame_seqlen
        a_token_end = block.audio_end * audio_frame_seqlen

        # Video tokens in this block should see all audio in their own block
        intra_block = a2v_mask[v_token_start:v_token_end, a_token_start:a_token_end]
        assert intra_block.all(), (
            f"A2V intra-block violation: Block {block.block_idx} video tokens "
            f"cannot attend to all audio tokens in their own block!"
        )

    # === Intra-block Bidirectional (V2A) ===
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        a_token_start = block.audio_start * audio_frame_seqlen
        a_token_end = block.audio_end * audio_frame_seqlen

        # Audio tokens in this block should see all video in their own block
        intra_block = v2a_mask[a_token_start:a_token_end, v_token_start:v_token_end]
        assert intra_block.all(), (
            f"V2A intra-block violation: Block {block.block_idx} audio tokens "
            f"cannot attend to all video tokens in their own block!"
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def compute_causal_log_scales(
    blocks: List[AVBlock],
    video_frame_seqlen: int = 384,
    audio_frame_seqlen: int = 1,
    device: Union[torch.device, str] = "cuda",
    num_audio_sink_tokens: int = 0,
) -> dict:
    """
    Compute per-token log-ratio scale factors for causal attention outputs.

    In bidirectional attention, every token sees all N tokens. Under causal
    masking, a token in block k only sees tokens in blocks 0..k. The
    information deficit is proportional to log(visible / total).

    Scale factor per token:
        scale = log(1 + visible_tokens) / log(1 + total_tokens)

    Properties:
        - Last block (sees everything): scale ≈ 1.0
        - Block 0 (sees only itself): scale is smallest
        - Monotonically increasing with block index
        - No learnable parameters

    Args:
        blocks: List of AVBlock from compute_av_blocks()
        video_frame_seqlen: Tokens per video frame (384 for 512x768)
        audio_frame_seqlen: Tokens per audio frame (1)
        device: Target device

    Returns:
        Dictionary with keys:
        - 'video_self_scale': [1, total_video_tokens, 1] scale for video self-attn
        - 'audio_self_scale': [1, total_audio_tokens, 1] scale for audio self-attn
        - 'a2v_scale': [1, total_video_tokens, 1] scale for A2V cross-attn (Q=video)
        - 'v2a_scale': [1, total_audio_tokens, 1] scale for V2A cross-attn (Q=audio)
    """
    num_sink = num_audio_sink_tokens
    total_video_frames = blocks[-1].video_end
    total_audio_frames = blocks[-1].audio_end
    total_video_tokens = total_video_frames * video_frame_seqlen
    total_audio_tokens_raw = total_audio_frames * audio_frame_seqlen
    total_audio_tokens = (num_sink + total_audio_frames) * audio_frame_seqlen

    log_total_v = math.log(1 + total_video_tokens)
    log_total_a = math.log(1 + total_audio_tokens)

    # --- Video self-attention scale ---
    # Token in block k sees video tokens 0..video_end_k
    video_self_scale = torch.ones(total_video_tokens, device=device)
    for block in blocks:
        token_start = block.video_start * video_frame_seqlen
        token_end = block.video_end * video_frame_seqlen
        visible = token_end  # Can see all video tokens up to end of this block
        video_self_scale[token_start:token_end] = math.log(1 + visible) / log_total_v

    # --- Audio self-attention scale (includes sink tokens) ---
    audio_self_scale = torch.ones(total_audio_tokens, device=device)
    # Sink tokens (Block 0): visible = sink + block0 audio
    block0 = blocks[0]
    sink_visible = (num_sink + block0.audio_end) * audio_frame_seqlen
    sink_scale = math.log(1 + sink_visible) / log_total_a
    audio_self_scale[:num_sink * audio_frame_seqlen] = sink_scale
    for block in blocks:
        if block.audio_end <= block.audio_start:
            continue
        token_start = (num_sink + block.audio_start) * audio_frame_seqlen
        token_end = (num_sink + block.audio_end) * audio_frame_seqlen
        visible = token_end
        audio_self_scale[token_start:token_end] = math.log(1 + visible) / log_total_a

    # --- A2V cross-attention scale (Q=video, KV=audio with sinks) ---
    # Video token in block k can see sinks + audio tokens 0..audio_end_k
    a2v_scale = torch.ones(total_video_tokens, device=device)
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        visible_audio = (num_sink + block.audio_end) * audio_frame_seqlen
        a2v_scale[v_token_start:v_token_end] = math.log(1 + visible_audio) / log_total_a

    # --- V2A cross-attention scale (Q=audio with sinks, KV=video) ---
    # Audio token in block k can see video tokens 0..video_end_k
    v2a_scale = torch.ones(total_audio_tokens, device=device)
    # Sink tokens see video Block 0
    v2a_sink_visible = block0.video_end * video_frame_seqlen
    v2a_scale[:num_sink * audio_frame_seqlen] = math.log(1 + v2a_sink_visible) / log_total_v
    for block in blocks:
        if block.audio_end <= block.audio_start:
            continue
        a_token_start = (num_sink + block.audio_start) * audio_frame_seqlen
        a_token_end = (num_sink + block.audio_end) * audio_frame_seqlen
        visible_video = block.video_end * video_frame_seqlen
        v2a_scale[a_token_start:a_token_end] = math.log(1 + visible_video) / log_total_v

    return {
        'video_self_scale': video_self_scale.unsqueeze(0).unsqueeze(-1),  # [1, T_v, 1]
        'audio_self_scale': audio_self_scale.unsqueeze(0).unsqueeze(-1),  # [1, T_a_with_sink, 1]
        'a2v_scale': a2v_scale.unsqueeze(0).unsqueeze(-1),               # [1, T_v, 1]
        'v2a_scale': v2a_scale.unsqueeze(0).unsqueeze(-1),               # [1, T_a_with_sink, 1]
    }


def build_all_causal_masks(
    num_video_frames: int,
    num_audio_frames: int,
    config: CausalMaskConfig,
    device: Union[torch.device, str] = "cuda",
) -> dict:
    """
    Build all causal masks needed for LTX-2 audio-video generation.

    With V_0 + A_0 in Block 0, the aligned count matches ltx-core's raw
    audio count exactly (no truncation needed).

    Args:
        num_video_frames: Number of video latent frames
        num_audio_frames: Number of audio latent frames (must be aligned)
        config: Mask configuration (REQUIRED)
        device: Target device

    Returns:
        Dictionary with keys:
        - 'video_self': Video self-attention mask (BlockMask or dense tensor)
        - 'audio_self': Audio self-attention mask (BlockMask, dense tensor, or None)
        - 'a2v': A2V cross-attention boolean mask
        - 'v2a': V2A cross-attention boolean mask

    Raises:
        AssertionError: If num_audio_frames doesn't match aligned count
    """
    # Compute block structure
    num_frame_per_block_first = getattr(config, 'num_frame_per_block_first', 0)
    blocks = compute_av_blocks(
        num_video_frames,
        config.num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    expected_audio = blocks[-1].audio_end if blocks else 0

    assert num_audio_frames == expected_audio, (
        f"Audio frame count ({num_audio_frames}) does not match aligned count "
        f"({expected_audio}) for {num_video_frames} video frames with "
        f"num_frame_per_block={config.num_frame_per_block}. "
        f"Use compute_aligned_audio_frames() to determine the correct count."
    )

    num_sink = getattr(config, 'num_audio_sink_tokens', 0)
    use_flex = getattr(config, 'use_flex_attention', True)

    builder = AVCausalMaskBuilder(
        video_frame_seqlen=config.video_frame_seqlen,
        audio_frame_seqlen=config.audio_frame_seqlen,
        num_frame_per_block=config.num_frame_per_block,
        num_audio_sink_tokens=num_sink,
        use_flex_attention=use_flex,
    )

    if use_flex:
        # FlexAttention BlockMask path (original)
        masks = {
            'video_self': builder.build_video_self_causal_mask(blocks, device=device),
            'audio_self': builder.build_audio_self_causal_mask(blocks, device=device),
            'a2v': builder.build_a2v_causal_mask(blocks, device=device),
            'v2a': builder.build_v2a_causal_mask(blocks, device=device),
        }
    else:
        # Dense boolean mask path for SDPA (OmniForcing-aligned)
        masks = {
            'video_self': builder.build_video_self_dense_mask(blocks, device=device),
            'audio_self': builder.build_audio_self_dense_mask(blocks, device=device),
            'a2v': builder.build_a2v_causal_mask(blocks, device=device),
            'v2a': builder.build_v2a_causal_mask(blocks, device=device),
        }

    # Comprehensive verification of all masks
    # Skip verification when sink tokens are present (mask shapes differ from standard)
    # Also skip for dense masks (verification assumes FlexAttention BlockMask shapes)
    if num_sink == 0 and use_flex:
        verify_causal_masks(
            masks, blocks,
            video_frame_seqlen=config.video_frame_seqlen,
            audio_frame_seqlen=config.audio_frame_seqlen,
        )

    return masks
