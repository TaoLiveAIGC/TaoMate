from typing import Any

import torch.nn as nn

from ltx_causal.transformer.causal_model import CausalLTXModel


class CausalLTX2DiffusionWrapper(nn.Module):
    """Own the causal generator and create empty inference KV caches."""

    def __init__(self, model: CausalLTXModel, **_: Any) -> None:
        super().__init__()
        self.model = model

    def init_kv_cache(self) -> Any:
        from ltx_causal.transformer.kv_cache import KVCache, LayerKVCache

        return KVCache(
            layers=[LayerKVCache() for _ in range(len(self.model.transformer_blocks))]
        )
