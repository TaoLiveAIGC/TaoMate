"""Layer streaming wrapper for memory-efficient inference.
Keeps most transformer/decoder layers on CPU pinned memory and streams them
to GPU on demand, using a secondary CUDA stream to prefetch upcoming layers
so that data transfer overlaps with compute.
General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (e.g. ``transformer_blocks``, ``layers``).
Each layer is evicted back to CPU immediately after its forward completes,
and prefetch uses modular indexing so the last layer's prefetch wraps around
to prepare early layers for the next forward pass.
Example
-------
>>> model = build_my_model(device=torch.device("cpu"))
>>> model = LayerStreamingWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)            # hooks handle layer streaming
>>> model.teardown()               # move everything back to CPU
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class _LayerStore:
    """Manages CPU-pinned copies of layer parameters/buffers.
    Tracks which layers currently reside on GPU so the prefetcher and evictor
    can make correct decisions.
    """

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        # CPU-pinned copies keyed by (layer_idx, param_name)
        self._pinned: list[dict[str, torch.Tensor]] = []
        self._on_gpu: set[int] = set()

        for layer in layers:
            pinned: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                pinned_tensor = tensor.data.pin_memory()
                tensor.data = pinned_tensor
                pinned[name] = pinned_tensor
            self._pinned.append(pinned)

    def _check_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def move_to_gpu(self, idx: int, layer: nn.Module, *, non_blocking: bool = False) -> None:
        """Move layer *idx* parameters from pinned CPU to *target_device*."""
        self._check_idx(idx)
        if idx in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name].to(self.target_device, non_blocking=non_blocking)
        self._on_gpu.add(idx)

    def evict_to_cpu(self, idx: int, layer: nn.Module) -> None:
        """Swap layer *idx* parameters back to their pinned CPU copies."""
        self._check_idx(idx)
        if idx not in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name]
        self._on_gpu.discard(idx)

    def cleanup(self) -> None:
        """Release all pinned memory references.
        After this call, the pinned tensors can be garbage-collected once
        the layer parameters (which still reference them via ``.data``) are
        also released (e.g. via ``.to("meta")``).
        """
        for pinned_dict in self._pinned:
            pinned_dict.clear()
        self._pinned.clear()


class _AsyncPrefetcher:
    """Issues H2D transfers on a dedicated CUDA stream.
    Uses per-layer CUDA events so that the compute stream only waits for the
    specific layer it needs, not all pending transfers.
    """

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        """Begin async transfer of layer *idx* to GPU (no-op if already there)."""
        if self._store.is_on_gpu(idx) or idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event

    def wait(self, idx: int) -> None:
        """Block the compute stream until layer *idx* transfer is complete."""
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        """Drain pending work and release CUDA stream/event resources."""
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None


class LayerStreamingWrapper(nn.Module):
    """Wraps a model to stream its sequential layers between CPU and GPU.
    Each layer is evicted immediately after its forward completes, and
    prefetch wraps around using modular indexing so the end of one forward
    pass prepares early layers for the next.
    Parameters
    ----------
    model:
        The model to wrap, with all parameters on **CPU**.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential layers
        (e.g. ``"transformer_blocks"`` or ``"model.language_model.layers"``).
    target_device:
        The GPU device to use for compute.
    prefetch_count:
        How many layers ahead to prefetch.  The maximum number of layers on
        GPU at once is ``1 + prefetch_count``.  Must be >= 1.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 2,
    ) -> None:
        if prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1")
        super().__init__()
        # Store the wrapped model as a submodule so parameters are discoverable.
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        # Clamp: no point prefetching more than num_layers - 1 (the rest are evicted).
        self._prefetch_count = min(prefetch_count, len(self._layers) - 1)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        # 1. Build the pinned CPU store (copies all layer tensors to pinned memory).
        self._store = _LayerStore(self._layers, self._target_device)

        # 2. Move all NON-layer params/buffers to GPU.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        # 3. Pre-load the first (1 + prefetch_count) layers synchronously.
        for idx in range(min(self._prefetch_count + 1, len(self._layers))):
            self._store.move_to_gpu(idx, self._layers[idx])

        # 4. Create the async prefetcher and register hooks.
        self._prefetcher = _AsyncPrefetcher(self._store, self._layers)
        self._register_hooks()

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        def _pre_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Wait only for THIS layer's H2D transfer (not all pending ones).
            self._prefetcher.wait(idx)
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx, module)

            # Record that the compute stream will read these weight tensors.
            # They were allocated on the prefetch stream, so without this the
            # caching allocator would allow the prefetch stream to reuse their
            # memory immediately after eviction — even if the compute kernel
            # that reads them hasn't finished yet.
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(compute_stream)

            # Kick off prefetch for upcoming layers (wraps around for next pass).
            for offset in range(1, self._prefetch_count + 1):
                self._prefetcher.prefetch((idx + offset) % num_layers)

        def _post_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            _output: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Evict this layer immediately — its computation is done.
            self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        """Remove hooks, release pinned memory, and move parameters back to CPU.
        After this call the wrapper is inert: hooks are removed, the prefetch
        stream is drained and destroyed, all parameters reside on regular
        (non-pinned) CPU memory, and the ``_LayerStore`` pinned-tensor cache is
        cleared.  Callers should still follow up with ``.to("meta")`` to release
        the CPU copies if the model is no longer needed.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Drain all in-flight async H2D copies, then release stream resources.
        # Without the synchronize, clearing the stream/events can trigger
        # use-after-free at the CUDA driver level.
        torch.cuda.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

        # Move everything to CPU.
        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        # Release pinned memory.  After evict_to_cpu() the layer parameters
        # still reference the pinned tensors (since .to("cpu") on a pinned
        # tensor is a no-op).  The caller is expected to follow up with
        # .to("meta") to drop the param refs; cleanup() drops the store's refs.
        self._store.cleanup()

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model.
        This allows calling methods like ``encode()`` on a wrapped
        GemmaTextEncoder without the caller needing to know about the wrapper.
        ``nn.Module.__getattr__`` is only called when normal attribute lookup
        fails, so ``_model``, ``_store``, etc. are found first via ``__dict__``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
