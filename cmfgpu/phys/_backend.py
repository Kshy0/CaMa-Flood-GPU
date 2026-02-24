# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Kernel backend selection and adapter for CaMa-Flood-GPU.

Set environment variable CMFGPU_BACKEND to choose between 'triton' (default)
and 'torch'.

    export CMFGPU_BACKEND=torch

When 'torch' is selected, a thin adapter wraps each PyTorch kernel so it
can be called with the existing Triton calling convention used in the model:

    kernel[grid](ptr_arg_ptr=tensor, scalar_arg=value, BLOCK_SIZE=bs)

The adapter:
  1. Ignores the ``[grid]`` subscript (grid is not needed for PyTorch).
  2. Strips the ``_ptr`` suffix from tensor argument names.
  3. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
  4. Passes scalars directly (no buffer conversion needed).
"""

import os
import warnings
from typing import Any, Callable


def _resolve_backend() -> str:
    """Resolve kernel backend, falling back to 'torch' when CUDA is unavailable."""
    explicit = os.environ.get("CMFGPU_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return "triton"
    except ImportError:
        pass
    warnings.warn(
        "CUDA is not available – automatically selecting the 'torch' backend. "
        "Set CMFGPU_BACKEND=triton to override.",
        stacklevel=2,
    )
    return "torch"


KERNEL_BACKEND: str = _resolve_backend()
os.environ.setdefault("CMFGPU_BACKEND", KERNEL_BACKEND)


class TorchAdapter:
    """Wrap a pure-PyTorch kernel so it can be called with Triton syntax.

    The adapter:
      1. Ignores the ``[grid]`` subscript.
      2. Strips the ``_ptr`` suffix from tensor argument names.
      3. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
      4. Passes scalars directly (no buffer conversion needed).

    The wrapped function should be ``torch.compile``-friendly.
    """

    def __init__(self, kernel_func: Callable, *, compile: bool = True):
        import inspect
        self._kernel_raw = kernel_func
        sig = inspect.signature(kernel_func)
        self._param_names = set(sig.parameters.keys())
        if compile:
            self._kernel = _torch_compile(kernel_func)
        else:
            self._kernel = kernel_func

    def __getitem__(self, grid):
        return self

    def __call__(self, **kwargs: Any):
        import torch
        adapted: dict[str, Any] = {}

        for key, value in kwargs.items():
            base_key = key[:-4] if key.endswith("_ptr") else key

            if base_key in self._param_names:
                adapted[base_key] = value
            elif key in self._param_names:
                adapted[key] = value
            # else: silently drop (BLOCK_SIZE, batch flags, etc.)

        return self._kernel(**adapted)


def _torch_compile(fn: Callable) -> Callable:
    """Apply torch.compile with inference-optimized settings.

    Uses ``reduce-overhead`` (CUDA-graph replay) on CUDA for minimal
    kernel-launch overhead.  Falls back to the default compile mode on
    non-CUDA devices (MPS, CPU) where CUDA graphs are not supported.
    """
    import torch
    if torch.cuda.is_available():
        return torch.compile(fn, mode="reduce-overhead", fullgraph=True)
    return torch.compile(fn)


def adapt_torch_kernel(kernel_func: Callable, *, compile: bool = True) -> TorchAdapter:
    """Create a Triton-compatible adapter for a pure-PyTorch kernel.

    Args:
        compile: If False the kernel is used as-is (useful for log / diagnostic
                 kernels that contain ``.item()`` calls which break the graph).
    """
    return TorchAdapter(kernel_func, compile=compile)
