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


def _torch_compile(fn: Callable) -> Callable:
    """Apply torch.compile with inference-optimized settings.

    Uses ``reduce-overhead`` (CUDA-graph replay) on CUDA for minimal
    kernel-launch overhead.  On non-CUDA devices (MPS, CPU) we still
    use ``fullgraph=True`` so that compilation errors surface at the
    first call rather than lazily on a rare code-path hours later.
    """
    import torch
    if torch.cuda.is_available():
        return torch.compile(fn, mode="reduce-overhead", fullgraph=True)
    return torch.compile(fn, fullgraph=True)


class TorchAdapter:
    """Wrap a pure-PyTorch kernel so it can be called with Triton syntax.

    The adapter:
      1. Ignores the ``[grid]`` subscript.
      2. Strips the ``_ptr`` suffix from tensor argument names.
      3. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
      4. Passes scalars directly (no buffer conversion needed).

    On the first call (or when the set of keyword arguments changes), a
    specialised caller function is generated via ``exec`` that maps caller
    kwargs directly to kernel parameters **without** building an intermediate
    dict.  Subsequent calls with the same kwarg keys hit the cached fast-path.
    """

    __slots__ = (
        "_kernel",
        "_kernel_raw",
        "_param_names",
        "_key_map",
        "_cached_key",
        "_fast_caller",
    )

    def __init__(self, kernel_func: Callable, *, compile: bool = True):
        import inspect
        self._kernel_raw = kernel_func
        sig = inspect.signature(kernel_func)
        self._param_names: set[str] = set(sig.parameters.keys())
        self._key_map: dict[str, str] = {}  # caller key -> kernel param name
        self._cached_key: tuple[str, ...] | None = None
        self._fast_caller: Callable | None = None
        if compile:
            self._kernel = _torch_compile(kernel_func)
        else:
            self._kernel = kernel_func

    # ------------------------------------------------------------------
    # Key resolution (lazy-cached)
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        """Map a caller kwarg name to the kernel parameter name.

        Returns the mapped name, or an empty string if the key should be
        dropped (e.g. ``BLOCK_SIZE``).
        """
        try:
            return self._key_map[key]
        except KeyError:
            pass
        base_key = key[:-4] if key.endswith("_ptr") else key
        if base_key in self._param_names:
            self._key_map[key] = base_key
            return base_key
        if key in self._param_names:
            self._key_map[key] = key
            return key
        self._key_map[key] = ""  # sentinel: drop this kwarg
        return ""

    # ------------------------------------------------------------------
    # Fast-caller generation
    # ------------------------------------------------------------------

    def _build_fast_caller(self, kwargs_keys: tuple[str, ...]) -> Callable:
        """Generate a specialised function that avoids per-call dict building.

        The generated function has the form::

            def _fast(kw, _k=_kernel):
                return _k(param_a=kw['a_ptr'], param_b=kw['b_ptr'], ...)
        """
        pairs = []
        for k in kwargs_keys:
            mapped = self._resolve_key(k)
            if mapped:
                pairs.append((k, mapped))

        args_str = ", ".join(f"{m}=_kw['{k}']" for k, m in pairs)
        code = f"def _fast(_kw, _k=_kernel): return _k({args_str})"
        ns: dict[str, Any] = {"_kernel": self._kernel}
        exec(code, ns)  # noqa: S102 – generated code is fully controlled
        return ns["_fast"]

    def __call__(self, **kwargs: Any):
        key = tuple(kwargs.keys())  # dict ordering is guaranteed (Python 3.7+)
        if key != self._cached_key:
            self._fast_caller = self._build_fast_caller(key)
            self._cached_key = key
        return self._fast_caller(kwargs)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __getitem__(self, grid):
        """Accept ``kernel[grid]`` syntax; grid is unused."""
        return self


def adapt_torch_kernel(kernel_func: Callable, *, compile: bool = True) -> TorchAdapter:
    """Create a Triton-compatible adapter for a pure-PyTorch kernel.

    Args:
        compile: If False the kernel is used as-is (useful for log / diagnostic
                 kernels that contain ``.item()`` calls which break the graph).
    """
    return TorchAdapter(kernel_func, compile=compile)
