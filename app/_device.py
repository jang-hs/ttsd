"""Cross-platform PyTorch device selection.

Picks the best available accelerator at runtime:
  - CUDA on Linux/Windows boxes with an NVIDIA GPU
  - MPS on Apple Silicon
  - CPU everywhere else

Override with the TTSD_DEVICE env var (e.g. TTSD_DEVICE=cpu).
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def pick_device() -> "torch.device":  # type: ignore[name-defined]
    import torch

    forced = os.environ.get("TTSD_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def pick_dtype() -> "torch.dtype":  # type: ignore[name-defined]
    """Match dtype to device: bf16 on CUDA, fp32 elsewhere (MPS bf16 is patchy)."""
    import torch

    dev = pick_device()
    if dev.type == "cuda":
        return torch.bfloat16
    return torch.float32
