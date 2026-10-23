"""Pick MPS when it works, and CPU when it does not."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_accelerator(requested: str = "auto") -> str:
    import torch

    choice = requested.lower()
    if choice == "cpu":
        return "cpu"
    if choice not in {"auto", "mps"}:
        raise ValueError("Accelerator must be auto, mps, or cpu.")
    if not torch.backends.mps.is_available():
        if choice == "mps":
            raise RuntimeError("MPS was requested but this machine cannot use it.")
        return "cpu"
    try:
        probe = torch.ones(2, device="mps")
        _ = (probe + probe).cpu()
    except Exception as exc:
        if choice == "mps":
            raise RuntimeError(f"MPS probe failed: {exc}") from exc
        logger.warning("MPS probe failed (%s); using CPU.", exc)
        return "cpu"
    return "mps"
