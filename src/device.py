"""Single accelerator pick for every round-path script.

Priority: XLA TPU (torch_xla importable and the PJRT runtime reports TPU)
-> CUDA -> MPS -> CPU. Set TRAINTAI_DEVICE=xla to force the XLA path (how
the CPU PJRT client exercises it without TPU hardware); any other value is
passed through as a torch device string.
"""

import os

import torch


def _xla_device():
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


def get_device():
    override = os.environ.get("TRAINTAI_DEVICE")
    if override:
        return _xla_device() if override == "xla" else torch.device(override)
    try:
        import torch_xla.runtime as xr
        if xr.device_type() == "TPU":
            return _xla_device()
    except Exception:
        pass
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_xla(device):
    return str(device).startswith("xla")


def optimizer_step(opt, device):
    """opt.step() that also closes the XLA graph step when on a TPU."""
    if is_xla(device):
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(opt)
    else:
        opt.step()


def mark_step(device):
    """Close the XLA graph step after parameter updates that bypass the
    optimizer (the muon block in train.py). No-op elsewhere."""
    if is_xla(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
