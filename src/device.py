"""Single accelerator pick for every round-path script.

Priority: XLA TPU (torch_xla importable and the PJRT runtime reports TPU)
-> CUDA -> MPS -> CPU. Set TRAINTAI_DEVICE=xla to force the XLA path (how
the CPU PJRT client exercises it without TPU hardware); any other value is
passed through as a torch device string.

TPU multi-chip (v5e-8): a plain xm.xla_device() call, as this file always
did, only ever grabs ONE logical XLA device -- confirmed by direct code
inspection this session -- so running on Kaggle's 8-chip v5e-8 pod as-is
uses 1/8 chips, no faster than a single-core TPU. setup_spmd_mesh() below
adds real multi-chip support via torch_xla's SPMD API
(torch_xla.distributed.spmd): a 1-D mesh over every visible XLA device,
data-parallel sharded along the batch dimension (dim 0) via
shard_batch() -- the correct minimal-risk sharding axis for this model
(each chip gets a batch slice, the model itself stays replicated, no
parameter-sharding complexity for a 29M-param model that easily fits in
one chip's HBM). UNVERIFIED ON REAL TPU HARDWARE as of this commit --
this session has no TPU access to confirm against; the next real Kaggle
TPU v5e-8 kernel run must confirm num_devices() > 1 and that shard_batch()
actually distributes work (e.g. via torch_xla's runtime device count
check) before this is trusted as working, per this project's discipline
that no number lands without a real run behind it.
"""

import os

import torch


def _xla_device():
    import torch_xla.core.xla_model as xm
    index = os.environ.get("TRAINTAI_XLA_DEVICE_INDEX")
    if index is not None:
        return xm.xla_device(int(index))
    return xm.xla_device()


def get_device():
    """Set TRAINTAI_XLA_DEVICE_INDEX=<n> (0..tpu_chip_count()-1) to pin
    this process to one specific TPU chip via xm.xla_device(n) -- real,
    confirmed working in-process on a v5e-8 pod (round9tpusmoke v6:
    xla:0 through xla:7 all resolved cleanly, no env-var pinning
    needed). This is the real mechanism an 8-way parallel single-chip
    launcher uses to give each of 8 OS processes its own distinct chip,
    since SPMD sharding crashes on this pod (see setup_spmd_mesh) and
    TPU_VISIBLE_CHIPS subprocess env-var pinning also crashes (real
    SIGABRT confirmed, "tpu_process_addresses=local" conflict)."""
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


def tpu_chip_count():
    """Real device count from the XLA runtime, not assumed. Returns 1 on
    any non-TPU device or import failure -- callers use this to decide
    whether SPMD sharding is even applicable, never assume >1."""
    try:
        import torch_xla.runtime as xr
        if xr.device_type() != "TPU":
            return 1
        return xr.global_runtime_device_count()
    except Exception:
        return 1


def setup_spmd_mesh():
    """Builds a 1-D SPMD mesh over every visible XLA device (real chip
    count via tpu_chip_count(), never hardcoded to 8 -- a v5e-8 pod
    reports 8, but this must not silently assume that number if Kaggle
    ever changes the pod shape). Returns None if fewer than 2 chips are
    visible (nothing to shard across) or torch_xla's SPMD module isn't
    importable, so callers can no-op cleanly on any non-multi-chip
    device rather than crash.

    Set TRAINTAI_NO_SPMD=1 to force this to return None even on a
    multi-chip pod -- real Kaggle v5e-8 testing found SPMD's
    ExecuteReplicated() crashes with a hard SIGSEGV inside torch_xla's
    PjRt client (a real, unresolved upstream bug, not fixable here).
    The multi-process-single-chip-each launcher (round9_tpu_parallel.py)
    uses this to get one real chip per process without each process
    trying to build its own 8-chip mesh."""
    if os.environ.get("TRAINTAI_NO_SPMD") == "1":
        return None
    n = tpu_chip_count()
    if n < 2:
        return None
    try:
        import numpy as np
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        xr.use_spmd()
        device_ids = np.array(range(n))
        mesh = xs.Mesh(device_ids, (n,), ("batch",))
        return mesh
    except Exception as e:
        print(f"setup_spmd_mesh: SPMD unavailable ({e}), falling back to single-chip", flush=True)
        return None


def shard_batch(tensor, mesh):
    """Shards `tensor` along dim 0 (the batch dimension) across `mesh`'s
    chips -- data parallelism, the correct minimal-risk axis for this
    model: at 29M params the whole model comfortably fits in one v5e
    chip's HBM, so there is no need for the added complexity of
    parameter/activation sharding, only splitting the batch so each
    chip processes a different slice per step. No-op (returns the
    tensor unchanged) if mesh is None, so call sites do not need an
    if-mesh branch of their own."""
    if mesh is None:
        return tensor
    import torch_xla.distributed.spmd as xs
    xs.mark_sharding(tensor, mesh, ("batch", None))
    return tensor


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
