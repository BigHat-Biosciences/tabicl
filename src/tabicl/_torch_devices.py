"""Shared torch device resolution helpers."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import subprocess
import sys
import warnings
from typing import Optional, Union

import torch

# Preference order when ``device=None``: CUDA → XPU → MPS → CPU.
#
# ``xla`` is deliberately absent. Probing it requires importing ``torch_xla``,
# which initializes the PJRT runtime as a side effect, so it must not happen
# during default device resolution on hosts that merely have the package
# installed. Pass ``device="xla"`` explicitly to select it.
DEFAULT_DEVICE_PREFERENCE = ("cuda", "xpu", "mps", "cpu")

# Backends that do NOT follow the ``torch.<device_type>`` convention, mapped to
# their top-level module. ``torch_xla`` (TPU, AWS Neuron/Inferentia) ships as its
# own distribution rather than as an attribute of ``torch``.
EXTERNAL_BACKEND_MODULES = {"xla": "torch_xla"}

# Virtualized Apple Silicon can silently corrupt MPS ``F.linear`` on 3D inputs.
MPS_NUMERICS_ISSUE_URL = "https://github.com/pytorch/pytorch/issues/192934"


def _sysctl(name: str) -> str | None:
    """Return a sysctl string value, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@functools.lru_cache(maxsize=1)
def mps_possibly_faulty() -> bool:
    """Return whether this macOS host may have broken MPS numerics.

    GitHub Actions macOS arm64 runners are VirtualMac guests (``hw.model`` like
    ``VirtualMac2,1``, CPU brand like ``Apple M1 (Virtual)``). On those hosts
    MPS can silently return incorrect results (PyTorch issue
    https://github.com/pytorch/pytorch/issues/192934). Real Apple Silicon is fine.

    Always runs the hardware identity check on Darwin; returns ``False`` on other
    platforms.
    """
    if sys.platform != "darwin":
        return False

    brand = _sysctl("machdep.cpu.brand_string") or ""
    model = _sysctl("hw.model") or ""
    return "Virtual" in brand or model.startswith("VirtualMac")


@functools.lru_cache(maxsize=None)
def external_backend_is_available(device_type: str) -> bool:
    """Return availability for a backend not exposed as ``torch.<device_type>``.

    ``torch_xla`` is a separate top-level distribution, so the ``torch.<backend>``
    convention cannot reach it. Importing it initializes the PJRT runtime, so this
    is only ever reached for a device type explicitly asked about — ``xla`` is
    absent from :data:`DEFAULT_DEVICE_PREFERENCE` precisely so that default device
    resolution never triggers that import.

    Cached because the import and the device count are both comparatively costly.
    """
    module_name = EXTERNAL_BACKEND_MODULES.get(device_type)
    if module_name is None:
        return False

    module = sys.modules.get(module_name)
    if module is None:
        # Cheap, side-effect-free installed check before paying for the import.
        # Only consulted when the module is not already imported: find_spec raises
        # for an entry in sys.modules that carries no __spec__.
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except (ImportError, ValueError):
            return False

        try:
            module = importlib.import_module(module_name)
        except Exception:
            return False

    device_count = getattr(module, "device_count", None)
    if callable(device_count):
        try:
            return device_count() > 0
        except Exception:
            return False

    # Importable but without a device-count API: treat the import as sufficient
    # rather than claiming unavailable, since torch_xla raises on import when no
    # runtime can be initialized.
    return True


def backend_is_available(device_type: str) -> bool:
    """Return whether ``torch.<device_type>`` reports itself available.

    Uses the usual backend convention where accelerators expose
    ``torch.<backend>.is_available()``. CPU is always available. Backends that do
    not follow that convention (see :data:`EXTERNAL_BACKEND_MODULES`) are resolved
    through their own module instead.
    """
    if device_type == "cpu":
        return True

    backend_api = getattr(torch, device_type, None)
    if backend_api is None:
        return external_backend_is_available(device_type)

    is_available = getattr(backend_api, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def resolve_default_device() -> torch.device:
    """Return the default device: CUDA → XPU → MPS → CPU.

    On virtualized macOS hosts with known-bad MPS numerics, MPS is skipped and
    CPU is used instead (with a warning).
    """
    for device_type in DEFAULT_DEVICE_PREFERENCE:
        if not backend_is_available(device_type):
            continue
        if device_type == "mps" and mps_possibly_faulty():
            warnings.warn(
                "MPS appears to run on virtualized Apple Silicon where PyTorch "
                f"can return incorrect results ({MPS_NUMERICS_ISSUE_URL}). "
                "Falling back to CPU because device=None. Pass device='mps' to "
                "force MPS anyway, or device='cpu' to choose CPU silently.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        return torch.device(device_type)
    return torch.device("cpu")


def resolve_torch_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve ``None``, a device string, or a ``torch.device`` to a concrete device.

    ``None`` selects :func:`resolve_default_device`. Explicit ``mps`` on a
    possibly faulty virtualized Mac keeps MPS but warns and recommends CPU.
    """
    if device is None:
        return resolve_default_device()

    resolved = torch.device(device) if isinstance(device, str) else device
    if resolved.type == "mps" and mps_possibly_faulty():
        warnings.warn(
            "device='mps' was requested on virtualized Apple Silicon where "
            f"PyTorch can return incorrect results ({MPS_NUMERICS_ISSUE_URL}). "
            "Consider passing device='cpu' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved
