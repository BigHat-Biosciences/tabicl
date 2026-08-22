"""Tests for shared torch device resolution helpers."""

from __future__ import annotations

import sys

import pytest
import torch

from tabicl import _torch_devices
from tabicl._torch_devices import (
    DEFAULT_DEVICE_PREFERENCE,
    EXTERNAL_BACKEND_MODULES,
    backend_is_available,
    external_backend_is_available,
    resolve_torch_device,
)


@pytest.fixture(autouse=True)
def _clear_external_backend_cache():
    """``external_backend_is_available`` is lru_cached; isolate each test."""
    external_backend_is_available.cache_clear()
    yield
    external_backend_is_available.cache_clear()


def test_cpu_is_always_available():
    assert backend_is_available("cpu") is True


def test_unknown_device_type_is_unavailable():
    assert backend_is_available("definitely-not-a-backend") is False


def test_xla_query_returns_bool_without_raising():
    """Querying ``xla`` must be safe whether or not torch_xla is installed."""
    assert isinstance(backend_is_available("xla"), bool)


def test_xla_absent_from_default_preference():
    """Default device resolution must never import torch_xla.

    Importing ``torch_xla`` initializes the PJRT runtime as a side effect, so
    probing it during ``device=None`` resolution would penalise (or disturb) any
    host that merely has the package installed. ``xla`` is therefore opt-in via an
    explicit ``device="xla"``. This test guards that design decision.
    """
    assert "xla" not in DEFAULT_DEVICE_PREFERENCE
    assert "xla" in EXTERNAL_BACKEND_MODULES


class _FakeXlaModule:
    def __init__(self, count):
        self._count = count

    def device_count(self):
        return self._count


class _FakeXlaModuleNoDeviceCount:
    pass


def _install_fake_backend(monkeypatch, module, name="fake_external_backend"):
    """Register a fake external backend module under device type ``ext``."""
    monkeypatch.setitem(EXTERNAL_BACKEND_MODULES, "ext", name)
    monkeypatch.setitem(sys.modules, name, module)
    external_backend_is_available.cache_clear()


def test_external_backend_available_when_devices_present(monkeypatch):
    _install_fake_backend(monkeypatch, _FakeXlaModule(2))
    assert backend_is_available("ext") is True


def test_external_backend_unavailable_when_no_devices(monkeypatch):
    _install_fake_backend(monkeypatch, _FakeXlaModule(0))
    assert backend_is_available("ext") is False


def test_external_backend_available_without_device_count_api(monkeypatch):
    """A successful import is sufficient when the module exposes no device count.

    ``torch_xla`` raises on import when no runtime can be initialized, so importing
    it cleanly is itself evidence of availability.
    """
    _install_fake_backend(monkeypatch, _FakeXlaModuleNoDeviceCount())
    assert backend_is_available("ext") is True


def test_external_backend_unavailable_when_device_count_raises(monkeypatch):
    class _Raising:
        def device_count(self):
            raise RuntimeError("no runtime")

    _install_fake_backend(monkeypatch, _Raising())
    assert backend_is_available("ext") is False


def test_torch_native_backend_takes_precedence_over_external(monkeypatch):
    """A backend reachable as ``torch.<type>`` must not consult the module map."""
    sentinel = {"called": False}

    def _boom(device_type):
        sentinel["called"] = True
        return True

    monkeypatch.setattr(_torch_devices, "external_backend_is_available", _boom)
    backend_is_available("cuda")
    assert sentinel["called"] is False


def test_resolve_torch_device_passes_through_explicit_device():
    assert resolve_torch_device("cpu") == torch.device("cpu")
    assert resolve_torch_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_torch_device_none_returns_available_backend():
    resolved = resolve_torch_device(None)
    assert isinstance(resolved, torch.device)
    assert backend_is_available(resolved.type)
