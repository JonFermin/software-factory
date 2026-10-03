"""Kind handlers: one module per kind (``kinds/<kind>.py``), each exposing a module-level ``HANDLER``
object that satisfies ``KindHandler`` (``kind`` attribute + ``async run(action, deps) -> Outcome``).

Handlers are imported lazily by ``get_handler`` so the dispatcher can start even when a kind's
module is missing (that kind then fails with a clear error instead of taking the tick down).
"""

from __future__ import annotations

import importlib

from ..actions import Kind
from .base import AuthFailure, Deps, KindHandler, Outcome, ProviderUnavailable

__all__ = [
    "HANDLER_MODULES",
    "AuthFailure",
    "Deps",
    "KindHandler",
    "KindHandlerError",
    "Outcome",
    "ProviderUnavailable",
    "get_handler",
]

HANDLER_MODULES: dict[str, str] = {kind.value: f"{__name__}.{kind.value}" for kind in Kind}


class KindHandlerError(RuntimeError):
    pass


def get_handler(kind: Kind | str) -> KindHandler:
    """Import ``software_factory.kinds.<kind>`` and return its ``HANDLER``."""
    value = kind.value if isinstance(kind, Kind) else str(kind).strip().lower()
    module_name = HANDLER_MODULES.get(value)
    if module_name is None:
        raise KindHandlerError(f"unknown kind {value!r}; expected one of {sorted(HANDLER_MODULES)}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise KindHandlerError(
                f"no handler module for kind {value!r}: {module_name} is missing "
                f"(expected a module exposing HANDLER)"
            ) from exc
        raise
    handler = getattr(module, "HANDLER", None)
    if handler is None or not callable(getattr(handler, "run", None)):
        raise KindHandlerError(
            f"{module_name} does not expose a HANDLER with an async run(action, deps) method"
        )
    return handler
