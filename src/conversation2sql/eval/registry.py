"""
Registry for dataset readers, predictors, and scorers.

Each registry validates that a decorated class implements the required protocol
methods before admission.  Decorators are applied at class-definition time, so
any module that imports a concrete implementation automatically registers it.

Usage
-----
    from conversation2sql.eval.registry import register_reader, reader_registry

    @register_reader
    class MyReader:
        @task
        def read(self) -> list[EvalSample]: ...

    # Later, instantiate by name:
    reader = reader_registry.build("MyReader")
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry with protocol-method validation."""

    def __init__(self, name: str, required_methods: list[str]) -> None:
        self._name = name
        self._required_methods = required_methods
        self._registry: dict[str, type] = {}

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def register(self, cls: type[T]) -> type[T]:
        """Class decorator — validate protocol compliance then register.

        Raises
        ------
        TypeError
            If ``cls`` is missing any of the required protocol methods.
        """
        missing = [
            m for m in self._required_methods
            if not callable(getattr(cls, m, None))
        ]
        if missing:
            raise TypeError(
                f"Cannot register '{cls.__name__}' in the '{self._name}' registry: "
                f"missing required method(s): {missing}"
            )
        self._registry[cls.__name__] = cls
        return cls

    # ------------------------------------------------------------------
    # Lookup / instantiation
    # ------------------------------------------------------------------

    def get(self, name: str) -> type:
        """Return the registered class for *name* (not an instance)."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in the '{self._name}' registry. "
                f"Available: {list(self._registry)}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class by name, forwarding *kwargs* to ``__init__``."""
        return self.get(name)(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available(self) -> list[str]:
        """Names of all registered classes."""
        return list(self._registry)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry(name={self._name!r}, entries={self.available})"


# ---------------------------------------------------------------------------
# Module-level registry instances (one per protocol)
# ---------------------------------------------------------------------------

reader_registry = Registry("readers", required_methods=["read"])
predictor_registry = Registry("predictors", required_methods=["predict"])
scorer_registry = Registry("scorers", required_methods=["score"])
