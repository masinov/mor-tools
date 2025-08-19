# model_objectives.py
"""
Objective-management mix-in.

Exposed by _ObjectivesMixin:
  - Minimize, Maximize
  - ObjectiveInfo dataclass
  - Single-active-objective helpers
"""

from __future__ import annotations
from typing import List, Optional

import ortools.sat.python.cp_model as _cp

from dataclasses import dataclass

@dataclass
class ObjectiveInfo:
    """Thin wrapper around an objective with metadata."""
    objective_type: str  # "Minimize" or "Maximize"
    linear_expr: _cp.LinearExprT
    name: str
    enabled: bool = True

class _ObjectivesMixin(_cp.CpModel):
    """Objective creation and single-active-objective enforcement."""

    # ------------------------------------------------------------------
    # Public creators
    # ------------------------------------------------------------------
    def Minimize(self, obj: _cp.LinearExprT, name: Optional[str] = None) -> None:
        objectives = self._ensure_objectives()
        name = name or f"minimize_{len(objectives)}"
        if any(o.name == name for o in objectives):
            raise ValueError(f"Objective name '{name}' already exists")

        # Disable all existing objectives
        for o in objectives:
            o.enabled = False

        # Add new objective as enabled
        objectives.append(ObjectiveInfo("Minimize", obj, name))
        super().Minimize(obj)

    def Maximize(self, obj: _cp.LinearExprT, name: Optional[str] = None) -> None:
        objectives = self._ensure_objectives()
        name = name or f"maximize_{len(objectives)}"
        if any(o.name == name for o in objectives):
            raise ValueError(f"Objective name '{name}' already exists")

        # Disable all existing objectives
        for o in objectives:
            o.enabled = False

        # Add new objective as enabled
        objectives.append(ObjectiveInfo("Maximize", obj, name))
        super().Maximize(obj)

    # ------------------------------------------------------------------
    # Single-active-objective helpers
    # ------------------------------------------------------------------
    def enable_objective(self, name: str) -> None:
        """Enable exactly one objective by name; disable all others."""
        objectives = self._ensure_objectives()
        found = False
        for obj in objectives:
            if obj.name == name:
                obj.enabled = True
                found = True
            else:
                obj.enabled = False
        if not found:
            raise ValueError(f"Objective '{name}' not found")

    def disable_objective(self, name: str) -> None:
        """Disable one objective by name."""
        for obj in self._ensure_objectives():
            if obj.name == name:
                obj.enabled = False
                return
        raise ValueError(f"Objective '{name}' not found")

    def get_enabled_objective(self) -> Optional[ObjectiveInfo]:
        """Return the single enabled objective (or None)."""
        enabled = [obj for obj in self._ensure_objectives() if obj.enabled]
        if len(enabled) > 1:
            raise RuntimeError(
                f"Multiple objectives enabled ({len(enabled)}); only one is allowed."
            )
        return enabled[0] if enabled else None

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------
    def _ensure_objectives(self) -> List[ObjectiveInfo]:
        """Lazy-initialise the objective registry for mix-in safety."""
        if not hasattr(self, "_objectives"):
            self._objectives: List[ObjectiveInfo] = []
        return self._objectives