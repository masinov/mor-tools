# model_debug.py
"""
Debugging and introspection mix-in.

Exposed by _DebugMixin:
  - debug_infeasible           (MIS finder)
  - summary / validate_model   (quick health checks)
  - All get_* helpers          (lists & filters)
  - create_relaxed_copy        (random relaxation)
  - create_subset_copy         (constraint subset)
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence

import ortools.sat.python.cp_model as _cp

from .constraints import ConstraintInfo


class _DebugMixin:
    """Introspection, MIS, relaxation & subset utilities."""

    # ------------------------------------------------------------------
    # High-level debugging
    # ------------------------------------------------------------------
    def debug_infeasible(
        self,
        solver: Optional[_cp.CpSolver] = None,
        **solver_params,
    ) -> Dict[str, Any]:
        """
        Find a minimal set of constraints to disable to make the model feasible.

        Returns
        -------
        dict with keys:
            status, feasible, disabled_constraints, total_disabled, method
        """
        solver = solver or _cp.CpSolver()

        # apply any user-supplied solver parameters
        for k, v in solver_params.items():
            if not hasattr(solver.parameters, k):
                raise ValueError(f"Unknown solver parameter: {k}")
            setattr(solver.parameters, k, v)

        # quick feasibility check
        simple_status = self._solve(solver)
        if simple_status in (_cp.OPTIMAL, _cp.FEASIBLE):
            return {
                "status": simple_status,
                "feasible": True,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "already_feasible",
            }

        # build MIS sub-model -------------------------------------------------
        mis_model = self.clone()
        disable_vars: List[_cp.IntVar] = []
        name_to_disable: Dict[str, _cp.IntVar] = {}

        for c_name in self.get_enabled_constraints():
            info = mis_model._constraints[c_name]
            disable_var = mis_model.NewBoolVar(f"disable_{c_name}")
            name_to_disable[c_name] = disable_var
            mis_model.Add(disable_var + info.enable_var == 1)
            disable_vars.append(disable_var)

        if not disable_vars:
            return {
                "status": _cp.INFEASIBLE,
                "feasible": False,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "no_constraints_to_disable",
            }

        mis_model.Minimize(sum(disable_vars))
        mis_status = solver.Solve(mis_model)

        if mis_status not in (_cp.OPTIMAL, _cp.FEASIBLE):
            return {
                "status": mis_status,
                "feasible": False,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "mis_solver_failed",
            }

        disabled = [
            c_name
            for c_name, dv in name_to_disable.items()
            if solver.Value(dv) == 1
        ]

        # verify the proposed set on the original model
        saved_states = {c: info.enabled for c, info in self._constraints.items()}
        for c in disabled:
            self.disable_constraint(c)
        verify_status = self._solve(solver)

        # restore original state
        for c, was_enabled in saved_states.items():
            if was_enabled:
                self.enable_constraint(c)
            else:
                self.disable_constraint(c)

        return {
            "status": verify_status,
            "feasible": verify_status in (_cp.OPTIMAL, _cp.FEASIBLE),
            "disabled_constraints": disabled,
            "total_disabled": len(disabled),
            "method": "minimal_infeasible_set",
            "objective_value": solver.ObjectiveValue()
            if mis_status == _cp.OPTIMAL
            else None,
        }

    # ------------------------------------------------------------------
    # Quick health checks
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """One-stop overview of the model."""
        c_types: Dict[str, int] = {}
        for info in self._ensure_constraints().values():
            c_types[info.constraint_type] = c_types.get(info.constraint_type, 0) + 1

        v_types: Dict[str, int] = {}
        for info in getattr(self, "_variables", {}).values():
            v_types[info.var_type] = v_types.get(info.var_type, 0) + 1

        all_tags = set().union(
            *(info.tags for info in self._ensure_constraints().values())
        )

        return {
            "total_constraints": len(self._ensure_constraints()),
            "enabled_constraints": len(self.get_enabled_constraints()),
            "disabled_constraints": len(self.get_disabled_constraints()),
            "constraint_types": c_types,
            "total_variables": len(getattr(self, "_variables", {})),
            "variable_types": v_types,
            "total_objectives": len(getattr(self, "_objectives", [])),
            "enabled_objective": self.get_enabled_objective() is not None,
            "all_tags": list(all_tags),
        }

    def validate_model(self) -> Dict[str, Any]:
        """Basic diagnostics: disabled constraints, multiple objectives, etc."""
        issues: List[str] = []
        warnings: List[str] = []

        disabled = self.get_disabled_constraints()
        if disabled:
            warnings.append(f"{len(disabled)} constraints are disabled")

        unused_vars: List[str] = []  # placeholder for future analysis

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "disabled_constraints": disabled,
            "unused_variables": unused_vars,
        }

    # ------------------------------------------------------------------
    # Model relaxation & subset helpers
    # ------------------------------------------------------------------
    def create_relaxed_copy(
        self,
        relaxation_factor: float = 0.1,
    ) -> Any:  # -> EnhancedCpModel via mix-in
        """
        Return a relaxed copy by randomly disabling `relaxation_factor`
        fraction of currently-enabled constraints.
        """
        if not (0.0 <= relaxation_factor <= 1.0):
            raise ValueError("relaxation_factor must be between 0.0 and 1.0")

        relaxed = self.clone()
        enabled = relaxed.get_enabled_constraints()
        k = int(len(enabled) * relaxation_factor)
        if k:
            to_disable = random.sample(enabled, k)
            relaxed.disable_constraints(to_disable)
        return relaxed

    def create_subset_copy(
        self,
        constraint_names: Sequence[str],
    ) -> Any:  # -> EnhancedCpModel via mix-in
        """
        Return a copy with *only* the listed constraints enabled.
        """
        subset = self.clone()
        subset.disable_constraints(subset.get_constraint_names())
        subset.enable_constraints(list(constraint_names))
        return subset

    # ------------------------------------------------------------------
    # Internal helper (safe mix-in registry access)
    # ------------------------------------------------------------------
    def _ensure_constraints(self) -> Dict[str, ConstraintInfo]:
        if not hasattr(self, "_constraints"):
            self._constraints: Dict[str, ConstraintInfo] = {}
        return self._constraints