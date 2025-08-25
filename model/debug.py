# debug.py
"""
Debugging and introspection mix-in.

Exposed by _DebugMixin:
  - debug_infeasible           (MIS finder)
  - summary / validate_model   (quick health checks)
  - create_relaxed_copy        (random relaxation)
  - create_subset_copy         (constraint subset)
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Any

import ortools.sat.python.cp_model as _cp

from constraints import ConstraintInfo
from variables import VariableInfo
from objectives import ObjectiveInfo

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import EnhancedCpModel

class _DebugMixin:
    """Introspection, MIS, relaxation & subset utilities."""

    # ------------------------------------------------------------------
    # High-level debugging
    # ------------------------------------------------------------------
    def _solve(self, solver: _cp.CpSolver) -> int:
        """
        (Internal) Creates a clean solving model from the current state 
        and solves it. Respects enabled/disabled constraints.
        """
        # _create_solving_model builds a new CpModel with only the enabled constraints
        solving_model = self._create_solving_model()

        return solver.Solve(solving_model)
    
    def debug_infeasible(
        self,
        solver: Optional[_cp.CpSolver] = None,
        **solver_params,
    ) -> Dict[str, object]:
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

        # 1. First check
        simple_status = self._solve(solver)
        
        if simple_status in (_cp.OPTIMAL, _cp.FEASIBLE):
            return {
                "status": simple_status,
                "feasible": True,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "already_feasible",
            }

        # 2. Build the specialized MIS (Minimal Infeasible Set) sub-model
        mis_model = _cp.CpModel()
        var_mapping = {}

        # Recreate all user variables in the mis_model
        for name, info in self._variables.items():
            new_var = self._recreate_variable_in_model_solver(info, {}, mis_model)
            var_mapping[name] = new_var

        disable_vars: List[_cp.IntVar] = []
        name_to_disable: Dict[str, _cp.IntVar] = {}

        # Iterate through the constraints that are currently enabled in the main model
        for c_name in self.get_enabled_constraints():
            info = self._constraints[c_name]

            # A. Recreate the original constraint in the mis_model
            recreated_ct = self._recreate_constraint_in_model_solver(info, var_mapping, mis_model)

            # B. Create the control variables *in-situ* within the mis_model
            disable_var = mis_model.NewBoolVar(f"disable_{c_name}")

            # C. Enforce the constraint only if its 'disable' var is false.
            recreated_ct.OnlyEnforceIf(disable_var.Not())

            disable_vars.append(disable_var)
            name_to_disable[c_name] = disable_var

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
    def summary(self) -> Dict[str, object]:
        """One-stop overview of the model."""
        constraints = self._ensure_constraints()
        variables = self._ensure_variables()
        objectives = self._ensure_objectives()

        c_types: Dict[str, int] = {}
        for info in constraints.values():
            c_types[info.constraint_type] = c_types.get(info.constraint_type, 0) + 1

        v_types: Dict[str, int] = {}
        for info in variables.values():
            v_types[info.var_type] = v_types.get(info.var_type, 0) + 1

        all_tags = set().union(*(info.tags for info in constraints.values())) if constraints else set()

        return {
            "total_constraints": len(constraints),
            "enabled_constraints": len(self.get_enabled_constraints()),
            "disabled_constraints": len(self.get_disabled_constraints()),
            "constraint_types": c_types,
            "total_variables": len(variables),
            "variable_types": v_types,
            "total_objectives": len(objectives),
            "enabled_objective": self.get_enabled_objective() is not None,
            "all_tags": list(all_tags),
        }

    def validate_model(self) -> Dict[str, Any]:
        """Basic diagnostics: disabled constraints, multiple objectives, OR-Tools validation, etc."""
        issues: List[str] = []
        warnings: List[str] = []

        # Disabled constraints
        disabled = self.get_disabled_constraints()
        if disabled:
            warnings.append(f"{len(disabled)} constraints are disabled")

        # Multiple objectives
        if len(getattr(self, "_objectives", [])) > 1:
            enabled_objectives = [
                obj.name for obj in self._objectives if obj.enabled
            ]
            if len(enabled_objectives) > 1:
                issues.append(
                    f"Multiple enabled objectives detected: {enabled_objectives}"
                )

        # OR-Tools internal validation
        ortools_validation = super().Validate()  # call base CpModel.Validate()
        if ortools_validation:
            issues.append(f"OR-Tools validation: {ortools_validation.strip()}")

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
    ) -> "EnhancedCpModel":
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
    ) -> "EnhancedCpModel":
        """
        Return a copy with *only* the listed constraints enabled.
        """
        subset = self.clone()
        subset.disable_constraints(subset.get_constraint_names())
        subset.enable_constraints(list(constraint_names))
        return subset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_constraints(self) -> Dict[str, ConstraintInfo]:
        if not hasattr(self, "_constraints"):
            self._constraints: Dict[str, ConstraintInfo] = {}
        return self._constraints

    def _ensure_variables(self) -> Dict[str, VariableInfo]:
        if not hasattr(self, "_variables"):
            self._variables: Dict[str, VariableInfo] = {}
        return self._variables

    def _ensure_objectives(self) -> Dict[str, ObjectiveInfo]:
        if not hasattr(self, "_objectives"):
            self._objectives: Dict[str, ObjectiveInfo] = {}
        return self._objectives
