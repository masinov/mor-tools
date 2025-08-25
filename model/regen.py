# regen.py
"""
Model regeneration / cloning mix-in.

Exposed by _RegenMixin:
  - clone()
  - clear_model()
  - _create_solving_model()
  - _map_expr_to_new_model()
  - _deep_map_expr()
  - _recreate_variable_in_model_metadata()
  - _recreate_variable_in_model_solver()
  - _recreate_constraint_in_model_metadata()
  - _recreate_constraint_in_model_solver()
  - _recreate_objective_in_model_metadata()
  - _recreate_objective_in_model_solver()
"""

from __future__ import annotations
from typing import Any, Dict
import ortools.sat.python.cp_model as _cp
from constraints import ConstraintInfo
from variables import VariableInfo
from objectives import ObjectiveInfo

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import EnhancedCpModel


class _RegenMixin:
    """Helpers for cloning and regenerating EnhancedCpModel instances."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def clone(self) -> "EnhancedCpModel":
        """Create a complete clone with metadata."""
        cloned = type(self)()  # assume EnhancedCpModel subclass
        var_mapping: Dict[str, Any] = {}

        # Recreate variables (metadata-aware)
        for name, info in self._variables.items():
            new_var = cloned._recreate_variable_in_model_metadata(info, var_mapping)
            var_mapping[name] = new_var

        # Recreate constraints (metadata-aware)
        for info in self._constraints.values():
            cloned._recreate_constraint_in_model_metadata(info, var_mapping)

        # Recreate objectives (metadata-aware)
        for obj in self._objectives:
            cloned._recreate_objective_in_model_metadata(obj, var_mapping)

        # Copy counters
        cloned._constraint_counter = self._constraint_counter
        cloned._variable_counter = self._variable_counter

        return cloned

    def clear_model(self) -> None:
        """Clear all model contents and metadata."""
        self.__dict__.update(_cp.CpModel().__dict__)
        self._constraints.clear()
        self._variables.clear()
        self._objectives.clear()
        self._constraint_counter = 0
        self._variable_counter = 0

    # ------------------------------------------------------------------
    # Build a clean CpModel for solving/export
    # ------------------------------------------------------------------
    def _create_solving_model(self) -> _cp.CpModel:
        """Return a clean CpModel with only enabled constraints and active objective."""
        solving_model = _cp.CpModel()
        var_mapping: Dict[str, Any] = {}

        # Variables (solver-only)
        for name, info in self._variables.items():
            new_var = self._recreate_variable_in_model_solver(info, var_mapping, solving_model)
            var_mapping[name] = new_var
  
        # Constraints (solver-only)
        for info in self._constraints.values():
            if info.enabled:
                self._recreate_constraint_in_model_solver(info, var_mapping, solving_model)

        # Objective (solver-only)
        enabled_obj = self.get_enabled_objective()
        if enabled_obj:
            self._recreate_objective_in_model_solver(enabled_obj, var_mapping, solving_model)

        return solving_model

    # ------------------------------------------------------------------
    # Expression remapping
    # ------------------------------------------------------------------
    def _map_expr_to_new_model(self, expr: Any, var_mapping: Dict[str, Any]) -> Any:
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "Name") and expr.Name() in var_mapping:
            return var_mapping[expr.Name()]
        if hasattr(expr, "GetVar"):
            inner = expr.GetVar()
            if inner.Name() in var_mapping:
                return var_mapping[inner.Name()].Not()
        return self._deep_map_expr(expr, var_mapping)

    def _deep_map_expr(self, expr: Any, var_mapping: Dict[str, Any]) -> Any:
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "IsConstant") and expr.IsConstant():
            return int(expr)
        if hasattr(expr, "GetVars"):
            terms = [(self._map_expr_to_new_model(v, var_mapping), c) for v, c in expr.GetVars()]
            const = getattr(expr, "Offset", 0)
            new_expr = _cp.LinearExpr.WeightedSum([v for v, _ in terms], [c for _, c in terms])
            if const:
                new_expr += const
            return new_expr
        return expr

    # ------------------------------------------------------------------
    # Variable recreation
    # ------------------------------------------------------------------
    def _recreate_variable_in_model_metadata(
        self, var_info: VariableInfo, var_mapping: Dict[str, Any]
    ) -> _cp.IntVar | _cp.BoolVar | _cp.IntervalVar | _cp.Constant:
        """Recreate variable in EnhancedCpModel and store metadata."""
        return self._recreate_variable(var_info, var_mapping, target=self, store_metadata=True)

    def _recreate_variable_in_model_solver(
        self, var_info: VariableInfo, var_mapping: Dict[str, Any], model: _cp.CpModel
    ) -> _cp.IntVar | _cp.BoolVar | _cp.IntervalVar | _cp.Constant:
        """Recreate variable in a solver-only CpModel (no metadata)."""
        return self._recreate_variable(var_info, var_mapping, target=model, store_metadata=False)

    def _recreate_variable(
        self,
        var_info: VariableInfo,
        var_mapping: Dict[str, Any],
        target: _cp.CpModel | EnhancedCpModel,
        store_metadata: bool,
    ) -> _cp.IntVar | _cp.BoolVar | _cp.IntervalVar | _cp.Constant:

        args = var_info.creation_args
        if var_info.var_type == "IntVar":
            lb, ub, _ = args
            new_var = target.NewIntVar(lb, ub, var_info.name)
        elif var_info.var_type == "BoolVar":
            new_var = target.NewBoolVar(var_info.name)
        elif var_info.var_type == "Constant":
            (value,) = args
            new_var = target.NewConstant(value)
        elif var_info.var_type == "IntervalVar":
            start, size, end, _ = args
            new_var = target.NewIntervalVar(
                self._map_expr_to_new_model(start, var_mapping),
                self._map_expr_to_new_model(size, var_mapping),
                self._map_expr_to_new_model(end, var_mapping),
                var_info.name,
            )
        elif var_info.var_type == "OptionalIntervalVar":
            start, size, end, is_present, _ = args
            new_var = target.NewOptionalIntervalVar(
                self._map_expr_to_new_model(start, var_mapping),
                self._map_expr_to_new_model(size, var_mapping),
                self._map_expr_to_new_model(end, var_mapping),
                self._map_expr_to_new_model(is_present, var_mapping),
                var_info.name,
            )
        else:
            raise ValueError(f"Unsupported variable type: {var_info.var_type}")

        if store_metadata:
            target._variables[var_info.name] = VariableInfo(
                name=var_info.name,
                var_type=var_info.var_type,
                ortools_var=new_var,
                creation_args=var_info.creation_args,
            )

        return new_var

    # ------------------------------------------------------------------
    # Constraint recreation
    # ------------------------------------------------------------------
    def _recreate_constraint_in_model_metadata(
        self, constraint_info: ConstraintInfo, var_mapping: Dict[str, Any]
    ) -> None:
        """Recreate constraint in EnhancedCpModel and store metadata."""
        return self._recreate_constraint(constraint_info, var_mapping, target=self, store_metadata=True)

    def _recreate_constraint_in_model_solver(
        self, constraint_info: ConstraintInfo, var_mapping: Dict[str, Any], model: _cp.CpModel
    ) -> None:
        """Recreate constraint in a solver-only CpModel (no metadata)."""
        return self._recreate_constraint(constraint_info, var_mapping, target=model, store_metadata=False)

    def _recreate_constraint(
        self, constraint_info: ConstraintInfo, var_mapping: Dict[str, Any], target: _cp.CpModel | EnhancedCpModel, store_metadata: bool
    ) -> None:

        constraint_type = constraint_info.constraint_type
        args = constraint_info.original_args

        def map_arg(arg: Any) -> Any:
            if isinstance(arg, str) and arg in var_mapping:
                return var_mapping[arg]
            if isinstance(arg, dict) and arg.get("type") == "LinearExpr":
                vars_coeffs = arg.get("vars", [])
                const = arg.get("constant", 0)
                vs = [var_mapping[v] for v, _ in vars_coeffs]
                cs = [c for _, c in vars_coeffs]
                return _cp.LinearExpr.WeightedSum(vs, cs) + const
            if isinstance(arg, (list, tuple)):
                return [map_arg(a) for a in arg]
            if isinstance(arg, (int, bool, _cp.Domain)):
                return arg
            return self._map_expr_to_new_model(arg, var_mapping)

        # Dispatch table
        new_ct: _cp.Constraint
        if constraint_type == "Generic":
            new_ct = target.Add(map_arg(args))
        elif constraint_type == "LinearConstraint":
            expr, lb, ub = args
            new_ct = target.AddLinearConstraint(map_arg(expr), lb, ub)
        elif constraint_type == "LinearExpressionInDomain":
            expr, dom = args
            d = _cp.Domain.FromIntervals(dom) if isinstance(dom, (list, tuple)) else dom
            new_ct = target.AddLinearExpressionInDomain(map_arg(expr), d)
        elif constraint_type == "AllDifferent":
            new_ct = target.AddAllDifferent([map_arg(v) for v in args])
        elif constraint_type == "Element":
            idx, vs, tgt = args
            new_ct = target.AddElement(map_arg(idx), [map_arg(v) for v in vs], map_arg(tgt))
        elif constraint_type == "Circuit":
            new_ct = target.AddCircuit([(h, t, map_arg(l)) for h, t, l in args])
        elif constraint_type == "MultipleCircuit":
            new_ct = target.AddMultipleCircuit([(h, t, map_arg(l)) for h, t, l in args])
        elif constraint_type == "AllowedAssignments":
            vs, ts = args
            new_ct = target.AddAllowedAssignments([map_arg(v) for v in vs], ts)
        elif constraint_type == "ForbiddenAssignments":
            vs, ts = args
            new_ct = target.AddForbiddenAssignments([map_arg(v) for v in vs], ts)
        elif constraint_type == "Automaton":
            vs, ss, fs, tr = args
            new_ct = target.AddAutomaton([map_arg(v) for v in vs], ss, list(fs), list(tr))
        elif constraint_type == "Inverse":
            vs, ivs = args
            new_ct = target.AddInverse([map_arg(v) for v in vs], [map_arg(v) for v in ivs])
        elif constraint_type == "ReservoirConstraint":
            ts, lc, mn, mx = args
            new_ct = target.AddReservoirConstraint([map_arg(t) for t in ts], [map_arg(c) for c in lc], mn, mx)
        elif constraint_type == "MinEquality":
            tgt, vs = args
            new_ct = target.AddMinEquality(map_arg(tgt), [map_arg(v) for v in vs])
        elif constraint_type == "MaxEquality":
            tgt, vs = args
            new_ct = target.AddMaxEquality(map_arg(tgt), [map_arg(v) for v in vs])
        elif constraint_type == "MultiplicationEquality":
            tgt, vs = args
            new_ct = target.AddMultiplicationEquality(map_arg(tgt), [map_arg(v) for v in vs])
        elif constraint_type == "DivisionEquality":
            tgt, n, d = args
            new_ct = target.AddDivisionEquality(map_arg(tgt), map_arg(n), map_arg(d))
        elif constraint_type == "AbsEquality":
            tgt, v = args
            new_ct = target.AddAbsEquality(map_arg(tgt), map_arg(v))
        elif constraint_type == "ModuloEquality":
            tgt, v, m = args
            new_ct = target.AddModuloEquality(map_arg(tgt), map_arg(v), map_arg(m))
        elif constraint_type == "BoolOr":
            new_ct = target.AddBoolOr([map_arg(l) for l in args])
        elif constraint_type == "BoolAnd":
            new_ct = target.AddBoolAnd([map_arg(l) for l in args])
        elif constraint_type == "BoolXor":
            new_ct = target.AddBoolXor([map_arg(l) for l in args])
        elif constraint_type == "Implication":
            a, b = args
            new_ct = target.AddImplication(map_arg(a), map_arg(b))
        elif constraint_type == "NoOverlap":
            new_ct = target.AddNoOverlap([map_arg(i) for i in args])
        elif constraint_type == "NoOverlap2D":
            xs, ys = args
            new_ct = target.AddNoOverlap2D([map_arg(i) for i in xs], [map_arg(i) for i in ys])
        elif constraint_type == "Cumulative":
            ints, dems, cap = args
            new_ct = target.AddCumulative([map_arg(i) for i in ints], [map_arg(d) for d in dems], map_arg(cap))
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")
        
        # Enforcement literals
        enforcement_literals = list(constraint_info.user_enforcement_literals)
        if enforcement_literals:
            new_ct.OnlyEnforceIf(enforcement_literals)

        # Register metadata if requested
        if store_metadata:
            target._constraints[constraint_info.name] = ConstraintInfo(
                name=constraint_info.name,
                original_args=args,
                constraint_type=constraint_type,
                ortools_ct=new_ct,
                enabled=constraint_info.enabled,
                tags=constraint_info.tags.copy(),
                user_enforcement_literals=list(constraint_info.user_enforcement_literals),
            )

        return new_ct

    # ------------------------------------------------------------------
    # Objective recreation
    # ------------------------------------------------------------------
    def _recreate_objective_in_model_metadata(self, obj: ObjectiveInfo, var_mapping: Dict[str, Any]) -> None:
        """Recreate objective in EnhancedCpModel and store metadata."""
        self._recreate_objective(obj, var_mapping, target=self, store_metadata=True)

    def _recreate_objective_in_model_solver(self, obj: ObjectiveInfo, var_mapping: Dict[str, Any], model: _cp.CpModel) -> None:
        """Recreate objective in solver-only CpModel (no metadata)."""
        self._recreate_objective(obj, var_mapping, target=model, store_metadata=False)

    def _recreate_objective(
        self, obj: ObjectiveInfo, var_mapping: Dict[str, Any], target: _cp.CpModel | EnhancedCpModel, store_metadata: bool
    ) -> None:
        new_expr = self._map_expr_to_new_model(obj.linear_expr, var_mapping)

        if store_metadata:
            new_obj = ObjectiveInfo(
                objective_type=obj.objective_type,
                linear_expr=new_expr,
                name=obj.name,
            )
            new_obj.enabled = obj.enabled
            target._objectives.append(new_obj)

        if obj.enabled:
            if obj.objective_type == "Minimize":
                target.Minimize(new_expr)
            elif obj.objective_type == "Maximize":
                target.Maximize(new_expr)