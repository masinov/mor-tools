# model_io.py
"""
Import / export mix-in.

Exposed by _IOMixin:
  - export_to_file
  - import_from_file
  - _serialize_arg
  - _deserialize_arg
  - _map_expr_to_new_model / _deep_map_expr
"""

from __future__ import annotations
import json
import zipfile
from typing import Any, Dict, List

import ortools.sat.python.cp_model as _cp
from ortools.sat import cp_model_pb2


class _IOMixin:
    """Model persistence helpers."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def export_to_file(self, filename: str) -> None:
        """
        Persist the *entire* EnhancedCpModel (proto + metadata) to disk.

        The file is a ZIP archive with two entries:
            model.pb   – raw OR-Tools proto
            meta.json  – all metadata (variables, constraints, objectives, tags, etc.)
        """
        # 1. Re-build a clean proto to ensure consistency
        temp_model = self._create_solving_model()
        proto_bytes = temp_model.Proto().SerializeToString()

        # 2. Collect metadata
        variables_meta: Dict[str, Dict[str, Any]] = {}
        for name, info in getattr(self, "_variables", {}).items():
            variables_meta[name] = {
                "var_type": info.var_type,
                "creation_args": self._serialize_arg(info.creation_args),
            }

        constraints_meta: Dict[str, Dict[str, Any]] = {}
        for name, info in getattr(self, "_constraints", {}).items():
            serialized_lits = [
                self._serialize_arg(lit) for lit in info.user_enforcement_literals
            ]
            constraints_meta[name] = {
                "original_args": self._serialize_arg(info.original_args),
                "constraint_type": info.constraint_type,
                "enabled": info.enabled,
                "tags": list(info.tags),
                "user_enforcement_literals": serialized_lits,
                # include enable_var explicitly for round-trips
                "enable_var": info.enable_var.Name(),
            }

        objectives_meta: List[Dict[str, Any]] = []
        for obj in getattr(self, "_objectives", []):
            objectives_meta.append(
                {
                    "objective_type": obj.objective_type,
                    "name": obj.name,
                    "enabled": obj.enabled,
                    "linear_expr": self._serialize_arg(obj.linear_expr),
                }
            )

        meta = {
            "variables": variables_meta,
            "constraints": constraints_meta,
            "objectives": objectives_meta,
            "constraint_counter": self._constraint_counter,
            "variable_counter": self._variable_counter,
        }

        # 3. Write ZIP
        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pb", proto_bytes)
            zf.writestr("meta.json", json.dumps(meta, separators=(",", ":")))

    def import_from_file(self, filename: str) -> None:
        """
        Load a model from a saved file, restoring OR-Tools proto and metadata.
        Re-creates constraints to guarantee Python-side consistency.
        """
        self.clear_model()

        with zipfile.ZipFile(filename, "r") as zf:
            # 1. Proto
            proto_bytes = zf.read("model.pb")
            proto = cp_model_pb2.CpModelProto()
            proto.ParseFromString(proto_bytes)
            self.Proto().CopyFrom(proto)

            # 2. Metadata
            meta_raw = zf.read("meta.json").decode("utf-8")
            meta: Dict[str, Any] = json.loads(meta_raw)

            # 3. Variables (registry only; proto already contains them)
            self._variables: Dict[str, Any] = {}  # type: ignore
            for name, vmeta in meta["variables"].items():
                var_type = vmeta["var_type"]
                args = tuple(vmeta["creation_args"])
                ortools_var = super().get_variable_by_name(name)
                if ortools_var is None:
                    # Re-create if missing (fallback)
                    if var_type == "IntVar":
                        lb, ub, _ = args
                        ortools_var = super().NewIntVar(lb, ub, name)
                    elif var_type == "BoolVar":
                        ortools_var = super().NewBoolVar(name)
                    elif var_type == "Constant":
                        (value,) = args
                        ortools_var = super().NewConstant(value)
                    else:
                        continue  # Interval vars handled later
                self._variables[name] = type(
                    "VariableInfo",
                    (),
                    {
                        "name": name,
                        "var_type": var_type,
                        "ortools_var": ortools_var,
                        "creation_args": args,
                    },
                )()

            var_mapping = {n: v.ortools_var for n, v in self._variables.items()}

            # 4. Objectives
            self._objectives: List[Any] = []  # type: ignore
            for ometa in meta["objectives"]:
                expr = self._deserialize_arg(ometa["linear_expr"], var_mapping)
                obj = type(
                    "ObjectiveInfo",
                    (),
                    {
                        "objective_type": ometa["objective_type"],
                        "name": ometa["name"],
                        "enabled": ometa["enabled"],
                        "linear_expr": expr,
                    },
                )()
                self._objectives.append(obj)
                if obj.enabled:
                    if obj.objective_type == "Minimize":
                        super().Minimize(expr)
                    elif obj.objective_type == "Maximize":
                        super().Maximize(expr)

            # 5. Constraints
            self._constraints: Dict[str, Any] = {}  # type: ignore
            recreate = getattr(self, "_recreate_constraint_in_model")

            for name, cmeta in meta["constraints"].items():
                enable_name = cmeta.get("enable_var", f"_enable_{name}")
                enable_var = super().get_variable_by_name(enable_name)
                if enable_var is None:
                    enable_var = super().NewBoolVar(enable_name)

                rehydrated_args = self._deserialize_arg(
                    cmeta["original_args"], var_mapping
                )
                info = type(
                    "ConstraintInfo",
                    (),
                    {
                        "name": name,
                        "original_args": rehydrated_args,
                        "constraint_type": cmeta["constraint_type"],
                        "ortools_ct": None,
                        "enable_var": enable_var,
                        "enabled": cmeta["enabled"],
                        "tags": set(cmeta.get("tags", [])),
                        "user_enforcement_literals": [
                            self._deserialize_arg(lit, var_mapping)
                            for lit in cmeta.get("user_enforcement_literals", [])
                        ],
                    },
                )()
                self._constraints[name] = info
                recreate(self, info, var_mapping)  # type: ignore

            # 6. Counters
            self._constraint_counter = meta["constraint_counter"]
            self._variable_counter = meta["variable_counter"]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _serialize_arg(self, arg: Any) -> Any:
        """Convert OR-Tools objects to JSON-serialisable primitives."""
        if isinstance(arg, (_cp.IntVar, _cp.IntervalVar)):
            return arg.Name()
        if hasattr(arg, "GetVar"):  # NotBooleanVariable
            return {"type": "Not", "var": arg.GetVar().Name()}
        if isinstance(arg, _cp.Domain):
            return list(arg.FlattenedIntervals())
        if isinstance(arg, _cp.LinearExpr):
            if arg.IsConstant():
                return int(arg)
            serialized = {"type": "LinearExpr"}
            if hasattr(arg, "GetVars"):
                serialized["vars"] = [(v.Name(), c) for v, c in arg.GetVars()]
                serialized["constant"] = getattr(arg, "Offset", 0)
            else:
                serialized["complex"] = True
                serialized["str"] = str(arg)
            return serialized
        if isinstance(arg, (list, tuple)):
            return [self._serialize_arg(a) for a in arg]
        if isinstance(arg, (int, str, bool)):
            return arg
        return str(arg)  # fallback

    def _deserialize_arg(self, serialized_arg: Any, var_mapping: Dict[str, Any]) -> Any:
        """Re-hydrate primitives back to OR-Tools objects."""
        if isinstance(serialized_arg, str) and serialized_arg in var_mapping:
            return var_mapping[serialized_arg]
        if isinstance(serialized_arg, dict):
            if serialized_arg.get("type") == "LinearExpr":
                vars_coeffs = serialized_arg.get("vars", [])
                const = serialized_arg.get("constant", 0)
                vs = [var_mapping[v] for v, _ in vars_coeffs]
                cs = [c for _, c in vars_coeffs]
                return _cp.LinearExpr.WeightedSum(vs, cs) + const
            if serialized_arg.get("type") == "Not":
                return var_mapping[serialized_arg["var"]].Not()
        if isinstance(serialized_arg, list):
            return [self._deserialize_arg(a, var_mapping) for a in serialized_arg]
        if isinstance(serialized_arg, (int, bool)):
            return serialized_arg
        return serialized_arg

    # ------------------------------------------------------------------
    # Expression-mapping helpers (moved here from variables.py)
    # ------------------------------------------------------------------
    def _map_expr_to_new_model(self, expr: Any, var_mapping: Dict[str, Any]) -> Any:
        """Return expr with all variables remapped via var_mapping."""
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "Name") and expr.Name() and expr.Name() in var_mapping:
            return var_mapping[expr.Name()]
        # Handle NotBooleanVariable
        if hasattr(expr, "GetVar"):
            inner = expr.GetVar()
            if inner.Name() in var_mapping:
                return var_mapping[inner.Name()].Not()
        return self._deep_map_expr(expr, var_mapping)

    def _deep_map_expr(self, expr: Any, var_mapping: Dict[str, Any]) -> Any:
        """Recursively rebuild linear expressions with remapped variables."""
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "IsConstant") and expr.IsConstant():
            return int(expr)
        if hasattr(expr, "GetVars"):
            terms = [
                (self._map_expr_to_new_model(var, var_mapping), coeff)
                for var, coeff in expr.GetVars()
            ]
            const = getattr(expr, "Offset", 0)
            new_expr = _cp.LinearExpr.WeightedSum(
                [v for v, _ in terms], [c for _, c in terms]
            )
            if const:
                new_expr += const
            return new_expr
        # Fallback
        return expr

    # ------------------------------------------------------------------
    # Internal helper for other mix-ins
    # ------------------------------------------------------------------
    def _create_solving_model(self) -> _cp.CpModel:
        """
        Build a *clean* CpModel with only enabled constraints & objectives.
        Used by export_to_file and debug_infeasible.
        """
        solving_model = _cp.CpModel()
        var_mapping: Dict[str, Any] = {}

        # 1. Re-create non-enable variables
        for name, info in getattr(self, "_variables", {}).items():
            if name.startswith("_enable_"):
                continue
            if info.var_type == "IntVar":
                lb, ub, _ = info.creation_args
                new_v = solving_model.NewIntVar(lb, ub, name)
            elif info.var_type == "BoolVar":
                new_v = solving_model.NewBoolVar(name)
            elif info.var_type == "Constant":
                (val,) = info.creation_args
                new_v = solving_model.NewConstant(val)
            elif info.var_type == "IntervalVar":
                start, size, end, _ = info.creation_args
                new_start = self._map_expr_to_new_model(start, var_mapping)
                new_size = self._map_expr_to_new_model(size, var_mapping)
                new_end = self._map_expr_to_new_model(end, var_mapping)
                new_v = solving_model.NewIntervalVar(new_start, new_size, new_end, name)
            elif info.var_type == "OptionalIntervalVar":
                start, size, end, pres, _ = info.creation_args
                new_start = self._map_expr_to_new_model(start, var_mapping)
                new_size = self._map_expr_to_new_model(size, var_mapping)
                new_end = self._map_expr_to_new_model(end, var_mapping)
                new_pres = self._map_expr_to_new_model(pres, var_mapping)
                new_v = solving_model.NewOptionalIntervalVar(
                    new_start, new_size, new_end, new_pres, name
                )
            else:
                continue  # unknown
            var_mapping[name] = new_v

        # 2. Re-create constraints
        recreate = getattr(self, "_recreate_constraint_in_model")
        for info in getattr(self, "_constraints", {}).values():
            if info.enabled:
                recreate(solving_model, info, var_mapping)

        # 3. Objective
        enabled_obj = getattr(self, "get_enabled_objective", lambda: None)()
        if enabled_obj:
            expr = self._map_expr_to_new_model(enabled_obj.linear_expr, var_mapping)
            if enabled_obj.objective_type == "Minimize":
                solving_model.Minimize(expr)
            else:
                solving_model.Maximize(expr)

        return solving_model
