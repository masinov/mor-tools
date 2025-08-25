# import_export.py
"""
Import / export mix-in.

Exposed by _IEMixin:
  - export_to_file
  - import_from_file
  - _serialize_arg
  - _deserialize_arg

Notes:
- Expression remapping helpers (_map_expr_to_new_model / _deep_map_expr)
  now live in regen.py (_RegenMixin) and are intentionally not duplicated here.
"""

from __future__ import annotations
import json
import zipfile
from typing import Any, Dict, List, Tuple

import ortools.sat.python.cp_model as _cp

from variables import VariableInfo
from constraints import ConstraintInfo
from objectives import ObjectiveInfo


class _IEMixin:
    """Model persistence helpers."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def export_to_file(self, filename: str) -> None:
        """
        Persist the *entire* EnhancedCpModel (proto + metadata) to disk.

        The file is a ZIP archive with two entries:
            model.pb   – clean OR-Tools proto (enabled constraints & active objective)
            meta.json  – all metadata (variables, constraints, objectives, tags, etc.)
        """
        # 1) Build a *clean* proto from enabled constraints/objective only.
        #    Uses _RegenMixin._create_solving_model
        temp_model = self._create_solving_model()
        proto_bytes = temp_model.Proto().SerializeToString()

        # 2) Collect metadata we need to fully reconstruct the Python-side model
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
                # explicitly persist the enable var name for deterministic round-trips
                "enable_var": info.enable_var.Name() if info.enable_var else None,
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
            "constraint_counter": getattr(self, "_constraint_counter", 0),
            "variable_counter": getattr(self, "_variable_counter", 0),
        }

        # 3) Write ZIP
        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pb", proto_bytes)
            zf.writestr("meta.json", json.dumps(meta, separators=(",", ":")))

    def import_from_file(self, filename: str) -> None:
        """
        Load a model from a saved file, restoring variables, constraints, and objectives.

        Implementation notes
        --------------------
        - We rebuild from *metadata only* to avoid duplicating constraints/variables
          that would occur if we injected the proto and then re-added everything.
        - The saved model.proto is still exported for portability/inspection, but is
          not applied during import.
        """
        # start from a clean EnhancedCpModel
        self.clear_model()

        with zipfile.ZipFile(filename, "r") as zf:
            # -- read metadata
            meta_raw = zf.read("meta.json").decode("utf-8")
            meta: Dict[str, Any] = json.loads(meta_raw)

        # -------------------
        # 1) Recreate variables
        # -------------------
        self._variables: Dict[str, VariableInfo] = {}

        # Build in two phases: basic vars first (so interval args can map properly)
        variables_meta: Dict[str, Dict[str, Any]] = meta.get("variables", {})
        basic_types = {"IntVar", "BoolVar", "Constant"}
        interval_types = {"IntervalVar", "OptionalIntervalVar"}

        # Phase A: basic vars
        for name, vmeta in variables_meta.items():
            var_type = vmeta["var_type"]
            if var_type not in basic_types:
                continue
            creation_args = tuple(self._deserialize_arg(vmeta["creation_args"], {}))  # no mapping needed
            if var_type == "IntVar":
                lb, ub, _ = creation_args  # name ignored – we use `name` explicitly
                ortools_var = self.NewIntVar(lb, ub, name)
            elif var_type == "BoolVar":
                ortools_var = self.NewBoolVar(name)
            elif var_type == "Constant":
                (value,) = creation_args
                ortools_var = self.NewConstant(value)
            else:
                continue

            # registry entry already added by Enhanced creators;
            # ensure metadata matches serialized args
            self._variables[name].creation_args = creation_args  # type: ignore[attr-defined]

        # Mapping for later expression rehydration
        var_mapping: Dict[str, Any] = {n: v.ortools_var for n, v in self._variables.items()}

        # Phase B: interval vars
        for name, vmeta in variables_meta.items():
            var_type = vmeta["var_type"]
            if var_type not in interval_types:
                continue
            # Rehydrate args now that base vars exist
            des = self._deserialize_arg(vmeta["creation_args"], var_mapping)
            creation_args = tuple(des) if isinstance(des, (list, tuple)) else (des,)

            if var_type == "IntervalVar":
                start, size, end, _nm = creation_args
                ortools_var = self.NewIntervalVar(
                    self._map_expr_to_new_model(start, var_mapping),
                    self._map_expr_to_new_model(size, var_mapping),
                    self._map_expr_to_new_model(end, var_mapping),
                    name,
                )
            else:  # OptionalIntervalVar
                start, size, end, pres, _nm = creation_args
                ortools_var = self.NewOptionalIntervalVar(
                    self._map_expr_to_new_model(start, var_mapping),
                    self._map_expr_to_new_model(size, var_mapping),
                    self._map_expr_to_new_model(end, var_mapping),
                    self._map_expr_to_new_model(pres, var_mapping),
                    name,
                )

            # registry entry already added by Enhanced creators;
            self._variables[name].creation_args = creation_args  # type: ignore[attr-defined]
            var_mapping[name] = ortools_var

        # -------------------
        # 2) Recreate objectives (metadata + set OR-Tools objective)
        # -------------------
        self._objectives: List[ObjectiveInfo] = []
        enabled_objective_expr = None
        enabled_objective_type = None

        for ometa in meta.get("objectives", []):
            # map expression using current name->var mapping
            expr = self._deserialize_arg(ometa["linear_expr"], var_mapping)
            info = ObjectiveInfo(
                objective_type=ometa["objective_type"],
                linear_expr=expr,
                name=ometa["name"],
                enabled=bool(ometa["enabled"]),
            )
            self._objectives.append(info)
            if info.enabled:
                enabled_objective_expr = expr
                enabled_objective_type = info.objective_type

        # Apply objective at OR-Tools level without altering our enabled flags
        if enabled_objective_expr is not None:
            if enabled_objective_type == "Minimize":
                super().Minimize(enabled_objective_expr)
            else:
                super().Maximize(enabled_objective_expr)

        # -------------------
        # 3) Recreate constraints (and metadata)
        # -------------------
        self._constraints: Dict[str, ConstraintInfo] = {}

        for name, cmeta in meta.get("constraints", {}).items():
            # ensure enable var exists under the same name used originally
            enable_name = cmeta.get("enable_var") or f"_enable_{name}"
            enable_var = var_mapping.get(enable_name)
            if enable_var is None:
                enable_var = self.NewBoolVar(enable_name)
                var_mapping[enable_name] = enable_var

            # rehydrate args and user literals
            rehydrated_args = self._deserialize_arg(cmeta["original_args"], var_mapping)
            user_lits = [
                self._deserialize_arg(lit, var_mapping)
                for lit in cmeta.get("user_enforcement_literals", [])
            ]

            # Build a temporary ConstraintInfo (the regen helper will register the cloned entry)
            temp_info = ConstraintInfo(
                name=name,
                original_args=rehydrated_args,
                constraint_type=cmeta["constraint_type"],
                ortools_ct=None,  # will be set by recreation
                enable_var=enable_var,
                enabled=bool(cmeta.get("enabled", True)),
                tags=set(cmeta.get("tags", [])),
                user_enforcement_literals=user_lits,
            )

            # Delegate to regen helper: creates OR-Tools ct, applies enforcement (enable_var + user lits),
            # and registers a proper ConstraintInfo in self._constraints.
            self._recreate_constraint_in_model_metadata(self, temp_info, var_mapping)

        # -------------------
        # 4) Counters
        # -------------------
        self._constraint_counter = int(meta.get("constraint_counter", 0))
        self._variable_counter = int(meta.get("variable_counter", 0))

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
            t = serialized_arg.get("type")
            if t == "LinearExpr":
                vars_coeffs = serialized_arg.get("vars", [])
                const = serialized_arg.get("constant", 0)
                vs = [var_mapping[v] for v, _ in vars_coeffs]
                cs = [c for _, c in vars_coeffs]
                return _cp.LinearExpr.WeightedSum(vs, cs) + const
            if t == "Not":
                return var_mapping[serialized_arg["var"]].Not()
        if isinstance(serialized_arg, list):
            return [self._deserialize_arg(a, var_mapping) for a in serialized_arg]
        if isinstance(serialized_arg, (int, bool)):
            return serialized_arg
        return serialized_arg
