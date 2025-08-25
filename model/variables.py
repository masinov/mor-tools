# model_variables.py
"""
Variable-creation mix-in and helpers.

Exposed by _VariablesMixin:
  - NewIntVar, NewBoolVar, NewIntervalVar, NewOptionalIntervalVar, NewConstant
  - VariableInfo dataclass (registry entry)
  - Introspection methods: get_variable_info, get_variable_names, get_variables_by_type, get_variable_by_name
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any, List, Optional
import ortools.sat.python.cp_model as _cp

@dataclass
class VariableInfo:
    """Rich wrapper around OR-Tools variables with metadata."""
    name: str
    var_type: str
    ortools_var: Union[_cp.IntVar, _cp.IntervalVar, _cp.Literal]
    creation_args: Tuple[Any, ...]

class _VariablesMixin:
    """All variable-creation helpers."""

    # ------------------------------------------------------------------
    # Public OR-Tools variable creators
    # ------------------------------------------------------------------
    def NewIntVar(self, lb: int, ub: int, name: str) -> _cp.IntVar:
        """Create a new integer variable with bounds [lb, ub]."""
        variables = self._ensure_variables()
        if name in variables:
            raise ValueError(f"Variable name '{name}' already exists")

        var = super().NewIntVar(lb, ub, name)
        variables[name] = VariableInfo(
            name=name,
            var_type="IntVar",
            ortools_var=var,
            creation_args=(lb, ub, name),
        )

        self._variable_counter = getattr(self, "_variable_counter", 0) + 1
        return var

    def NewBoolVar(self, name: str) -> _cp.IntVar:
        """Create a new boolean variable."""
        variables = self._ensure_variables()
        if name in variables:
            raise ValueError(f"Variable name '{name}' already exists")

        var = super().NewBoolVar(name)
        variables[name] = VariableInfo(
            name=name,
            var_type="BoolVar",
            ortools_var=var,
            creation_args=(name,),
        )

        self._variable_counter = getattr(self, "_variable_counter", 0) + 1
        return var

    def NewIntervalVar(
        self,
        start: _cp.LinearExprT,
        size: _cp.LinearExprT,
        end: _cp.LinearExprT,
        name: str,
    ) -> _cp.IntervalVar:
        """Create a new interval variable."""
        variables = self._ensure_variables()
        if name in variables:
            raise ValueError(f"Variable name '{name}' already exists")

        var = super().NewIntervalVar(start, size, end, name)
        variables[name] = VariableInfo(
            name=name,
            var_type="IntervalVar",
            ortools_var=var,
            creation_args=(start, size, end, name),
        )

        self._variable_counter = getattr(self, "_variable_counter", 0) + 1
        return var

    def NewOptionalIntervalVar(
        self,
        start: _cp.LinearExprT,
        size: _cp.LinearExprT,
        end: _cp.LinearExprT,
        is_present: _cp.LiteralT,
        name: str,
    ) -> _cp.IntervalVar:
        """Create a new optional interval variable."""
        variables = self._ensure_variables()
        if name in variables:
            raise ValueError(f"Variable name '{name}' already exists")

        var = super().NewOptionalIntervalVar(start, size, end, is_present, name)
        variables[name] = VariableInfo(
            name=name,
            var_type="OptionalIntervalVar",
            ortools_var=var,
            creation_args=(start, size, end, is_present, name),
        )

        self._variable_counter = getattr(self, "_variable_counter", 0) + 1
        return var

    def NewConstant(self, value: int, name: Optional[str] = None) -> _cp.IntVar:
        """Create a new constant. Allows a custom name or auto-generated unique name."""
        variables = self._ensure_variables()

        if name is not None:
            # User-provided name must be unique
            if name in variables:
                raise ValueError(f"Variable name '{name}' already exists")
            unique_name = name
        else:
            # Auto-generate a unique name
            base_name = f"_const_{value}"
            unique_name = base_name
            counter = getattr(self, "_variable_counter", 0)
            while unique_name in variables:
                counter += 1
                unique_name = f"{base_name}_{counter}"
            self._variable_counter = counter + 1  # Update for next call

        var = super().NewConstant(value)
        variables[unique_name] = VariableInfo(
            name=unique_name,
            var_type="Constant",
            ortools_var=var,
            creation_args=(value, unique_name),
        )
        return var

    # ------------------------------------------------------------------
    # Variable introspection / debug methods
    # ------------------------------------------------------------------
    def get_variable_info(self, name: str) -> VariableInfo:
        """Get detailed information about a variable."""
        variables = self._ensure_variables()
        if name not in variables:
            raise ValueError(f"Variable '{name}' not found")
        return variables[name]

    def get_variable_names(self) -> List[str]:
        """Get all variable names."""
        variables = self._ensure_variables()
        return list(variables.keys())

    def get_variables_by_type(self, var_type: str) -> List[str]:
        """Get all variables of a specific type."""
        variables = self._ensure_variables()
        return [name for name, info in variables.items() if info.var_type == var_type]

    def get_variable_by_name(self, name: str) -> Optional[Union[_cp.IntVar, _cp.IntervalVar]]:
        """Get a variable by its name."""
        variables = self._ensure_variables()
        if name in variables:
            return variables[name].ortools_var
        return None

    # ------------------------------------------------------------------
    # Internal bookkeeping helper
    # ------------------------------------------------------------------
    def _ensure_variables(self) -> Dict[str, VariableInfo]:
        """Lazy-initialise the variable registry for mix-in safety."""
        if not hasattr(self, "_variables"):
            self._variables: Dict[str, VariableInfo] = {}
        return self._variables