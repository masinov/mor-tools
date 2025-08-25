# model_constraints.py
"""
Constraint-creation mix-in.

Exposed by _ConstraintsMixin:
  - All public Add*() methods
  - ConstraintInfo dataclass
  - _register_constraint()
  - _ConstraintProxy
"""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple, Set
import ortools.sat.python.cp_model as _cp

from dataclasses import dataclass, field

@dataclass
class ConstraintInfo:
    """Wrapper around a single constraint with metadata for debugging/enable/disable."""
    name: str
    original_args: Any
    constraint_type: str
    ortools_ct: _cp.Constraint
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    user_enforcement_literals: List[_cp.LiteralT] = field(default_factory=list)

class _ConstraintProxy:
    """Proxy returned by Add*/etc. to capture user calls like OnlyEnforceIf."""

    def __init__(self, ct: _cp.Constraint, info: ConstraintInfo) -> None:
        self._ct = ct
        self._info = info

    def OnlyEnforceIf(self, lits) -> "_ConstraintProxy":
        if not isinstance(lits, (list, tuple)):
            lits = [lits]
        self._info.user_enforcement_literals.extend(lits)
        self._ct.OnlyEnforceIf(lits)
        return self

    def WithName(self, name: str) -> "_ConstraintProxy":
        self._ct.WithName(name)
        self._info.name = name
        return self

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ct, attr)

class _ConstraintsMixin:
    """All constraint-creation helpers."""

    # ------------------------------------------------------------------
    # Public constraint creators
    # ------------------------------------------------------------------

    def Add(self, ct, name: Optional[str] = None) -> _cp.Constraint:
        """Add a generic constraint."""
        return self._register_constraint(
            constraint=super().Add(ct),
            original_args=ct,
            constraint_type="Generic",
            name=name,
        )

    def AddLinearConstraint(
        self,
        linear_expr: _cp.LinearExprT,
        lb: int,
        ub: int,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddLinearConstraint(linear_expr, lb, ub),
            original_args=(linear_expr, lb, ub),
            constraint_type="LinearConstraint",
            name=name,
        )

    def AddLinearExpressionInDomain(
        self,
        linear_expr: _cp.LinearExprT,
        domain: _cp.Domain,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddLinearExpressionInDomain(linear_expr, domain),
            original_args=(linear_expr, domain),
            constraint_type="LinearExpressionInDomain",
            name=name,
        )

    def AddAllDifferent(
        self,
        variables: Sequence[_cp.LinearExprT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddAllDifferent(variables),
            original_args=tuple(variables),
            constraint_type="AllDifferent",
            name=name,
        )

    def AddElement(
        self,
        index: _cp.LinearExprT,
        variables: Sequence[_cp.LinearExprT],
        target: _cp.LinearExprT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddElement(index, variables, target),
            original_args=(index, tuple(variables), target),
            constraint_type="Element",
            name=name,
        )

    def AddCircuit(
        self,
        arcs: Sequence[Tuple[int, int, _cp.LiteralT]],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddCircuit(arcs),
            original_args=tuple(arcs),
            constraint_type="Circuit",
            name=name,
        )

    def AddMultipleCircuit(
        self,
        arcs: Sequence[Tuple[int, int, _cp.LiteralT]],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddMultipleCircuit(arcs),
            original_args=tuple(arcs),
            constraint_type="MultipleCircuit",
            name=name,
        )

    def AddAllowedAssignments(
        self,
        variables: Sequence[_cp.IntVar],
        tuples_list: Sequence[Sequence[int]],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddAllowedAssignments(variables, tuples_list),
            original_args=(tuple(variables), tuple(tuple(t) for t in tuples_list)),
            constraint_type="AllowedAssignments",
            name=name,
        )

    def AddForbiddenAssignments(
        self,
        variables: Sequence[_cp.IntVar],
        tuples_list: Sequence[Sequence[int]],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddForbiddenAssignments(variables, tuples_list),
            original_args=(tuple(variables), tuple(tuple(t) for t in tuples_list)),
            constraint_type="ForbiddenAssignments",
            name=name,
        )

    def AddAutomaton(
        self,
        transition_variables: Sequence[_cp.IntVar],
        starting_state: int,
        final_states: Sequence[int],
        transition_triples: Sequence[Tuple[int, int, int]],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddAutomaton(
                transition_variables, starting_state, final_states, transition_triples
            ),
            original_args=(
                tuple(transition_variables),
                starting_state,
                tuple(final_states),
                tuple(transition_triples),
            ),
            constraint_type="Automaton",
            name=name,
        )

    def AddInverse(
        self,
        variables: Sequence[_cp.IntVar],
        inverse_variables: Sequence[_cp.IntVar],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddInverse(variables, inverse_variables),
            original_args=(tuple(variables), tuple(inverse_variables)),
            constraint_type="Inverse",
            name=name,
        )

    def AddReservoirConstraint(
        self,
        times: Sequence[_cp.LinearExprT],
        level_changes: Sequence[_cp.LinearExprT],
        min_level: int,
        max_level: int,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddReservoirConstraint(
                times, level_changes, min_level, max_level
            ),
            original_args=(tuple(times), tuple(level_changes), min_level, max_level),
            constraint_type="ReservoirConstraint",
            name=name,
        )

    # Arithmetic / aggregation helpers
    def AddMinEquality(
        self,
        target: _cp.LinearExprT,
        variables: Sequence[_cp.LinearExprT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddMinEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MinEquality",
            name=name,
        )

    def AddMaxEquality(
        self,
        target: _cp.LinearExprT,
        variables: Sequence[_cp.LinearExprT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddMaxEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MaxEquality",
            name=name,
        )

    def AddMultiplicationEquality(
        self,
        target: _cp.LinearExprT,
        variables: Sequence[_cp.LinearExprT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddMultiplicationEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MultiplicationEquality",
            name=name,
        )

    def AddDivisionEquality(
        self,
        target: _cp.LinearExprT,
        numerator: _cp.LinearExprT,
        denominator: _cp.LinearExprT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddDivisionEquality(target, numerator, denominator),
            original_args=(target, numerator, denominator),
            constraint_type="DivisionEquality",
            name=name,
        )

    def AddAbsEquality(
        self,
        target: _cp.LinearExprT,
        variable: _cp.LinearExprT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddAbsEquality(target, variable),
            original_args=(target, variable),
            constraint_type="AbsEquality",
            name=name,
        )

    def AddModuloEquality(
        self,
        target: _cp.LinearExprT,
        variable: _cp.LinearExprT,
        modulo: _cp.LinearExprT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddModuloEquality(target, variable, modulo),
            original_args=(target, variable, modulo),
            constraint_type="ModuloEquality",
            name=name,
        )

    # Boolean / logical constraints
    def AddBoolOr(
        self,
        literals: Sequence[_cp.LiteralT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddBoolOr(literals),
            original_args=tuple(literals),
            constraint_type="BoolOr",
            name=name,
        )

    def AddBoolAnd(
        self,
        literals: Sequence[_cp.LiteralT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddBoolAnd(literals),
            original_args=tuple(literals),
            constraint_type="BoolAnd",
            name=name,
        )

    def AddBoolXor(
        self,
        literals: Sequence[_cp.LiteralT],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddBoolXor(literals),
            original_args=tuple(literals),
            constraint_type="BoolXor",
            name=name,
        )

    def AddImplication(
        self,
        a: _cp.LiteralT,
        b: _cp.LiteralT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddImplication(a, b),
            original_args=(a, b),
            constraint_type="Implication",
            name=name,
        )

    # Scheduling helpers
    def AddNoOverlap(
        self,
        intervals: Sequence[_cp.IntervalVar],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddNoOverlap(intervals),
            original_args=tuple(intervals),
            constraint_type="NoOverlap",
            name=name,
        )

    def AddNoOverlap2D(
        self,
        x_intervals: Sequence[_cp.IntervalVar],
        y_intervals: Sequence[_cp.IntervalVar],
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddNoOverlap2D(x_intervals, y_intervals),
            original_args=(tuple(x_intervals), tuple(y_intervals)),
            constraint_type="NoOverlap2D",
            name=name,
        )

    def AddCumulative(
        self,
        intervals: Sequence[_cp.IntervalVar],
        demands: Sequence[_cp.LinearExprT],
        capacity: _cp.LinearExprT,
        name: Optional[str] = None,
    ) -> _cp.Constraint:
        return self._register_constraint(
            constraint=super().AddCumulative(intervals, demands, capacity),
            original_args=(tuple(intervals), tuple(demands), capacity),
            constraint_type="Cumulative",
            name=name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_constraint(
        self,
        constraint: _cp.Constraint,
        original_args: Any,
        constraint_type: str,
        name: Optional[str],
    ) -> _cp.Constraint:
        """Register a constraint with full metadata."""
        constraints = self._ensure_constraints()
        if not hasattr(self, "_constraint_counter"):
            self._constraint_counter = 0

        if name is None:
            name = f"{constraint_type.lower()}_{self._constraint_counter}"
        elif name in constraints:
            raise ValueError(f"Constraint name '{name}' already exists")

        self._constraint_counter += 1

        info = ConstraintInfo(
            name=name,
            original_args=original_args,
            constraint_type=constraint_type,
            ortools_ct=constraint,
        )
        constraints[name] = info
        return _ConstraintProxy(constraint, info)

    def _ensure_constraints(self) -> dict[str, ConstraintInfo]:
        """Lazy-initialise the constraint registry for mix-in safety."""
        if not hasattr(self, "_constraints"):
            self._constraints: dict[str, ConstraintInfo] = {}
        return self._constraints

    # ------------------------------------------------------------------
    # Vanilla introspection & enable/disable/tagging methods
    # ------------------------------------------------------------------
    def get_constraint_info(self, name: str) -> ConstraintInfo:
        """Get detailed information about a constraint."""
        constraints = self._ensure_constraints()
        if name not in constraints:
            raise ValueError(f"Constraint '{name}' not found")
        return constraints[name]

    def get_constraint_names(self) -> List[str]:
        """Get all constraint names."""
        return list(self._ensure_constraints().keys())

    def get_constraints_by_type(self, constraint_type: str) -> List[str]:
        """Get all constraints of a specific type."""
        return [name for name, info in self._ensure_constraints().items()
                if info.constraint_type == constraint_type]

    def get_constraints_by_tag(self, tag: str) -> List[str]:
        """Get all constraints with a specific tag."""
        return [name for name, info in self._ensure_constraints().items()
                if tag in info.tags]

    def get_enabled_constraints(self) -> List[str]:
        """Get all currently enabled constraints."""
        return [name for name, info in self._ensure_constraints().items() if info.enabled]

    def get_disabled_constraints(self) -> List[str]:
        """Get all currently disabled constraints."""
        return [name for name, info in self._ensure_constraints().items() if not info.enabled]

    def enable_constraint(self, name: str) -> None:
        """Enable a specific constraint by name."""
        info = self.get_constraint_info(name)
        info.enabled = True

    def disable_constraint(self, name: str) -> None:
        """Disable a specific constraint by name."""
        info = self.get_constraint_info(name)
        info.enabled = False

    def enable_constraints(self, names: Sequence[str]) -> None:
        """Enable multiple constraints by name."""
        for name in names:
            self.enable_constraint(name)

    def disable_constraints(self, names: Sequence[str]) -> None:
        """Disable multiple constraints by name."""
        for name in names:
            self.disable_constraint(name)

    def add_constraint_tag(self, name: str, tag: str) -> None:
        """Add a tag to a constraint for group operations."""
        info = self.get_constraint_info(name)
        info.tags.add(tag)

    def add_constraint_tags(self, name: str, tags: List[str]) -> None:
        """Add multiple tags to a constraint for group operations."""
        info = self.get_constraint_info(name)
        info.tags.update(tags)

    def enable_constraints_by_tag(self, tag: str) -> None:
        """Enable all constraints with a specific tag."""
        for info in self._ensure_constraints().values():
            if tag in info.tags:
                info.enabled = True

    def disable_constraints_by_tag(self, tag: str) -> None:
        """Disable all constraints with a specific tag."""
        for info in self._ensure_constraints().values():
            if tag in info.tags:
                info.enabled = False