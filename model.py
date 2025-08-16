from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Sequence, Tuple
from ortools.sat.python import cp_model as _cp


###############################################################################
# Helper classes for debugging
###############################################################################
class ConstraintInfo:
    """Rich wrapper around a single constraint with debugging metadata."""
    __slots__ = ("name", "original_args", "constraint_type", "ortools_ct", "enable_var", "enabled", "tags")

    def __init__(
        self,
        name: str,
        original_args: Any,
        constraint_type: str,
        ortools_ct: _cp.Constraint,
        enable_var: _cp.IntVar,
    ):
        self.name = name
        self.original_args = original_args
        self.constraint_type = constraint_type
        self.ortools_ct = ortools_ct
        self.enable_var = enable_var
        self.enabled = True
        self.tags: set[str] = set()


class VariableInfo:
    """Rich wrapper around variables with metadata."""
    __slots__ = ("name", "var_type", "ortools_var", "creation_args")

    def __init__(self, name: str, var_type: str, ortools_var: Union[_cp.IntVar, _cp.IntervalVar], creation_args: tuple):
        self.name = name
        self.var_type = var_type
        self.ortools_var = ortools_var
        self.creation_args = creation_args


###############################################################################
class EnhancedCpModel(_cp.CpModel):
    """
    Drop-in replacement for CpModel that provides:
    - Complete constraint and variable registration
    - Assumption-based constraint enable/disable
    - Rich debugging and introspection capabilities
    - Minimal Infeasible Subset (MIS) finding
    """

    def __init__(self) -> None:
        super().__init__()
        
        # Core registries
        self._constraints: Dict[str, ConstraintInfo] = {}
        self._variables: Dict[str, VariableInfo] = {}
        self._constraint_counter = 0
        self._variable_counter = 0

    ###########################################################################
    # Variable Creation - All CP-SAT Variable Types
    ###########################################################################
    
    def NewIntVar(self, lb: int, ub: int, name: str) -> _cp.IntVar:
        """Create a new integer variable with bounds [lb, ub]."""
        if name in self._variables:
            raise ValueError(f"Variable name '{name}' already exists")
        
        var = super().NewIntVar(lb, ub, name)
        self._variables[name] = VariableInfo(
            name=name,
            var_type="IntVar",
            ortools_var=var,
            creation_args=(lb, ub, name)
        )
        return var

    def NewBoolVar(self, name: str) -> _cp.IntVar:
        """Create a new boolean variable."""
        if name in self._variables:
            raise ValueError(f"Variable name '{name}' already exists")
        
        var = super().NewBoolVar(name)
        self._variables[name] = VariableInfo(
            name=name,
            var_type="BoolVar",
            ortools_var=var,
            creation_args=(name,)
        )
        return var

    def NewIntervalVar(
        self,
        start: _cp.LinearExprT,
        size: _cp.LinearExprT,
        end: _cp.LinearExprT,
        name: str,
    ) -> _cp.IntervalVar:
        """Create a new interval variable."""
        if name in self._variables:
            raise ValueError(f"Variable name '{name}' already exists")
        
        var = super().NewIntervalVar(start, size, end, name)
        self._variables[name] = VariableInfo(
            name=name,
            var_type="IntervalVar",
            ortools_var=var,
            creation_args=(start, size, end, name)
        )
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
        if name in self._variables:
            raise ValueError(f"Variable name '{name}' already exists")
        
        var = super().NewOptionalIntervalVar(start, size, end, is_present, name)
        self._variables[name] = VariableInfo(
            name=name,
            var_type="OptionalIntervalVar",
            ortools_var=var,
            creation_args=(start, size, end, is_present, name)
        )
        return var

    def NewConstant(self, value: int) -> _cp.IntVar:
        """Create a new constant."""
        var = super().NewConstant(value)
        # Constants don't need names, so we generate one
        name = f"_constant_{self._variable_counter}_{value}"
        self._variable_counter += 1
        
        self._variables[name] = VariableInfo(
            name=name,
            var_type="Constant",
            ortools_var=var,
            creation_args=(value,)
        )
        return var

    ###########################################################################
    # Constraint Creation - All CP-SAT Constraint Types
    ###########################################################################

    def Add(self, ct, name: Optional[str] = None) -> _cp.Constraint:
        """Add a generic constraint."""
        return self._register_constraint(
            constraint=super().Add(ct),
            original_args=ct,
            constraint_type="Generic",
            name=name
        )

    def AddLinearConstraint(self, linear_expr, lb: int, ub: int, name: Optional[str] = None) -> _cp.Constraint:
        """Add a linear constraint lb <= linear_expr <= ub."""
        return self._register_constraint(
            constraint=super().AddLinearConstraint(linear_expr, lb, ub),
            original_args=(linear_expr, lb, ub),
            constraint_type="LinearConstraint",
            name=name
        )

    def AddLinearExpressionInDomain(self, linear_expr, domain: _cp.Domain, name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that linear_expr is in domain."""
        return self._register_constraint(
            constraint=super().AddLinearExpressionInDomain(linear_expr, domain),
            original_args=(linear_expr, domain),
            constraint_type="LinearExpressionInDomain",
            name=name
        )

    def AddAllDifferent(self, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None) -> _cp.Constraint:
        """Add an all different constraint."""
        return self._register_constraint(
            constraint=super().AddAllDifferent(variables),
            original_args=tuple(variables),
            constraint_type="AllDifferent",
            name=name
        )

    def AddElement(self, index: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], target: _cp.LinearExprT, name: Optional[str] = None) -> _cp.Constraint:
        """Add an element constraint: variables[index] == target."""
        return self._register_constraint(
            constraint=super().AddElement(index, variables, target),
            original_args=(index, tuple(variables), target),
            constraint_type="Element",
            name=name
        )

    def AddCircuit(self, arcs: Sequence[Tuple[int, int, _cp.LiteralT]], name: Optional[str] = None) -> _cp.Constraint:
        """Add a circuit constraint."""
        return self._register_constraint(
            constraint=super().AddCircuit(arcs),
            original_args=tuple(arcs),
            constraint_type="Circuit",
            name=name
        )

    def AddMultipleCircuit(self, arcs: Sequence[Tuple[int, int, _cp.LiteralT]], name: Optional[str] = None) -> _cp.Constraint:
        """Add a multiple circuit constraint."""
        return self._register_constraint(
            constraint=super().AddMultipleCircuit(arcs),
            original_args=tuple(arcs),
            constraint_type="MultipleCircuit",
            name=name
        )

    def AddAllowedAssignments(self, variables: Sequence[_cp.IntVar], tuples_list: Sequence[Sequence[int]], name: Optional[str] = None) -> _cp.Constraint:
        """Add an allowed assignments constraint."""
        return self._register_constraint(
            constraint=super().AddAllowedAssignments(variables, tuples_list),
            original_args=(tuple(variables), tuple(tuple(t) for t in tuples_list)),
            constraint_type="AllowedAssignments",
            name=name
        )

    def AddForbiddenAssignments(self, variables: Sequence[_cp.IntVar], tuples_list: Sequence[Sequence[int]], name: Optional[str] = None) -> _cp.Constraint:
        """Add a forbidden assignments constraint."""
        return self._register_constraint(
            constraint=super().AddForbiddenAssignments(variables, tuples_list),
            original_args=(tuple(variables), tuple(tuple(t) for t in tuples_list)),
            constraint_type="ForbiddenAssignments",
            name=name
        )

    def AddAutomaton(self, transition_variables: Sequence[_cp.IntVar], starting_state: int, final_states: Sequence[int], transition_triples: Sequence[Tuple[int, int, int]], name: Optional[str] = None) -> _cp.Constraint:
        """Add an automaton constraint."""
        return self._register_constraint(
            constraint=super().AddAutomaton(transition_variables, starting_state, final_states, transition_triples),
            original_args=(tuple(transition_variables), starting_state, tuple(final_states), tuple(transition_triples)),
            constraint_type="Automaton",
            name=name
        )

    def AddInverse(self, variables: Sequence[_cp.IntVar], inverse_variables: Sequence[_cp.IntVar], name: Optional[str] = None) -> _cp.Constraint:
        """Add an inverse constraint."""
        return self._register_constraint(
            constraint=super().AddInverse(variables, inverse_variables),
            original_args=(tuple(variables), tuple(inverse_variables)),
            constraint_type="Inverse",
            name=name
        )

    def AddReservoirConstraint(self, times: Sequence[_cp.LinearExprT], level_changes: Sequence[_cp.LinearExprT], min_level: int, max_level: int, name: Optional[str] = None) -> _cp.Constraint:
        """Add a reservoir constraint."""
        return self._register_constraint(
            constraint=super().AddReservoirConstraint(times, level_changes, min_level, max_level),
            original_args=(tuple(times), tuple(level_changes), min_level, max_level),
            constraint_type="ReservoirConstraint",
            name=name
        )

    # Boolean constraints
    def AddBoolOr(self, literals: Sequence[_cp.LiteralT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a boolean OR constraint."""
        return self._register_constraint(
            constraint=super().AddBoolOr(literals),
            original_args=tuple(literals),
            constraint_type="BoolOr",
            name=name
        )

    def AddBoolAnd(self, literals: Sequence[_cp.LiteralT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a boolean AND constraint."""
        return self._register_constraint(
            constraint=super().AddBoolAnd(literals),
            original_args=tuple(literals),
            constraint_type="BoolAnd",
            name=name
        )

    def AddBoolXor(self, literals: Sequence[_cp.LiteralT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a boolean XOR constraint."""
        return self._register_constraint(
            constraint=super().AddBoolXor(literals),
            original_args=tuple(literals),
            constraint_type="BoolXor",
            name=name
        )

    def AddImplication(self, a: _cp.LiteralT, b: _cp.LiteralT, name: Optional[str] = None) -> _cp.Constraint:
        """Add an implication constraint a => b."""
        return self._register_constraint(
            constraint=super().AddImplication(a, b),
            original_args=(a, b),
            constraint_type="Implication",
            name=name
        )

    # Scheduling constraints
    def AddNoOverlap(self, intervals: Sequence[_cp.IntervalVar], name: Optional[str] = None) -> _cp.Constraint:
        """Add a no overlap constraint."""
        return self._register_constraint(
            constraint=super().AddNoOverlap(intervals),
            original_args=tuple(intervals),
            constraint_type="NoOverlap",
            name=name
        )

    def AddNoOverlap2D(self, x_intervals: Sequence[_cp.IntervalVar], y_intervals: Sequence[_cp.IntervalVar], name: Optional[str] = None) -> _cp.Constraint:
        """Add a 2D no overlap constraint."""
        return self._register_constraint(
            constraint=super().AddNoOverlap2D(x_intervals, y_intervals),
            original_args=(tuple(x_intervals), tuple(y_intervals)),
            constraint_type="NoOverlap2D",
            name=name
        )

    def AddCumulative(self, intervals: Sequence[_cp.IntervalVar], demands: Sequence[_cp.LinearExprT], capacity: _cp.LinearExprT, name: Optional[str] = None) -> _cp.Constraint:
        """Add a cumulative constraint."""
        return self._register_constraint(
            constraint=super().AddCumulative(intervals, demands, capacity),
            original_args=(tuple(intervals), tuple(demands), capacity),
            constraint_type="Cumulative",
            name=name
        )

    ###########################################################################
    # Constraint Management
    ###########################################################################

    def enable_constraint(self, name: str) -> None:
        """Enable a specific constraint by name."""
        if name not in self._constraints:
            raise ValueError(f"Constraint '{name}' not found")
        self._constraints[name].enabled = True

    def disable_constraint(self, name: str) -> None:
        """Disable a specific constraint by name."""
        if name not in self._constraints:
            raise ValueError(f"Constraint '{name}' not found")
        self._constraints[name].enabled = False

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
        if name not in self._constraints:
            raise ValueError(f"Constraint '{name}' not found")
        self._constraints[name].tags.add(tag)

    def enable_constraints_by_tag(self, tag: str) -> None:
        """Enable all constraints with a specific tag."""
        for info in self._constraints.values():
            if tag in info.tags:
                info.enabled = True

    def disable_constraints_by_tag(self, tag: str) -> None:
        """Disable all constraints with a specific tag."""
        for info in self._constraints.values():
            if tag in info.tags:
                info.enabled = False

    ###########################################################################
    # Solving with Debugging
    ###########################################################################

    def solve(self, solver: Optional[_cp.CpSolver] = None, **solver_params) -> _cp.CpSolverStatus:
        """
        Solve the model respecting constraint enable/disable flags.
        
        Args:
            solver: Optional solver instance. If None, creates a new one.
            **solver_params: Parameters to set on the solver.
            
        Returns:
            Solver status.
        """
        solver = solver or _cp.CpSolver()
        
        # Apply solver parameters
        for param_name, value in solver_params.items():
            if hasattr(solver.parameters, param_name):
                setattr(solver.parameters, param_name, value)
            else:
                raise ValueError(f"Unknown solver parameter: {param_name}")

        # Create a copy of the model for solving with constraints fixed
        model = self._create_model()
        return solver.Solve(model)

    def debug_infeasible(self, solver: Optional[_cp.CpSolver] = None, **solver_params) -> Dict[str, Any]:
        """
        Find a minimal set of constraints to disable to make the model feasible.
        
        Args:
            solver: Optional solver instance. If None, creates a new one.
            **solver_params: Parameters to set on the solver.
            
        Returns:
            Dictionary with debugging information including disabled constraints.
        """
        solver = solver or _cp.CpSolver()
        
        # Apply solver parameters
        for param_name, value in solver_params.items():
            if hasattr(solver.parameters, param_name):
                setattr(solver.parameters, param_name, value)
            else:
                raise ValueError(f"Unknown solver parameter: {param_name}")

        # Create debug model that minimizes disabled constraints
        debug_model = self._create_debug_optimization_model()
        
        # Solve the debug model
        status = solver.Solve(debug_model)
        
        if status in [_cp.OPTIMAL, _cp.FEASIBLE]:
            # Extract which constraints were disabled
            disabled_constraints = []
            for name, info in self._constraints.items():
                if info.enabled:  # Only check constraints that should be enabled
                    # Find the corresponding enable var in the debug model
                    enable_var_name = f"_enable_{name}"
                    # The enable var is 0 when constraint is disabled
                    if solver.Value(debug_model.GetVarValueMap()[enable_var_name]) == 0:
                        disabled_constraints.append(name)
            
            return {
                "status": status,
                "feasible": True,
                "disabled_constraints": disabled_constraints,
                "total_disabled": len(disabled_constraints),
            }
        else:
            return {
                "status": status,
                "feasible": False,
                "disabled_constraints": [],
                "total_disabled": 0,
            }

    ###########################################################################
    # Debugging and Introspection
    ###########################################################################

    def get_constraint_info(self, name: str) -> ConstraintInfo:
        """Get detailed information about a constraint."""
        if name not in self._constraints:
            raise ValueError(f"Constraint '{name}' not found")
        return self._constraints[name]

    def get_variable_info(self, name: str) -> VariableInfo:
        """Get detailed information about a variable."""
        if name not in self._variables:
            raise ValueError(f"Variable '{name}' not found")
        return self._variables[name]

    def get_constraint_names(self) -> List[str]:
        """Get all constraint names."""
        return list(self._constraints.keys())

    def get_variable_names(self) -> List[str]:
        """Get all variable names."""
        return list(self._variables.keys())

    def get_constraints_by_type(self, constraint_type: str) -> List[str]:
        """Get all constraints of a specific type."""
        return [name for name, info in self._constraints.items() 
                if info.constraint_type == constraint_type]

    def get_variables_by_type(self, var_type: str) -> List[str]:
        """Get all variables of a specific type."""
        return [name for name, info in self._variables.items() 
                if info.var_type == var_type]

    def get_constraints_by_tag(self, tag: str) -> List[str]:
        """Get all constraints with a specific tag."""
        return [name for name, info in self._constraints.items() 
                if tag in info.tags]

    def get_enabled_constraints(self) -> List[str]:
        """Get all currently enabled constraints."""
        return [name for name, info in self._constraints.items() if info.enabled]

    def get_disabled_constraints(self) -> List[str]:
        """Get all currently disabled constraints."""
        return [name for name, info in self._constraints.items() if not info.enabled]

    def summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the model state."""
        constraint_types = {}
        for info in self._constraints.values():
            constraint_types[info.constraint_type] = constraint_types.get(info.constraint_type, 0) + 1

        variable_types = {}
        for info in self._variables.values():
            variable_types[info.var_type] = variable_types.get(info.var_type, 0) + 1

        return {
            "total_constraints": len(self._constraints),
            "enabled_constraints": len(self.get_enabled_constraints()),
            "disabled_constraints": len(self.get_disabled_constraints()),
            "constraint_types": constraint_types,
            "total_variables": len(self._variables),
            "variable_types": variable_types,
            "all_tags": list(set().union(*(info.tags for info in self._constraints.values()))),
        }

    def validate_model(self) -> Dict[str, Any]:
        """
        Validate the current model state and return diagnostic information.
        
        Returns:
            Dictionary with validation results and potential issues.
        """
        issues = []
        warnings = []

        # Check for disabled constraints
        disabled = self.get_disabled_constraints()
        if disabled:
            warnings.append(f"{len(disabled)} constraints are disabled")

        # Check for unused variables (variables not referenced in any enabled constraint)
        # This is a simplified check - full implementation would need constraint analysis
        referenced_vars = set()
        for info in self._constraints.values():
            if info.enabled:
                # This would need proper constraint parsing to be complete
                pass

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "disabled_constraints": disabled,
        }

    ###########################################################################
    # Internal Helpers
    ###########################################################################

    def _register_constraint(
        self,
        constraint: _cp.Constraint,
        original_args: Any,
        constraint_type: str,
        name: Optional[str]
    ) -> _cp.Constraint:
        """Register a constraint with full metadata and enable variable."""
        # Generate unique name if not provided
        if name is None:
            name = f"{constraint_type.lower()}_{self._constraint_counter}"
        elif name in self._constraints:
            raise ValueError(f"Constraint name '{name}' already exists")
        
        self._constraint_counter += 1

        # Create enable variable and attach to constraint
        enable_var = super().NewBoolVar(f"_enable_{name}")
        constraint.OnlyEnforceIf(enable_var)

        # Store constraint information
        self._constraints[name] = ConstraintInfo(
            name=name,
            original_args=original_args,
            constraint_type=constraint_type,
            ortools_ct=constraint,
            enable_var=enable_var,
        )

        return constraint

    def _create_model(self) -> _cp.CpModel:
        """Create a copy of this model with constraints fixed according to enabled flags."""
        debug_model = _cp.CpModel()
        
        # Copy this model's proto
        debug_model.CopyFrom(self)
        
        # Add constraints to fix enable variables according to enabled flags
        for info in self._constraints.values():
            if info.enabled:
                debug_model.Add(info.enable_var == 1)
            else:
                debug_model.Add(info.enable_var == 0)
                
        return debug_model

    def _create_debug_optimization_model(self) -> _cp.CpModel:
        """Create a model that minimizes the number of disabled constraints."""
        debug_model = _cp.CpModel()
        
        # Copy this model's proto
        debug_model.CopyFrom(self)
        
        # For disabled constraints, force them to stay disabled
        # For enabled constraints, allow them to be disabled but minimize this
        disabled_vars = []
        for info in self._constraints.values():
            if not info.enabled:
                # Force disabled constraints to stay disabled
                debug_model.Add(info.enable_var == 0)
            else:
                # For enabled constraints, add (1 - enable_var) to objective
                # This counts how many enabled constraints get disabled
                disabled_vars.append(1 - info.enable_var)
        
        # Minimize the number of enabled constraints that get disabled
        if disabled_vars:
            debug_model.Minimize(sum(disabled_vars))
            
        return debug_model

    def __len__(self) -> int:
        """Return the number of constraints."""
        return len(self._constraints)

    def __contains__(self, name: str) -> bool:
        """Check if a constraint name exists."""
        return name in self._constraints

    def __getitem__(self, name: str) -> ConstraintInfo:
        """Get constraint info by name."""
        return self.get_constraint_info(name)

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"EnhancedCpModel(constraints={len(self._constraints)}, variables={len(self._variables)})"