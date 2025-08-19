from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Sequence, Tuple
from ortools.sat.python import cp_model as _cp
import json
import zipfile

from base import EnhancedConstructorsMixin


###############################################################################
# Helper classes for debugging
###############################################################################
class ConstraintInfo:
    """Wrapper around a single constraint with metadata for debug/enable/disable."""

    __slots__ = (
        "name",
        "original_args",
        "constraint_type",
        "ortools_ct",
        "enable_var",
        "enabled",
        "tags",
        "user_enforcement_literals",
    )

    def __init__(
        self,
        name: str,
        original_args: Any,
        constraint_type: str,
        ortools_ct: _cp.Constraint,
        enable_var: _cp.IntVar,
    ):
        self.name = name
        self.original_args = original_args  # tuple/list of arguments used to build ct
        self.constraint_type = constraint_type
        self.ortools_ct = ortools_ct        # the actual OR-Tools Constraint
        self.enable_var = enable_var        # _enable_<name>, not attached to ct
        self.enabled = True                 # logical flag; solver wrapper enforces it
        self.tags: set[str] = set()
        self.user_enforcement_literals: List[_cp.LiteralT] = []  # from OnlyEnforceIf()

class VariableInfo:
    """Rich wrapper around variables with metadata."""
    __slots__ = ("name", "var_type", "ortools_var", "creation_args")

    def __init__(self, name: str, var_type: str, ortools_var: Union[_cp.IntVar, _cp.IntervalVar], creation_args: tuple):
        self.name = name
        self.var_type = var_type
        self.ortools_var = ortools_var
        self.creation_args = creation_args


class ObjectiveInfo:
    """Wrapper around objective with metadata."""
    __slots__ = ("objective_type", "linear_expr", "enabled", "name")

    def __init__(self, objective_type: str, linear_expr: _cp.LinearExprT, name: Optional[str] = None):
        self.objective_type = objective_type
        self.linear_expr = linear_expr
        self.enabled = True
        self.name = name or f"objective_{objective_type.lower()}"

################################################################################
# Constraint Proxy for user-friendly constraint manipulation

class _ConstraintProxy:
    """Proxy returned by Add()/etc. to capture user calls like OnlyEnforceIf."""

    def __init__(self, ct: _cp.Constraint, info: ConstraintInfo):
        self._ct = ct
        self._info = info

    def OnlyEnforceIf(self, lits):
        if not isinstance(lits, (list, tuple)):
            lits = [lits]
        self._info.user_enforcement_literals.extend(lits)
        self._ct.OnlyEnforceIf(lits)
        return self

    def WithName(self, name: str):
        self._ct.WithName(name)
        self._info.name = name
        return self

    def __getattr__(self, attr):
        return getattr(self._ct, attr)


###############################################################################
class EnhancedCpModel(EnhancedConstructorsMixin, _cp.CpModel):
    """
    Drop-in replacement for CpModel that provides:
    - Complete constraint and variable registration
    - Assumption-based constraint enable/disable
    - Rich debugging and introspection capabilities
    - Minimal Infeasible Subset (MIS) finding
    - Proper model cloning capabilities
    - Objective management with enable/disable
    """

    def __init__(self) -> None:
        super().__init__()
        
        # Core registries
        self._constraints: Dict[str, ConstraintInfo] = {}
        self._variables: Dict[str, VariableInfo] = {}
        self._objectives: List[ObjectiveInfo] = []
        self._constraint_counter = 0
        self._variable_counter = 0

    ###########################################################################
    # Model Cloning and Copying Methods
    ###########################################################################
    
    def clone(self) -> 'EnhancedCpModel':
        """
        Create a complete clone by recreating variables, constraints, and objectives.
        """
        cloned = EnhancedCpModel()
        var_mapping = {}

        # Recreate variables (non-intervals first to avoid dependency issues)
        for name, info in self._variables.items():
            if info.var_type in ("IntervalVar", "OptionalIntervalVar"):
                continue
            args = info.creation_args
            if info.var_type == "IntVar":
                lb, ub, _ = args
                new_var = cloned.NewIntVar(lb, ub, name)
            elif info.var_type == "BoolVar":
                new_var = cloned.NewBoolVar(name)
            elif info.var_type == "Constant":
                value, = args
                new_var = cloned.NewConstant(value)
            var_mapping[name] = new_var

        # Recreate interval variables (depend on expressions)
        for name, info in self._variables.items():
            if info.var_type not in ("IntervalVar", "OptionalIntervalVar"):
                continue
            args = info.creation_args
            if info.var_type == "IntervalVar":
                start, size, end, _ = args
                new_start = cloned._map_expr_to_new_model(start, var_mapping)
                new_size = cloned._map_expr_to_new_model(size, var_mapping)
                new_end = cloned._map_expr_to_new_model(end, var_mapping)
                new_var = cloned.NewIntervalVar(new_start, new_size, new_end, name)
            else:
                start, size, end, is_present, _ = args
                new_start = cloned._map_expr_to_new_model(start, var_mapping)
                new_size = cloned._map_expr_to_new_model(size, var_mapping)
                new_end = cloned._map_expr_to_new_model(end, var_mapping)
                new_is_present = cloned._map_expr_to_new_model(is_present, var_mapping)
                new_var = cloned.NewOptionalIntervalVar(new_start, new_size, new_end, new_is_present, name)
            var_mapping[name] = new_var

        # Recreate constraints
        for info in self._constraints.values():
            cloned._recreate_constraint_in_model(cloned, info, var_mapping)

        # Copy constraint metadata states (enabled, tags, etc.)
        for name, original_info in self._constraints.items():
            new_info = cloned._constraints[name]
            new_info.enabled = original_info.enabled
            new_info.tags = original_info.tags.copy()
            # user_enforcement_literals are already replayed in recreation

        # Recreate objectives
        for obj in self._objectives:
            new_expr = cloned._map_expr_to_new_model(obj.linear_expr, var_mapping)
            new_obj = ObjectiveInfo(obj.objective_type, new_expr, obj.name)
            new_obj.enabled = obj.enabled
            cloned._objectives.append(new_obj)
            if obj.objective_type == "Minimize":
                cloned.Minimize(new_expr)
            elif obj.objective_type == "Maximize":
                cloned.Maximize(new_expr)

        # Copy counters
        cloned._constraint_counter = self._constraint_counter
        cloned._variable_counter = self._variable_counter

        return cloned
    
    def _clear_model(self) -> None:
        """Clear all model contents and metadata."""
        # Clear the underlying proto by creating a new one
        self.__dict__.update(_cp.CpModel().__dict__)
        
        # Clear metadata
        self._constraints.clear()
        self._variables.clear()
        self._objectives.clear()
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
        """Create a new constant with a unique name."""
        base_name = f"_const_{value}"
        name = base_name
        counter = self._variable_counter
        while name in self._variables or self.get_variable_by_name(name) is not None:
            counter += 1
            name = f"{base_name}_{counter}"
        self._variable_counter = counter + 1
        
        var = super().NewConstant(value)
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

    # Missing constraint types
    def AddMinEquality(self, target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == min(variables)."""
        return self._register_constraint(
            constraint=super().AddMinEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MinEquality",
            name=name
        )

    def AddMaxEquality(self, target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == max(variables)."""
        return self._register_constraint(
            constraint=super().AddMaxEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MaxEquality",
            name=name
        )
    
    def AddMultiplicationEquality(self, target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == variables[0] * variables[1] * ..."""
        return self._register_constraint(
            constraint=super().AddMultiplicationEquality(target, variables),
            original_args=(target, tuple(variables)),
            constraint_type="MultiplicationEquality",
            name=name
        )
    
    def AddDivisionEquality(self, target: _cp.LinearExprT, numerator: _cp.LinearExprT, denominator: _cp.LinearExprT, name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == numerator // denominator."""
        return self._register_constraint(
            constraint=super().AddDivisionEquality(target, numerator, denominator),
            original_args=(target, numerator, denominator),
            constraint_type="DivisionEquality",
            name=name
        )

    def AddAbsEquality(self, target: _cp.LinearExprT, variable: _cp.LinearExprT, name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == abs(variable)."""
        return self._register_constraint(
            constraint=super().AddAbsEquality(target, variable),
            original_args=(target, variable),
            constraint_type="AbsEquality",
            name=name
        )

    def AddModuloEquality(self, target: _cp.LinearExprT, variable: _cp.LinearExprT, modulo: _cp.LinearExprT, name: Optional[str] = None) -> _cp.Constraint:
        """Add a constraint that target == variable % modulo."""
        return self._register_constraint(
            constraint=super().AddModuloEquality(target, variable, modulo),
            original_args=(target, variable, modulo),
            constraint_type="ModuloEquality",
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
    # Objective Methods
    ###########################################################################

    def Minimize(self, obj: _cp.LinearExprT, name: Optional[str] = None) -> None:
        """Set the objective to minimize the given expression."""
        super().Minimize(obj)
        self._objectives.append(ObjectiveInfo("Minimize", obj, name))

    def Maximize(self, obj: _cp.LinearExprT, name: Optional[str] = None) -> None:
        """Set the objective to maximize the given expression."""
        super().Maximize(obj)
        self._objectives.append(ObjectiveInfo("Maximize", obj, name))

    def enable_objective(self, name: str) -> None:
        """Enable an objective by name, disabling all others."""
        for obj in self._objectives:
            obj.enabled = (obj.name == name)
        if not any(obj.enabled for obj in self._objectives):
            raise ValueError(f"Objective '{name}' not found")

    def disable_objective(self, name: str) -> None:
        """Disable an objective by name."""
        for obj in self._objectives:
            if obj.name == name:
                obj.enabled = False
                return
        raise ValueError(f"Objective '{name}' not found")

    def get_enabled_objective(self) -> List[ObjectiveInfo]:
        """Get all currently enabled objectives."""
        enabled = [obj for obj in self._objectives if obj.enabled]
        if len(enabled) > 1:
            raise RuntimeError("Multiple objectives enabled; only one is allowed")
        return enabled[0] if enabled else None

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

    def add_constraint_tags(self, name: str, tags: List[str]) -> None:
        """Add multiple tags to a constraint for group operations."""
        if name not in self._constraints:
            raise ValueError(f"Constraint '{name}' not found")
        for tag in tags:
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

    def _solve(self, solver: Optional[_cp.CpSolver] = None, **solver_params) -> int:
        """
        Solve the model respecting constraint enable/disable flags. Method only intended for model debugging purposes.
        
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

        # Create a proper copy with enable/disable constraints applied
        solving_model = self._create_solving_model()
        return solver.Solve(solving_model)

    def _create_solving_model(self) -> _cp.CpModel:
        """
        Create a copy of this model with enable variables fixed according to enabled flags.
        This method properly handles variable mapping between original and copy.
        """
        # Create new model
        solving_model = _cp.CpModel()
        
        # Create variable name to new variable mapping
        var_name_to_new_var = {}
        
        # First pass: recreate all user variables (not enable variables)
        for var_name, var_info in self._variables.items():
            if not var_name.startswith('_enable_'):  # Skip enable variables for now
                if var_info.var_type == "IntVar":
                    lb, ub, name = var_info.creation_args
                    new_var = solving_model.NewIntVar(lb, ub, name)
                elif var_info.var_type == "BoolVar":
                    name = var_info.creation_args[0]
                    new_var = solving_model.NewBoolVar(name)
                elif var_info.var_type == "IntervalVar":
                    start, size, end, name = var_info.creation_args
                    # Map start, size, end to new variables if they are variables
                    new_start = self._map_expr_to_new_model(start, var_name_to_new_var)
                    new_size = self._map_expr_to_new_model(size, var_name_to_new_var)
                    new_end = self._map_expr_to_new_model(end, var_name_to_new_var)
                    new_var = solving_model.NewIntervalVar(new_start, new_size, new_end, name)
                elif var_info.var_type == "OptionalIntervalVar":
                    start, size, end, is_present, name = var_info.creation_args
                    new_start = self._map_expr_to_new_model(start, var_name_to_new_var)
                    new_size = self._map_expr_to_new_model(size, var_name_to_new_var)
                    new_end = self._map_expr_to_new_model(end, var_name_to_new_var)
                    new_is_present = self._map_expr_to_new_model(is_present, var_name_to_new_var)
                    new_var = solving_model.NewOptionalIntervalVar(new_start, new_size, new_end, new_is_present, name)
                elif var_info.var_type == "Constant":
                    value = var_info.creation_args[0]
                    new_var = solving_model.NewConstant(value)
                else:
                    continue  # Skip unknown variable types
                
                var_name_to_new_var[var_name] = new_var
        
        # Second pass: recreate constraints with mapped variables
        for constraint_name, constraint_info in self._constraints.items():
            if constraint_info.enabled:
                # Only add enabled constraints - skip disabled ones entirely
                try:
                    self._recreate_constraint_in_model(
                        solving_model, 
                        constraint_info, 
                        var_name_to_new_var
                    )
                except Exception as e:
                    # Skip constraints that can't be recreated
                    print(f"Warning: Could not recreate constraint {constraint_name}: {e}")
                    continue

        # Add enabled objectives
        enabled_objective = self.get_enabled_objective()
        if enabled_objective:
            mapped_expr = self._map_expr_to_new_model(enabled_objective.linear_expr, var_name_to_new_var)
            if enabled_objective.objective_type == "Minimize":
                solving_model.Minimize(mapped_expr)
            else:
                solving_model.Maximize(mapped_expr)
        
        return solving_model

    def _map_expr_to_new_model(self, expr, var_mapping):
        """Return expr with all variables remapped via var_mapping."""
        if isinstance(expr, int):
            return expr
        if hasattr(expr, 'Name') and expr.Name() and expr.Name() in var_mapping:
            return var_mapping[expr.Name()]
        # Recursive handling for linear expressions
        return self._deep_map_expr(expr, var_mapping)
    
    def _deep_map_expr(self, expr, var_mapping):
        """Recursively rebuild linear expressions with remapped variables."""
        if isinstance(expr, int):
            return expr
        # LinearExpr API: LinearExpr.Sum, LinearExpr.WeightedSum, etc.
        if hasattr(expr, 'GetVars'):
            new_terms = []
            for var, coeff in expr.GetVars():
                new_var = self._map_expr_to_new_model(var, var_mapping)
                new_terms.append((new_var, coeff))
            const = getattr(expr, 'Offset', 0)
            new_expr = _cp.LinearExpr.WeightedSum([v for v, _ in new_terms],
                                                  [c for _, c in new_terms])
            if const:
                new_expr += const
            return new_expr
        # Fallback for anything else
        return expr
    
    def _recreate_constraint_in_model(
        self,
        model: _cp.CpModel,
        constraint_info: ConstraintInfo,
        var_mapping: Dict[str, Any]
    ) -> None:
        """
        Recreate a constraint in a new model with mapped variables.
        - Replays only user-provided enforcements.
        - Does not attach enable_var; solver/debug will handle that.

        Args:
            model: The target model to add the constraint to
            constraint_info: The constraint info from the original model
            var_mapping: Mapping from variable names to new variables
        """
        constraint_type = constraint_info.constraint_type
        args = constraint_info.original_args

        # Helper to map arguments, handling both serialized (str/dict/list) and OR-Tools objects
        def map_arg(arg):
            if isinstance(arg, str) and arg in var_mapping:
                return var_mapping[arg]
            elif isinstance(arg, dict) and arg.get("type") == "LinearExpr":
                vars_coeffs = arg.get("vars", [])
                constant = arg.get("constant", 0)
                vars = [var_mapping[var_name] for var_name, _ in vars_coeffs]
                coeffs = [coeff for _, coeff in vars_coeffs]
                return _cp.LinearExpr.WeightedSum(vars, coeffs) + constant
            elif isinstance(arg, (list, tuple)) and all(isinstance(a, (int, str)) for a in arg):
                return [map_arg(a) for a in arg]
            elif isinstance(arg, (list, tuple)) and all(isinstance(a, (list, tuple)) for a in arg):
                return [tuple(map_arg(b) for b in a) for a in arg]
            elif isinstance(arg, (int, bool)):
                return arg
            elif isinstance(arg, _cp.Domain):
                return arg  # Domain objects are not mapped, assumed to be unchanged
            else:
                return self._map_expr_to_new_model(arg, var_mapping)

        # Recreate the constraint based on type
        if constraint_type == "Generic":
            if isinstance(args, str) and args in var_mapping:
                mapped_expr = var_mapping[args]
            else:
                mapped_expr = map_arg(args)
            new_ct = model.Add(mapped_expr)

        elif constraint_type == "LinearConstraint":
            linear_expr, lb, ub = args
            mapped_expr = map_arg(linear_expr)
            new_ct = model.AddLinearConstraint(mapped_expr, lb, ub)

        elif constraint_type == "LinearExpressionInDomain":
            linear_expr, domain = args
            mapped_expr = map_arg(linear_expr)
            # Domain may be serialized as list of intervals
            mapped_domain = _cp.Domain.FromIntervals(domain) if isinstance(domain, (list, tuple)) else domain
            new_ct = model.AddLinearExpressionInDomain(mapped_expr, mapped_domain)

        elif constraint_type == "AllDifferent":
            variables = args
            mapped_vars = [map_arg(var) for var in variables]
            new_ct = model.AddAllDifferent(mapped_vars)

        elif constraint_type == "Element":
            index, variables, target = args
            mapped_index = map_arg(index)
            mapped_vars = [map_arg(var) for var in variables]
            mapped_target = map_arg(target)
            new_ct = model.AddElement(mapped_index, mapped_vars, mapped_target)

        elif constraint_type == "Circuit":
            arcs = args
            mapped_arcs = [(head, tail, map_arg(lit)) for head, tail, lit in arcs]
            new_ct = model.AddCircuit(mapped_arcs)

        elif constraint_type == "MultipleCircuit":
            arcs = args
            mapped_arcs = [(head, tail, map_arg(lit)) for head, tail, lit in arcs]
            new_ct = model.AddMultipleCircuit(mapped_arcs)

        elif constraint_type == "AllowedAssignments":
            variables, tuples_list = args
            mapped_vars = [map_arg(var) for var in variables]
            mapped_tuples = tuples_list if isinstance(tuples_list, (list, tuple)) else tuples_list
            new_ct = model.AddAllowedAssignments(mapped_vars, mapped_tuples)

        elif constraint_type == "ForbiddenAssignments":
            variables, tuples_list = args
            mapped_vars = [map_arg(var) for var in variables]
            mapped_tuples = tuples_list if isinstance(tuples_list, (list, tuple)) else tuples_list
            new_ct = model.AddForbiddenAssignments(mapped_vars, mapped_tuples)

        elif constraint_type == "Automaton":
            transition_variables, starting_state, final_states, transition_triples = args
            mapped_vars = [map_arg(var) for var in transition_variables]
            mapped_final_states = final_states if isinstance(final_states, (list, tuple)) else final_states
            mapped_triples = transition_triples if isinstance(transition_triples, (list, tuple)) else transition_triples
            new_ct = model.AddAutomaton(mapped_vars, starting_state, mapped_final_states, mapped_triples)

        elif constraint_type == "Inverse":
            variables, inverse_variables = args
            mapped_vars = [map_arg(var) for var in variables]
            mapped_inv_vars = [map_arg(var) for var in inverse_variables]
            new_ct = model.AddInverse(mapped_vars, mapped_inv_vars)

        elif constraint_type == "ReservoirConstraint":
            times, level_changes, min_level, max_level = args
            mapped_times = [map_arg(t) if isinstance(t, (int, str)) else self._map_expr_to_new_model(t, var_mapping) for t in times]
            mapped_changes = [map_arg(lc) if isinstance(lc, (int, str)) else self._map_expr_to_new_model(lc, var_mapping) for lc in level_changes]
            new_ct = model.AddReservoirConstraint(mapped_times, mapped_changes, min_level, max_level)

        elif constraint_type == "MinEquality":
            target, variables = args
            mapped_target = map_arg(target)
            mapped_vars = [map_arg(var) for var in variables]
            new_ct = model.AddMinEquality(mapped_target, mapped_vars)

        elif constraint_type == "MaxEquality":
            target, variables = args
            mapped_target = map_arg(target)
            mapped_vars = [map_arg(var) for var in variables]
            new_ct = model.AddMaxEquality(mapped_target, mapped_vars)

        elif constraint_type == "DivisionEquality":
            target, numerator, denominator = args
            mapped_target = map_arg(target)
            mapped_num = map_arg(numerator)
            mapped_den = map_arg(denominator)
            new_ct = model.AddDivisionEquality(mapped_target, mapped_num, mapped_den)

        elif constraint_type == "MultiplicationEquality":
            target, variables = args
            mapped_target = map_arg(target)
            mapped_vars = [map_arg(var) for var in variables]
            new_ct = model.AddMultiplicationEquality(mapped_target, mapped_vars)

        elif constraint_type == "AbsEquality":
            target, variable = args
            mapped_target = map_arg(target)
            mapped_var = map_arg(variable)
            new_ct = model.AddAbsEquality(mapped_target, mapped_var)

        elif constraint_type == "ModuloEquality":
            target, variable, modulo = args
            mapped_target = map_arg(target)
            mapped_var = map_arg(variable)
            mapped_mod = map_arg(modulo)
            new_ct = model.AddModuloEquality(mapped_target, mapped_var, mapped_mod)

        elif constraint_type == "BoolOr":
            literals = args
            mapped_literals = [map_arg(lit) for lit in literals]
            new_ct = model.AddBoolOr(mapped_literals)

        elif constraint_type == "BoolAnd":
            literals = args
            mapped_literals = [map_arg(lit) for lit in literals]
            new_ct = model.AddBoolAnd(mapped_literals)

        elif constraint_type == "BoolXor":
            literals = args
            mapped_literals = [map_arg(lit) for lit in literals]
            new_ct = model.AddBoolXor(mapped_literals)

        elif constraint_type == "Implication":
            a, b = args
            mapped_a = map_arg(a)
            mapped_b = map_arg(b)
            new_ct = model.AddImplication(mapped_a, mapped_b)

        elif constraint_type == "NoOverlap":
            intervals = args
            mapped_intervals = [map_arg(interval) for interval in intervals]
            new_ct = model.AddNoOverlap(mapped_intervals)

        elif constraint_type == "NoOverlap2D":
            x_intervals, y_intervals = args
            mapped_x = [map_arg(interval) if isinstance(interval, str) else self._map_expr_to_new_model(interval, var_mapping) for interval in x_intervals]
            mapped_y = [map_arg(interval) if isinstance(interval, str) else self._map_expr_to_new_model(interval, var_mapping) for interval in y_intervals]
            new_ct = model.AddNoOverlap2D(mapped_x, mapped_y)

        elif constraint_type == "Cumulative":
            intervals, demands, capacity = args
            mapped_intervals = [map_arg(interval) if isinstance(interval, str) else self._map_expr_to_new_model(interval, var_mapping) for interval in intervals]
            mapped_demands = [map_arg(demand) if isinstance(demand, (int, str)) else self._map_expr_to_new_model(demand, var_mapping) for demand in demands]
            mapped_capacity = map_arg(capacity) if isinstance(capacity, (int, str)) else self._map_expr_to_new_model(capacity, var_mapping)
            new_ct = model.AddCumulative(mapped_intervals, mapped_demands, mapped_capacity)

        else:
            raise ValueError(f"Unsupported constraint type for recreation: {constraint_type}")

        # Replay user enforcements
        if constraint_info.user_enforcement_literals:
            mapped_lits = [
                self._map_expr_to_new_model(lit, var_mapping)
                for lit in constraint_info.user_enforcement_literals
            ]
            new_ct.OnlyEnforceIf(mapped_lits)

    def debug_infeasible(self, solver: Optional[_cp.CpSolver] = None, **solver_params) -> Dict[str, Any]:
        """
        Find a minimal set of constraints to disable to make the model feasible.
        
        Returns:
            Dictionary with debugging information including minimal disabled constraints.
        """
        solver = solver or _cp.CpSolver()
        
        # Apply solver parameters
        for param_name, value in solver_params.items():
            if hasattr(solver.parameters, param_name):
                setattr(solver.parameters, param_name, value)
            else:
                raise ValueError(f"Unknown solver parameter: {param_name}")

        # First check if the model is already feasible
        simple_status = self._solve(solver)
        if simple_status in [_cp.OPTIMAL, _cp.FEASIBLE]:
            return {
                "status": simple_status,
                "feasible": True,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "already_feasible"
            }
        
        # Clone the model to create the MIS problem
        mis_model = self.clone()
        
        # Create disable variables for each enabled constraint
        disable_vars = []
        constraint_name_to_disable_var = {}
        
        enabled_constraint_names = self.get_enabled_constraints()
        
        for constraint_name in enabled_constraint_names:
            constraint_info = mis_model._constraints[constraint_name]
            
            # Get the existing enable variable from the cloned model
            enable_var = constraint_info.enable_var
            
            # Create disable variable: disable_var = 1 - enable_var
            disable_var = mis_model.NewBoolVar(f"disable_{constraint_name}")
            constraint_name_to_disable_var[constraint_name] = disable_var
            
            # Add constraint: disable_var + enable_var = 1
            mis_model.Add(disable_var + enable_var == 1)
            
            disable_vars.append(disable_var)
        
        # Set objective to minimize sum of disabled constraints
        if disable_vars:
            mis_model.Minimize(sum(disable_vars))
        else:
            return {
                "status": _cp.INFEASIBLE,
                "feasible": False,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "no_constraints_to_disable"
            }
        
        # Solve the MIS problem
        mis_status = solver.Solve(mis_model)
        
        if mis_status not in [_cp.OPTIMAL, _cp.FEASIBLE]:
            return {
                "status": mis_status,
                "feasible": False,
                "disabled_constraints": [],
                "total_disabled": 0,
                "method": "mis_solver_failed"
            }
        
        # Extract the solution - which constraints should be disabled
        disabled_constraints = []
        for constraint_name in enabled_constraint_names:
            if constraint_name in constraint_name_to_disable_var:
                disable_var = constraint_name_to_disable_var[constraint_name]
                if solver.Value(disable_var) == 1:
                    disabled_constraints.append(constraint_name)
        
        # Verify the solution by applying it to the original model
        original_states = {}
        for name, info in self._constraints.items():
            original_states[name] = info.enabled
        
        # Disable the identified constraints
        for constraint_name in disabled_constraints:
            self.disable_constraint(constraint_name)
        
        # Test feasibility
        verification_status = self._solve(solver)
        
        # Restore original constraint states
        for name, original_state in original_states.items():
            if original_state:
                self.enable_constraint(name)
            else:
                self.disable_constraint(name)
        
        return {
            "status": verification_status,
            "feasible": verification_status in [_cp.OPTIMAL, _cp.FEASIBLE],
            "disabled_constraints": disabled_constraints,
            "total_disabled": len(disabled_constraints),
            "method": "minimal_infeasible_set",
            "objective_value": solver.ObjectiveValue() if mis_status == _cp.OPTIMAL else None
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
    
    def get_variable_by_name(self, name: str) -> Optional[Union[_cp.IntVar, _cp.IntervalVar]]:
        """Get a variable by its name."""
        if name in self._variables:
            return self._variables[name].ortools_var
        return None

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
            "total_objectives": len(self._objectives),
            "enabled_objective": len(self.get_enabled_objective()),
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

        # Check for multiple enabled objectives
        enabled_objective = self.get_enabled_objective()

        # Check for variables not referenced in any enabled constraint
        # This is a basic check - could be enhanced with proper constraint analysis
        unused_vars = []
        for var_name, var_info in self._variables.items():
            if not var_name.startswith('_') and var_info.var_type != "Constant":
                # Simplified check: assume all non-internal variables are used
                pass

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "disabled_constraints": disabled,
            "unused_variables": unused_vars,
        }

    ###########################################################################
    # Internal Helpers
    ###########################################################################

    def _register_constraint(
        self,
        constraint: _cp.Constraint,
        original_args: Any,
        constraint_type: str,
        name: Optional[str],
        enforce_enable_var: bool = False
    ) -> _cp.Constraint:
        """Register a constraint with full metadata and optional enable variable enforcement."""
        if name is None:
            name = f"{constraint_type.lower()}_{self._constraint_counter}"
        elif name in self._constraints:
            raise ValueError(f"Constraint name '{name}' already exists")
        
        self._constraint_counter += 1

        enable_var = super().NewBoolVar(f"_enable_{name}")
        if enforce_enable_var:
            constraint.OnlyEnforceIf(enable_var) # Potential problem with this. Constraint saved, but onlyenforceif not saved.
        
        info = ConstraintInfo(
            name=name,
            original_args=original_args,
            constraint_type=constraint_type,
            ortools_ct=constraint,
            enable_var=enable_var,
        )
        self._constraints[name] = info

        return _ConstraintProxy(constraint, info)

    ###########################################################################
    # Export/Import Methods
    ###########################################################################
    
    def _serialize_arg(self, arg: Any) -> Any:
        """
        Serialize an argument to a JSON-compatible format.
        Handles OR-Tools objects (IntVar, IntervalVar, NotBooleanVariable, LinearExpr, Domain)
        and nested structures (lists, tuples).
        """
        if isinstance(arg, _cp.IntVar):
            return arg.Name()
        elif isinstance(arg, _cp.IntervalVar):
            return arg.Name()
        elif isinstance(arg, _cp.NotBooleanVariable):
                return {"type": "NotBoolean", "var": arg.GetVar().Name()}
        elif isinstance(arg, (int, str, bool)):
            return arg
        elif isinstance(arg, _cp.Domain):
            return list(arg.FlattenedIntervals())

            #Must be properly implemented, incomplete!
        elif isinstance(arg, _cp.LinearExpr):
                serialized = {"type": "LinearExpr"}
                if arg.IsConstant():
                    serialized["constant"] = int(arg)
                    serialized["vars"] = []
                else:
                    vars_coeffs = []
                    if hasattr(arg, 'GetVars'):
                        vars_coeffs = [(var.Name(), coeff) for var, coeff in arg.GetVars()]
                    else:
                        # Handle complex expressions (Sum, WeightedSum)
                        serialized["complex"] = True
                        serialized["str"] = str(arg)  # Fallback for debugging
                    serialized["vars"] = vars_coeffs
                    serialized["constant"] = getattr(arg, 'Offset', 0)
                return serialized
        elif isinstance(arg, (list, tuple)):
            # Handle nested lists/tuples (e.g., for AllowedAssignments, Circuit)
            return [self._serialize_arg(a) for a in arg]
        elif isinstance(arg, Sequence) and all(isinstance(a, tuple) for a in arg):
            # Handle sequence of tuples (e.g., tuples_list in AllowedAssignments)
            return [tuple(self._serialize_arg(b) for b in a) for a in arg]
        else:
            # Fallback: convert to string for debug purposes
            return str(arg)
        
    def _deserialize_arg(self, serialized_arg: Any, var_mapping: Dict[str, Any]) -> Any:
        """
        Rehydrate a serialized argument back to an OR-Tools object.
        Handles dicts for LinearExpr, strings for variables, lists for Domain/tuples, etc.
        """
        if isinstance(serialized_arg, str) and serialized_arg in var_mapping:
            return var_mapping[serialized_arg]
        elif isinstance(serialized_arg, dict) and serialized_arg.get("type") == "LinearExpr":
            vars_coeffs = serialized_arg.get("vars", [])
            constant = serialized_arg.get("constant", 0)
            vars = [var_mapping[var_name] for var_name, _ in vars_coeffs]
            coeffs = [coeff for _, coeff in vars_coeffs]
            return _cp.LinearExpr.WeightedSum(vars, coeffs) + constant
        elif isinstance(serialized_arg, dict) and "not" in serialized_arg:
            var = var_mapping[serialized_arg["not"]]
            return var.Not()
        elif isinstance(serialized_arg, list) and serialized_arg and isinstance(serialized_arg[0], (int, list)):
            # For Domain intervals or nested lists
            if all(isinstance(i, (list, tuple)) and len(i) == 2 for i in serialized_arg):
                return _cp.Domain.FromIntervals(serialized_arg)
            return [self._deserialize_arg(a, var_mapping) for a in serialized_arg]
        elif isinstance(serialized_arg, (list, tuple)):
            return [self._deserialize_arg(a, var_mapping) for a in serialized_arg]
        elif isinstance(serialized_arg, (int, bool)):
            return serialized_arg
        else:
            # Fallback: assume it's already an object or unhandled
            return serialized_arg
        
    def export_to_file(self, filename: str) -> None:
        """
        Persist the *entire* EnhancedCpModel (proto + metadata) to disk.

        The file is a zip archive with two entries:
            model.pb   – raw OR-Tools proto
            meta.json  – all metadata (variables, constraints, objectives, tags, etc.)
        """
        # 1. Serialize the OR-Tools proto
        proto_bytes = self.Proto().SerializeToString()

        # 2. Build a plain-python representation of metadata
        variables_meta: Dict[str, Dict[str, Any]] = {}
        for name, info in self._variables.items():
            variables_meta[name] = {
                "var_type": info.var_type,
                "creation_args": self._serialize_arg(info.creation_args),
            }

        constraints_meta: Dict[str, Dict[str, Any]] = {}
        for name, info in self._constraints.items():
            serialized_lits = [self._serialize_arg(lit) for lit in info.user_enforcement_literals]
            constraints_meta[name] = {
                "original_args": self._serialize_arg(info.original_args),
                "constraint_type": info.constraint_type,
                "enabled": info.enabled,
                "tags": list(info.tags),
                "user_enforcement_literals": serialized_lits,
            }

        objectives_meta = [
            {
                "objective_type": obj.objective_type,
                "name": obj.name,
                "enabled": obj.enabled,
                "linear_expr": self._serialize_arg(obj.linear_expr),
            }
            for obj in self._objectives
        ]

        meta = {
            "variables": variables_meta,
            "constraints": constraints_meta,
            "objectives": objectives_meta,
            "constraint_counter": self._constraint_counter,
            "variable_counter": self._variable_counter,
        }

        # 3. Write both pieces into a zip file
        with zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pb", proto_bytes)
            zf.writestr("meta.json", json.dumps(meta, separators=(",", ":")))

    def import_from_file(self, filename: str) -> None:
        """
        Load a model from a saved file, restoring OR-Tools proto and metadata.
        Rebuilds constraints from metadata to ensure consistency.
        """
        self._clear_model()

        with zipfile.ZipFile(filename, 'r') as zf:
            # 1. Load proto
            proto_bytes = zf.read("model.pb")
            from ortools.sat import cp_model_pb2
            proto = cp_model_pb2.CpModelProto()
            proto.ParseFromString(proto_bytes)
            self.Proto().CopyFrom(proto)

            # 2. Load metadata
            meta_raw = zf.read("meta.json").decode("utf-8")
            meta = json.loads(meta_raw)

            # 3. Variables
            self._variables.clear()
            for name, vmeta in meta["variables"].items():
                var_type = vmeta["var_type"]
                args = tuple(vmeta["creation_args"])
                ortools_var = self.get_variable_by_name(name)
                if ortools_var is None:
                    raise RuntimeError(f"Variable {name} not found in proto")
                self._variables[name] = VariableInfo(
                    name=name,
                    var_type=var_type,
                    ortools_var=ortools_var,
                    creation_args=args,
                )

            # Create var_mapping
            var_mapping = {name: info.ortools_var for name, info in self._variables.items()}

            # 4. Objectives
            self._objectives.clear()
            for ometa in meta["objectives"]:
                linear_expr = self._deserialize_arg(ometa["linear_expr"], var_mapping)
                obj = ObjectiveInfo(
                    objective_type=ometa["objective_type"],
                    linear_expr=linear_expr,
                    name=ometa["name"],
                )
                obj.enabled = ometa["enabled"]
                self._objectives.append(obj)
                if obj.enabled:
                    if obj.objective_type == "Minimize":
                        self.Minimize(linear_expr)
                    elif obj.objective_type == "Maximize":
                        self.Maximize(linear_expr)

            # 5. Constraints (rebuild to ensure consistency)
            self._constraints.clear()
            for name, cmeta in meta["constraints"].items():
                enable_name = f"_enable_{name}"
                enable_var = self.get_variable_by_name(enable_name)
                if enable_var is None:
                    enable_var = super().NewBoolVar(enable_name)

                rehydrated_args = self._deserialize_arg(cmeta["original_args"], var_mapping)
                info = ConstraintInfo(
                    name=name,
                    original_args=rehydrated_args,
                    constraint_type=cmeta["constraint_type"],
                    ortools_ct=None,  # Set by _recreate_constraint_in_model
                    enable_var=enable_var,
                )
                info.enabled = cmeta["enabled"]
                info.tags = set(cmeta["tags"])
                info.user_enforcement_literals = [
                    self._deserialize_arg(lit, var_mapping)
                    for lit in cmeta.get("user_enforcement_literals", [])
                    if self._deserialize_arg(lit, var_mapping) is not None
                ]

                # Rebuild constraint
                self._recreate_constraint_in_model(self, info, var_mapping)

            # 6. Counters
            self._constraint_counter = meta["constraint_counter"]
            self._variable_counter = meta["variable_counter"]

    ###########################################################################
    # Advanced Methods
    ###########################################################################
    
    def create_relaxed_copy(self, relaxation_factor: float = 0.1) -> 'EnhancedCpModel':
        """
        Create a relaxed version of this model by disabling some constraints.
        
        Args:
            relaxation_factor: Fraction of constraints to disable (0.0 to 1.0).
            
        Returns:
            A new EnhancedCpModel with some constraints disabled.
        """
        if not 0.0 <= relaxation_factor <= 1.0:
            raise ValueError("Relaxation factor must be between 0.0 and 1.0")
        
        relaxed_model = self.clone()
        
        # Calculate number of constraints to disable
        enabled_constraints = relaxed_model.get_enabled_constraints()
        num_to_disable = int(len(enabled_constraints) * relaxation_factor)
        
        if num_to_disable > 0:
            import random
            constraints_to_disable = random.sample(enabled_constraints, num_to_disable)
            relaxed_model.disable_constraints(constraints_to_disable)
        
        return relaxed_model
    
    def create_subset_copy(self, constraint_names: Sequence[str]) -> 'EnhancedCpModel':
        """
        Create a copy of this model with only specified constraints enabled.
        
        Args:
            constraint_names: Names of constraints to keep enabled.
            
        Returns:
            A new EnhancedCpModel with only specified constraints enabled.
        """
        subset_model = self.clone()
        
        # Disable all constraints first
        subset_model.disable_constraints(subset_model.get_constraint_names())
        
        # Enable only the specified constraints
        subset_model.enable_constraints(constraint_names)
        
        return subset_model

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
        return f"EnhancedCpModel(constraints={len(self._constraints)}, variables={len(self._variables)}, objectives={len(self._objectives)})"