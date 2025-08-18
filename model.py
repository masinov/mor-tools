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


class ObjectiveInfo:
    """Wrapper around objective with metadata."""
    __slots__ = ("objective_type", "linear_expr", "enabled", "name")

    def __init__(self, objective_type: str, linear_expr: _cp.LinearExprT, name: Optional[str] = None):
        self.objective_type = objective_type
        self.linear_expr = linear_expr
        self.enabled = True
        self.name = name or f"objective_{objective_type.lower()}"


###############################################################################
class EnhancedCpModel(_cp.CpModel):
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
        Create a complete clone of this model including all metadata.
        
        Returns:
            A new EnhancedCpModel instance that is an exact copy of this model.
        """
        cloned = EnhancedCpModel()
        
        # Copy the underlying proto
        cloned._clone_proto_from(self)
        
        # Copy metadata with adjusted variable references
        cloned._clone_metadata_from(self)
        
        return cloned
    
    def copy_from(self, other: Union['EnhancedCpModel', _cp.CpModel]) -> None:
        """
        Copy the contents of another model into this model.
        
        Args:
            other: The model to copy from. Can be EnhancedCpModel or regular CpModel.
        """
        # Clear current model
        self._clear_model()
        
        if isinstance(other, EnhancedCpModel):
            # Copy from another EnhancedCpModel - preserve all metadata
            self._clone_proto_from(other)
            self._clone_metadata_from(other)
        else:
            # Copy from regular CpModel - only copy the proto
            self._clone_proto_from(other)
            # Reset metadata since we don't have it from the source
            self._constraints.clear()
            self._variables.clear()
            self._objectives.clear()
            self._constraint_counter = 0
            self._variable_counter = 0
    
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
    
    def _clone_proto_from(self, other: _cp.CpModel) -> None:
        """
        Clone the protocol buffer from another CpModel.
        
        Args:
            other: The source model to clone from.
        """
        # Get the proto from the source model
        source_proto = other.Proto()
        
        # Create a copy of the proto and merge it into this model
        self.Proto().CopyFrom(source_proto)
    
    def _clone_metadata_from(self, other: 'EnhancedCpModel') -> None:
        """
        Clone metadata from another EnhancedCpModel using name-based mapping.
        
        Args:
            other: The source EnhancedCpModel to clone metadata from.
        """
        # Copy counters
        self._constraint_counter = other._constraint_counter
        self._variable_counter = other._variable_counter
        
        # Clone variables by name - keep original references since proto was copied
        self._variables = {}
        for var_name, var_info in other._variables.items():
            self._variables[var_name] = VariableInfo(
                name=var_info.name,
                var_type=var_info.var_type,
                ortools_var=var_info.ortools_var,
                creation_args=var_info.creation_args
            )
        
        # Clone constraints by name - keep original references since proto was copied
        self._constraints = {}
        for constraint_name, constraint_info in other._constraints.items():
            new_constraint_info = ConstraintInfo(
                name=constraint_info.name,
                original_args=constraint_info.original_args,
                constraint_type=constraint_info.constraint_type,
                ortools_ct=constraint_info.ortools_ct,
                enable_var=constraint_info.enable_var,
            )
            
            # Copy enabled state and tags
            new_constraint_info.enabled = constraint_info.enabled
            new_constraint_info.tags = constraint_info.tags.copy()
            
            self._constraints[constraint_name] = new_constraint_info
        
        # Clone objectives
        self._objectives = []
        for obj_info in other._objectives:
            self._objectives.append(ObjectiveInfo(
                objective_type=obj_info.objective_type,
                linear_expr=obj_info.linear_expr,
                name=obj_info.name
            ))

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
        # Constants get unique names to avoid conflicts
        name = f"_const_{value}_{id(var)}"
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
        """Enable an objective by name."""
        for obj in self._objectives:
            if obj.name == name:
                obj.enabled = True
                return
        raise ValueError(f"Objective '{name}' not found")

    def disable_objective(self, name: str) -> None:
        """Disable an objective by name."""
        for obj in self._objectives:
            if obj.name == name:
                obj.enabled = False
                return
        raise ValueError(f"Objective '{name}' not found")

    def get_enabled_objectives(self) -> List[ObjectiveInfo]:
        """Get all currently enabled objectives."""
        return [obj for obj in self._objectives if obj.enabled]

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
        enabled_objectives = self.get_enabled_objectives()
        if enabled_objectives:
            # Use the last enabled objective (OR-Tools only supports one objective)
            last_obj = enabled_objectives[-1]
            mapped_expr = self._map_expr_to_new_model(last_obj.linear_expr, var_name_to_new_var)
            if last_obj.objective_type == "Minimize":
                solving_model.Minimize(mapped_expr)
            else:
                solving_model.Maximize(mapped_expr)
        
        return solving_model

    def _map_expr_to_new_model(self, expr, var_mapping):
        """
        Map a linear expression from the original model to the new model.
        
        Args:
            expr: The expression to map (can be a variable, constant, or expression)
            var_mapping: Dictionary mapping old variable names to new variables
            
        Returns:
            The mapped expression in the new model
        """
        # If it's a simple integer, return as is
        if isinstance(expr, int):
            return expr
        
        # If it's a variable, look it up in our mapping
        if hasattr(expr, 'Name'):
            var_name = expr.Name()
            if var_name in var_mapping:
                return var_mapping[var_name]
            else:
                # Try to find by matching variable info
                for name, info in self._variables.items():
                    if info.ortools_var == expr and name in var_mapping:
                        return var_mapping[name]
                
                # If we can't find the variable, it might be a constant
                if hasattr(expr, 'Value'):
                    return expr.Value()
                else:
                    # Last resort: return the expression as-is
                    return expr
        
        # For complex expressions, we'd need more sophisticated mapping
        return expr

    def _recreate_constraint_in_model(self, model: _cp.CpModel, constraint_info: ConstraintInfo, var_mapping: Dict[str, Any]) -> None:
        """
        Recreate a constraint in the new model with mapped variables.
        
        Args:
            model: The target model to add the constraint to
            constraint_info: The constraint info from the original model
            var_mapping: Mapping from variable names to new variables
        """
        constraint_type = constraint_info.constraint_type
        args = constraint_info.original_args
        
        # Map the arguments to the new model
        if constraint_type == "Generic":
            mapped_expr = self._map_expr_to_new_model(args, var_mapping)
            model.Add(mapped_expr)
            
        elif constraint_type == "LinearConstraint":
            linear_expr, lb, ub = args
            mapped_expr = self._map_expr_to_new_model(linear_expr, var_mapping)
            model.AddLinearConstraint(mapped_expr, lb, ub)
            
        elif constraint_type == "LinearExpressionInDomain":
            linear_expr, domain = args
            mapped_expr = self._map_expr_to_new_model(linear_expr, var_mapping)
            model.AddLinearExpressionInDomain(mapped_expr, domain)
            
        elif constraint_type == "AllDifferent":
            variables = args
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            model.AddAllDifferent(mapped_vars)
            
        elif constraint_type == "Element":
            index, variables, target = args
            mapped_index = self._map_expr_to_new_model(index, var_mapping)
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            model.AddElement(mapped_index, mapped_vars, mapped_target)
            
        elif constraint_type == "Circuit":
            arcs = args
            mapped_arcs = [(head, tail, self._map_expr_to_new_model(lit, var_mapping)) for head, tail, lit in arcs]
            model.AddCircuit(mapped_arcs)
            
        elif constraint_type == "MultipleCircuit":
            arcs = args
            mapped_arcs = [(head, tail, self._map_expr_to_new_model(lit, var_mapping)) for head, tail, lit in arcs]
            model.AddMultipleCircuit(mapped_arcs)
            
        elif constraint_type == "AllowedAssignments":
            variables, tuples_list = args
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            model.AddAllowedAssignments(mapped_vars, tuples_list)
            
        elif constraint_type == "ForbiddenAssignments":
            variables, tuples_list = args
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            model.AddForbiddenAssignments(mapped_vars, tuples_list)
            
        elif constraint_type == "Automaton":
            transition_variables, starting_state, final_states, transition_triples = args
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in transition_variables]
            model.AddAutomaton(mapped_vars, starting_state, final_states, transition_triples)
            
        elif constraint_type == "Inverse":
            variables, inverse_variables = args
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            mapped_inv_vars = [self._map_expr_to_new_model(var, var_mapping) for var in inverse_variables]
            model.AddInverse(mapped_vars, mapped_inv_vars)
            
        elif constraint_type == "ReservoirConstraint":
            times, level_changes, min_level, max_level = args
            mapped_times = [self._map_expr_to_new_model(t, var_mapping) for t in times]
            mapped_changes = [self._map_expr_to_new_model(lc, var_mapping) for lc in level_changes]
            model.AddReservoirConstraint(mapped_times, mapped_changes, min_level, max_level)
            
        elif constraint_type == "MinEquality":
            target, variables = args
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            model.AddMinEquality(mapped_target, mapped_vars)
            
        elif constraint_type == "MaxEquality":
            target, variables = args
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            mapped_vars = [self._map_expr_to_new_model(var, var_mapping) for var in variables]
            model.AddMaxEquality(mapped_target, mapped_vars)
            
        elif constraint_type == "DivisionEquality":
            target, numerator, denominator = args
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            mapped_num = self._map_expr_to_new_model(numerator, var_mapping)
            mapped_den = self._map_expr_to_new_model(denominator, var_mapping)
            model.AddDivisionEquality(mapped_target, mapped_num, mapped_den)
            
        elif constraint_type == "AbsEquality":
            target, variable = args
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            mapped_var = self._map_expr_to_new_model(variable, var_mapping)
            model.AddAbsEquality(mapped_target, mapped_var)
            
        elif constraint_type == "ModuloEquality":
            target, variable, modulo = args
            mapped_target = self._map_expr_to_new_model(target, var_mapping)
            mapped_var = self._map_expr_to_new_model(variable, var_mapping)
            mapped_mod = self._map_expr_to_new_model(modulo, var_mapping)
            model.AddModuloEquality(mapped_target, mapped_var, mapped_mod)
            
        elif constraint_type == "BoolOr":
            literals = args
            mapped_literals = [self._map_expr_to_new_model(lit, var_mapping) for lit in literals]
            model.AddBoolOr(mapped_literals)
            
        elif constraint_type == "BoolAnd":
            literals = args
            mapped_literals = [self._map_expr_to_new_model(lit, var_mapping) for lit in literals]
            model.AddBoolAnd(mapped_literals)
            
        elif constraint_type == "BoolXor":
            literals = args
            mapped_literals = [self._map_expr_to_new_model(lit, var_mapping) for lit in literals]
            model.AddBoolXor(mapped_literals)
            
        elif constraint_type == "Implication":
            a, b = args
            mapped_a = self._map_expr_to_new_model(a, var_mapping)
            mapped_b = self._map_expr_to_new_model(b, var_mapping)
            model.AddImplication(mapped_a, mapped_b)
            
        elif constraint_type == "NoOverlap":
            intervals = args
            mapped_intervals = [self._map_expr_to_new_model(interval, var_mapping) for interval in intervals]
            model.AddNoOverlap(mapped_intervals)
            
        elif constraint_type == "NoOverlap2D":
            x_intervals, y_intervals = args
            mapped_x = [self._map_expr_to_new_model(interval, var_mapping) for interval in x_intervals]
            mapped_y = [self._map_expr_to_new_model(interval, var_mapping) for interval in y_intervals]
            model.AddNoOverlap2D(mapped_x, mapped_y)
            
        elif constraint_type == "Cumulative":
            intervals, demands, capacity = args
            mapped_intervals = [self._map_expr_to_new_model(interval, var_mapping) for interval in intervals]
            mapped_demands = [self._map_expr_to_new_model(demand, var_mapping) for demand in demands]
            mapped_capacity = self._map_expr_to_new_model(capacity, var_mapping)
            model.AddCumulative(mapped_intervals, mapped_demands, mapped_capacity)
            
        else:
            raise ValueError(f"Unsupported constraint type for recreation: {constraint_type}")

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
            "enabled_objectives": len(self.get_enabled_objectives()),
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
        enabled_objectives = self.get_enabled_objectives()
        if len(enabled_objectives) > 1:
            warnings.append(f"Multiple objectives enabled ({len(enabled_objectives)}), only the last will be used")
        
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
        name: Optional[str]
    ) -> _cp.Constraint:
        """Register a constraint with full metadata and enable variable."""
        # Generate unique name if not provided
        if name is None:
            name = f"{constraint_type.lower()}_{self._constraint_counter}"
        elif name in self._constraints:
            raise ValueError(f"Constraint name '{name}' already exists")
        
        self._constraint_counter += 1

        # Create enable variable and conditionally enforce constraint
        enable_var = super().NewBoolVar(f"_enable_{name}")
        
        # Check if constraint supports conditional enforcement
        try:
            constraint.OnlyEnforceIf(enable_var)
        except AttributeError:
            # If constraint doesn't support OnlyEnforceIf, we'll handle it differently
            # For now, just create the enable variable but don't attach it
            pass

        # Store constraint information
        self._constraints[name] = ConstraintInfo(
            name=name,
            original_args=original_args,
            constraint_type=constraint_type,
            ortools_ct=constraint,
            enable_var=enable_var,
        )

        return constraint

    ###########################################################################
    # Export/Import Methods
    ###########################################################################
    
    def export_to_file(self, filename: str) -> None:
        """
        Export the model to a file in proto format.
        
        Args:
            filename: Path to the output file.
        """
        # Write proto to file manually since ExportToFile doesn't exist
        with open(filename, 'wb') as f:
            f.write(self.Proto().SerializeToString())
    
    def import_from_file(self, filename: str) -> None:
        """
        Import a model from a file in proto format.
        
        Args:
            filename: Path to the input file.
        """
        # Clear current model
        self._clear_model()
        
        # Read proto from file manually since ImportFromFile doesn't exist
        with open(filename, 'rb') as f:
            proto_data = f.read()
        
        # Parse and load the proto
        from ortools.sat import cp_model_pb2
        proto = cp_model_pb2.CpModelProto()
        proto.ParseFromString(proto_data)
        self.Proto().CopyFrom(proto)
        
        # Note: Importing from file loses all EnhancedCpModel metadata

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