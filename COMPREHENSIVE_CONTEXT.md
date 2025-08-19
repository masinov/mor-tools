# Comprehensive Repository Analysis

Generated: La fecha actual es: 19/08/2025 
Escriba la nueva fecha: (dd-mm-aa)
Files analyzed: 8

# Enhanced CP-SAT Model Integration Guide

## Repository Architecture

This codebase extends Google's OR-Tools CP-SAT solver with enhanced functionality through a mixin-based architecture. The core components are:

- **EnhancedCpModel** (`model/model.py`): Main model class that inherits from `CpModel` and combines all mixins
- **Mixin Classes**: Modular functionality added through multiple inheritance:
  - `_ConstraintsMixin`: Enhanced constraint management with tracking
  - `_VariablesMixin`: Extended variable creation methods
  - `_ObjectivesMixin`: Objective function management
  - `_DebugMixin`: Debugging and infeasibility analysis tools
  - `_IOMixin`: Import/export capabilities
- **Constructor Tools** (`model/constructor_tools.py`): Helper functions for creating complex boolean expressions
- **Constraint Proxy** (`model/constraints.py`): Wrapper for constraint manipulation

## Module Dependencies

```
EnhancedCpModel (model.py)
    ├── _ConstraintsMixin (constraints.py)
    │   └── ConstraintInfo, _ConstraintProxy
    ├── _VariablesMixin (variables.py)
    ├── _ObjectivesMixin (objectives.py)
    ├── _DebugMixin (debug.py)
    ├── _IOMixin (io.py)
    └── EnhancedConstructorsMixin (constructors.py)
        └── Constructor Tools (constructor_tools.py)*
            ├── NewAndBoolVar, NewOrBoolVar, NewEqualBoolVar, etc.
            └── M (helper class)
```

*Note: Constructor tools are imported as functions, not as direct dependencies*

### Import Patterns

```python
# Core model creation
from model.model import EnhancedCpModel

# Constructor tools (optional)
from model.constructor_tools import NewAndBoolVar, NewOrBoolVar

# Typical usage
model = EnhancedCpModel()
```

## API Usage Patterns

### Basic Model Creation
```python
from model.model import EnhancedCpModel

model = EnhancedCpModel()
```

### Variable Creation
```python
# Standard variables
x = model.NewIntVar(0, 10, 'x')
b = model.NewBoolVar('b')

# Enhanced functionality (tracked variables)
y = model.NewIntVar(0, 100, 'y', track=True)
```

### Constraint Management
```python
# Standard constraints (automatically tracked)
model.Add(x + y <= 15)

# Enhanced constraints with names and enforcement
model.Add(x == 5, name='x_fixed').OnlyEnforceIf(b)

# Complex boolean expressions
b_and = model.NewAndBoolVar([b1, b2, b3])
model.Add(x > 0).OnlyEnforceIf(b_and)
```

### Objective Functions
```python
# Minimization
model.Minimize(x + 2*y)

# Maximization  
model.Maximize(3*x - y)
```

### Debugging
```python
# Check for infeasibility
if model.debug_infeasible():
    print("Model is infeasible")
    
# Create subset for debugging
subset = model.create_subset_copy(['constraint1', 'constraint2'])
```

### Import/Export
```python
# Export model
model.export_to_file('model.zip')

# Import model  
new_model = EnhancedCpModel()
new_model.import_from_file('model.zip')
```

## Integration Points

### 1. Constraint Registration System
- All constraints are wrapped in `_ConstraintProxy`
- `ConstraintInfo` tracks enforcement literals and metadata
- `_register_constraint()` handles the registration process

### 2. Variable Mapping
- `_map_expr_to_new_model()` and `_deep_map_expr()` handle expression recreation
- Essential for model copying and import/export functionality

### 3. Serialization System
- `_serialize_arg()` and `_deserialize_arg()` handle complex data types
- JSON-based metadata storage alongside protobuf model

### 4. Constructor Tools Integration
- Boolean expression helpers integrate seamlessly with constraint enforcement
- All tools return variables compatible with the constraint system

## Potential Issues

### Missing Imports (Critical)
The analysis shows numerous undefined references that will cause runtime errors:

```python
# Missing in constraints.py
from ortools.sat.python import cp_model as _cp
from ortools.sat.python.cp_model import LinearExpr, Domain

# Missing in various files
import json
import zipfile
import random
from ortools.sat import cp_model_pb2
```

### Circular Dependency Risks
- `constructors.py` depends on `constructor_tools.py`
- All mixins depend on the base `EnhancedCpModel`
- Careful import ordering required

### Interface Mismatches
- Constructor tools expect certain methods (`model.Add`, `model.NewBoolVar`) that must be available
- Serialization system assumes specific variable and constraint interfaces

### Performance Considerations
- Constraint tracking adds overhead
- Deep expression mapping can be expensive for large models
- Serialization creates additional memory usage

## Quick Start Guide

### 1. Installation and Setup
```python
# Install required packages
pip install ortools

# Import the enhanced model
from model.model import EnhancedCpModel
```

### 2. Basic Model Creation
```python
model = EnhancedCpModel()

# Create variables
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')
b = model.NewBoolVar('b')

# Add constraints
model.Add(x + y <= 15)
model.Add(x >= 5).OnlyEnforceIf(b)

# Set objective
model.Maximize(x + 2*y)
```

### 3. Using Constructor Tools
```python
from model.constructor_tools import NewAndBoolVar, NewOrBoolVar

# Create complex boolean expressions
b1 = model.NewBoolVar('b1')
b2 = model.NewBoolVar('b2')
b3 = model.NewBoolVar('b3')

and_var = NewAndBoolVar(model, [b1, b2, b3])
or_var = NewOrBoolVar(model, [b1, b2, b3])

# Use in constraints
model.Add(x > 0).OnlyEnforceIf(and_var)
```

### 4. Debugging and Analysis
```python
# Check model status
if model.debug_infeasible():
    print("Model is infeasible - checking constraints...")
    
# Export for later use
model.export_to_file('my_model.zip')

# Create a subset for focused debugging
subset = model.create_subset_copy(['main_constraint', 'secondary_constraint'])
```

### 5. Advanced Usage
```python
# Import existing model
loaded_model = EnhancedCpModel()
loaded_model.import_from_file('my_model.zip')

# Work with constraint metadata
for constraint_name, constraint_info in model._constraints.items():
    print(f"Constraint {constraint_name} has {len(constraint_info.user_enforcement_literals)} enforcement literals")
```

## Best Practices

1. **Always use named constraints** for better debugging: `model.Add(expr, name='constraint_name')`
2. **Enable tracking** for important variables: `model.NewIntVar(0, 10, 'x', track=True)`
3. **Use constructor tools** for complex boolean logic rather than manual construction
4. **Export models** before major changes for easy rollback
5. **Use the debug tools** early to catch infeasibility issues

This architecture provides powerful extensions to the standard CP-SAT model while maintaining backward compatibility with existing OR-Tools code.

## Global Symbol Map

- **`ConstraintInfo`** defined in: `model\constraints.py`
- **`EnhancedConstructorsMixin`** defined in: `model\constructors.py`
- **`EnhancedCpModel`** defined in: `model\model.py`
- **`M`** defined in: `model\constructor_tools.py`
- **`NewAndBoolVar`** defined in: `model\constructor_tools.py`
- **`NewAndSubjectToBools`** defined in: `model\constructor_tools.py`
- **`NewContainedInBoolVar`** defined in: `model\constructor_tools.py`
- **`NewEqualBoolVar`** defined in: `model\constructor_tools.py`
- **`NewGreaterBoolVar`** defined in: `model\constructor_tools.py`
- **`NewGreaterOrEqualBoolVar`** defined in: `model\constructor_tools.py`
- **`NewLessBoolVar`** defined in: `model\constructor_tools.py`
- **`NewLessOrEqualBoolVar`** defined in: `model\constructor_tools.py`
- **`NewMaxSubjectToBools`** defined in: `model\constructor_tools.py`
- **`NewMinSubjectToBools`** defined in: `model\constructor_tools.py`
- **`NewNotEqualBoolVar`** defined in: `model\constructor_tools.py`
- **`NewOrBoolVar`** defined in: `model\constructor_tools.py`
- **`NewOrSubjectToBools`** defined in: `model\constructor_tools.py`
- **`NewOverlapBoolVar`** defined in: `model\constructor_tools.py`
- **`NewPointInIntervalBoolVar`** defined in: `model\constructor_tools.py`
- **`ObjectiveInfo`** defined in: `model\objectives.py`
- **`VariableInfo`** defined in: `model\variables.py`
- **`_ConstraintProxy`** defined in: `model\constraints.py`
- **`_ConstraintsMixin`** defined in: `model\constraints.py`
- **`_DebugMixin`** defined in: `model\debug.py`
- **`_IOMixin`** defined in: `model\io.py`
- **`_ObjectivesMixin`** defined in: `model\objectives.py`
- **`_VariablesMixin`** defined in: `model\variables.py`

## Potential Issues

### Missing References

- `model.AddElement` used in `model\constraints.py` but not defined
- `model.AddAutomaton` used in `model\constraints.py` but not defined
- `self._map_expr_to_new_model` used in `model\constraints.py` but not defined
- `new_ct.OnlyEnforceIf` used in `model\constraints.py` but not defined
- `model.AddLinearExpressionInDomain` used in `model\constraints.py` but not defined
- `model.AddModuloEquality` used in `model\constraints.py` but not defined
- `constraint` used in `model\constraints.py` but not defined
- `model.AddBoolAnd` used in `model\constraints.py` but not defined
- `model.Add` used in `model\constraints.py` but not defined
- `model.AddAllowedAssignments` used in `model\constraints.py` but not defined
- `model.AddAllDifferent` used in `model\constraints.py` but not defined
- `_cp.CpModel` used in `model\constraints.py` but not defined
- `model.AddReservoirConstraint` used in `model\constraints.py` but not defined
- `model.AddNoOverlap2D` used in `model\constraints.py` but not defined
- `model.AddBoolOr` used in `model\constraints.py` but not defined
- `model` used in `model\constraints.py` but not defined
- `self._info.user_enforcement_literals` used in `model\constraints.py` but not defined
- `model.AddAbsEquality` used in `model\constraints.py` but not defined
- `self._ensure_constraints` used in `model\constraints.py` but not defined
- `model.AddDivisionEquality` used in `model\constraints.py` but not defined
- `self._ct` used in `model\constraints.py` but not defined
- `model.AddBoolXor` used in `model\constraints.py` but not defined
- `map_arg` used in `model\constraints.py` but not defined
- `self` used in `model\constraints.py` but not defined
- `constraint.OnlyEnforceIf` used in `model\constraints.py` but not defined
- `model.AddForbiddenAssignments` used in `model\constraints.py` but not defined
- `model.AddInverse` used in `model\constraints.py` but not defined
- `model.AddMaxEquality` used in `model\constraints.py` but not defined
- `new_ct` used in `model\constraints.py` but not defined
- `_cp.LinearExpr` used in `model\constraints.py` but not defined
- `self._ct.WithName` used in `model\constraints.py` but not defined
- `self._ct.OnlyEnforceIf` used in `model\constraints.py` but not defined
- `model.AddCircuit` used in `model\constraints.py` but not defined
- `model.AddNoOverlap` used in `model\constraints.py` but not defined
- `self._register_constraint` used in `model\constraints.py` but not defined
- `_cp.Domain` used in `model\constraints.py` but not defined
- `arg` used in `model\constraints.py` but not defined
- `arg.get` used in `model\constraints.py` but not defined
- `model.AddMultiplicationEquality` used in `model\constraints.py` but not defined
- `model.AddImplication` used in `model\constraints.py` but not defined
- `model.AddMultipleCircuit` used in `model\constraints.py` but not defined
- `constraint_type.lower` used in `model\constraints.py` but not defined
- `model.AddCumulative` used in `model\constraints.py` but not defined
- `constraint_type` used in `model\constraints.py` but not defined
- `model.AddLinearConstraint` used in `model\constraints.py` but not defined
- `self._info.user_enforcement_literals.extend` used in `model\constraints.py` but not defined
- `_cp.Domain.FromIntervals` used in `model\constraints.py` but not defined
- `_cp.LinearExpr.WeightedSum` used in `model\constraints.py` but not defined
- `model.AddMinEquality` used in `model\constraints.py` but not defined
- `model.AddBoolAnd.OnlyEnforceIf` used in `model\constructor_tools.py` but not defined
- `model.NewIntVar` used in `model\constructor_tools.py` but not defined
- `or_var` used in `model\constructor_tools.py` but not defined
- `interval2.StartExpr` used in `model\constructor_tools.py` but not defined
- `interval1.StartExpr` used in `model\constructor_tools.py` but not defined
- `interval.EndExpr` used in `model\constructor_tools.py` but not defined
- `equal_and_check_vars` used in `model\constructor_tools.py` but not defined
- `model.AddMultiplicationEquality` used in `model\constructor_tools.py` but not defined
- `l_var.Not` used in `model\constructor_tools.py` but not defined
- `interval.StartExpr` used in `model\constructor_tools.py` but not defined
- `interval1.EndExpr` used in `model\constructor_tools.py` but not defined
- `and_var` used in `model\constructor_tools.py` but not defined
- `equal_and_check_vars.append` used in `model\constructor_tools.py` but not defined
- `interval2` used in `model\constructor_tools.py` but not defined
- `var_lower_domains` used in `model\constructor_tools.py` but not defined
- `neq_var.Not` used in `model\constructor_tools.py` but not defined
- `var_upper_domains.append` used in `model\constructor_tools.py` but not defined
- `equal_check_vars` used in `model\constructor_tools.py` but not defined
- `interval1` used in `model\constructor_tools.py` but not defined
- `l_var` used in `model\constructor_tools.py` but not defined
- `interval2.EndExpr` used in `model\constructor_tools.py` but not defined
- `model.AddBoolOr` used in `model\constructor_tools.py` but not defined
- `model.AddMaxEquality` used in `model\constructor_tools.py` but not defined
- `var_lower_domains.append` used in `model\constructor_tools.py` but not defined
- `model` used in `model\constructor_tools.py` but not defined
- `model.AddBoolOr.OnlyEnforceIf` used in `model\constructor_tools.py` but not defined
- `var.Proto` used in `model\constructor_tools.py` but not defined
- `var_upper_domains` used in `model\constructor_tools.py` but not defined
- `leq_var.Not` used in `model\constructor_tools.py` but not defined
- `eq_var` used in `model\constructor_tools.py` but not defined
- `model.add_constraint_tags` used in `model\constructor_tools.py` but not defined
- `geq_var.Not` used in `model\constructor_tools.py` but not defined
- `g_var.Not` used in `model\constructor_tools.py` but not defined
- `eq_var.Not` used in `model\constructor_tools.py` but not defined
- `equal_check_vars.append` used in `model\constructor_tools.py` but not defined
- `and_var.Not` used in `model\constructor_tools.py` but not defined
- `leq_var` used in `model\constructor_tools.py` but not defined
- `model.NewBoolVar` used in `model\constructor_tools.py` but not defined
- `model.AddBoolAnd` used in `model\constructor_tools.py` but not defined
- `or_var.Not` used in `model\constructor_tools.py` but not defined
- `interval` used in `model\constructor_tools.py` but not defined
- `model.Add` used in `model\constructor_tools.py` but not defined
- `geq_var` used in `model\constructor_tools.py` but not defined
- `product_vars.append` used in `model\constructor_tools.py` but not defined
- `variable.Not` used in `model\constructor_tools.py` but not defined
- `var` used in `model\constructor_tools.py` but not defined
- `variable` used in `model\constructor_tools.py` but not defined
- `model.Add.OnlyEnforceIf` used in `model\constructor_tools.py` but not defined
- `g_var` used in `model\constructor_tools.py` but not defined
- `neq_var` used in `model\constructor_tools.py` but not defined
- `product_vars` used in `model\constructor_tools.py` but not defined
- `c_types` used in `model\debug.py` but not defined
- `self` used in `model\debug.py` but not defined
- `disable_vars.append` used in `model\debug.py` but not defined
- `subset.get_constraint_names` used in `model\debug.py` but not defined
- `solver` used in `model\debug.py` but not defined
- `subset.enable_constraints` used in `model\debug.py` but not defined
- `relaxed.get_enabled_constraints` used in `model\debug.py` but not defined
- `subset` used in `model\debug.py` but not defined
- `solver_params` used in `model\debug.py` but not defined
- `_cp.CpModel` used in `model\debug.py` but not defined
- `random.sample` used in `model\debug.py` but not defined
- `self.enable_constraint` used in `model\debug.py` but not defined
- `mis_model.Add` used in `model\debug.py` but not defined
- `v_types.get` used in `model\debug.py` but not defined
- `_cp.CpSolver` used in `model\debug.py` but not defined
- `solver_params.items` used in `model\debug.py` but not defined
- `self.clone` used in `model\debug.py` but not defined
- `v_types` used in `model\debug.py` but not defined
- `self.get_disabled_constraints` used in `model\debug.py` but not defined
- `self.get_enabled_objective` used in `model\debug.py` but not defined
- `warnings.append` used in `model\debug.py` but not defined
- `subset.disable_constraints` used in `model\debug.py` but not defined
- `warnings` used in `model\debug.py` but not defined
- `solver.ObjectiveValue` used in `model\debug.py` but not defined
- `saved_states.items` used in `model\debug.py` but not defined
- `self._ensure_constraints.values` used in `model\debug.py` but not defined
- `relaxed` used in `model\debug.py` but not defined
- `saved_states` used in `model\debug.py` but not defined
- `solver.Value` used in `model\debug.py` but not defined
- `mis_model.NewBoolVar` used in `model\debug.py` but not defined
- `self._constraints.items` used in `model\debug.py` but not defined
- `c_types.get` used in `model\debug.py` but not defined
- `mis_model.Minimize` used in `model\debug.py` but not defined
- `self._constraints` used in `model\debug.py` but not defined
- `self._ensure_constraints` used in `model\debug.py` but not defined
- `self.get_enabled_constraints` used in `model\debug.py` but not defined
- `solver.Solve` used in `model\debug.py` but not defined
- `self.disable_constraint` used in `model\debug.py` but not defined
- `disable_vars` used in `model\debug.py` but not defined
- `name_to_disable` used in `model\debug.py` but not defined
- `mis_model` used in `model\debug.py` but not defined
- `relaxed.disable_constraints` used in `model\debug.py` but not defined
- `self._ensure_constraints.keys` used in `model\debug.py` but not defined
- `self._ensure_constraints.items` used in `model\debug.py` but not defined
- `name_to_disable.items` used in `model\debug.py` but not defined
- `self._solve` used in `model\debug.py` but not defined
- `solving_model.NewBoolVar` used in `model\io.py` but not defined
- `zf` used in `model\io.py` but not defined
- `self._objectives` used in `model\io.py` but not defined
- `self._map_expr_to_new_model` used in `model\io.py` but not defined
- `proto.ParseFromString` used in `model\io.py` but not defined
- `self._variables` used in `model\io.py` but not defined
- `solving_model.NewOptionalIntervalVar` used in `model\io.py` but not defined
- `name.startswith` used in `model\io.py` but not defined
- `temp_model.Proto.SerializeToString` used in `model\io.py` but not defined
- `arg.FlattenedIntervals` used in `model\io.py` but not defined
- `json.loads` used in `model\io.py` but not defined
- `self.Proto` used in `model\io.py` but not defined
- `self._create_solving_model` used in `model\io.py` but not defined
- `arg.GetVars` used in `model\io.py` but not defined
- `self._variables.items` used in `model\io.py` but not defined
- `_cp.CpModel` used in `model\io.py` but not defined
- `v.Name` used in `model\io.py` but not defined
- `temp_model` used in `model\io.py` but not defined
- `arg.Name` used in `model\io.py` but not defined
- `temp_model.Proto` used in `model\io.py` but not defined
- `zf.writestr` used in `model\io.py` but not defined
- `serialized_arg` used in `model\io.py` but not defined
- `v` used in `model\io.py` but not defined
- `self.Proto.CopyFrom` used in `model\io.py` but not defined
- `arg.GetVar.Name` used in `model\io.py` but not defined
- `name` used in `model\io.py` but not defined
- `solving_model` used in `model\io.py` but not defined
- `self._deserialize_arg` used in `model\io.py` but not defined
- `cmeta` used in `model\io.py` but not defined
- `proto` used in `model\io.py` but not defined
- `cmeta.get` used in `model\io.py` but not defined
- `self` used in `model\io.py` but not defined
- `recreate` used in `model\io.py` but not defined
- `serialized_arg.get` used in `model\io.py` but not defined
- `arg.GetVar` used in `model\io.py` but not defined
- `solving_model.Minimize` used in `model\io.py` but not defined
- `_cp.LinearExpr` used in `model\io.py` but not defined
- `solving_model.NewConstant` used in `model\io.py` but not defined
- `objectives_meta` used in `model\io.py` but not defined
- `self._serialize_arg` used in `model\io.py` but not defined
- `Not` used in `model\io.py` but not defined
- `json.dumps` used in `model\io.py` but not defined
- `zf.read` used in `model\io.py` but not defined
- `solving_model.NewIntVar` used in `model\io.py` but not defined
- `arg` used in `model\io.py` but not defined
- `objectives_meta.append` used in `model\io.py` but not defined
- `solving_model.Maximize` used in `model\io.py` but not defined
- `zf.read.decode` used in `model\io.py` but not defined
- `items` used in `model\io.py` but not defined
- `arg.IsConstant` used in `model\io.py` but not defined
- `_cp.LinearExpr.WeightedSum` used in `model\io.py` but not defined
- `self._clear_model` used in `model\io.py` but not defined
- `cp_model_pb2.CpModelProto` used in `model\io.py` but not defined
- `self._objectives.append` used in `model\io.py` but not defined
- `zipfile.ZipFile` used in `model\io.py` but not defined
- `solving_model.NewIntervalVar` used in `model\io.py` but not defined
- `self._objectives` used in `model\model.py` but not defined
- `self.__dict__` used in `model\model.py` but not defined
- `self.__dict__.update` used in `model\model.py` but not defined
- `self._constraints` used in `model\model.py` but not defined
- `self._variables` used in `model\model.py` but not defined
- `self._constraints.clear` used in `model\model.py` but not defined
- `self._variables.clear` used in `model\model.py` but not defined
- `_cp.CpModel` used in `model\model.py` but not defined
- `self._objectives.clear` used in `model\model.py` but not defined
- `self._ensure_objectives.append` used in `model\objectives.py` but not defined
- `self._ensure_objectives` used in `model\objectives.py` but not defined
- `self` used in `model\objectives.py` but not defined
- `objective_type.lower` used in `model\objectives.py` but not defined
- `objective_type` used in `model\objectives.py` but not defined
- `_cp.CpModel` used in `model\objectives.py` but not defined
- `self` used in `model\variables.py` but not defined
- `expr.IsConstant` used in `model\variables.py` but not defined
- `_cp.CpModel` used in `model\variables.py` but not defined
- `self._map_expr_to_new_model` used in `model\variables.py` but not defined
- `inner.Name` used in `model\variables.py` but not defined
- `_cp.LinearExpr` used in `model\variables.py` but not defined
- `expr` used in `model\variables.py` but not defined
- `Not` used in `model\variables.py` but not defined
- `expr.GetVars` used in `model\variables.py` but not defined
- `self._ensure_variables` used in `model\variables.py` but not defined
- `inner` used in `model\variables.py` but not defined
- `self._deep_map_expr` used in `model\variables.py` but not defined
- `_cp.LinearExpr.WeightedSum` used in `model\variables.py` but not defined
- `expr.GetVar` used in `model\variables.py` but not defined
- `expr.Name` used in `model\variables.py` but not defined

## Most Called Functions/Methods

- **`super`** (43 calls)
  - Called from `model\constraints.py` line 81 (class:_ConstraintsMixin::function:Add)
  - Called from `model\constraints.py` line 95 (class:_ConstraintsMixin::function:AddLinearConstraint)
  - Called from `model\constraints.py` line 108 (class:_ConstraintsMixin::function:AddLinearExpressionInDomain)
- **`map_arg`** (43 calls)
  - Called from `model\constraints.py` line 473 (class:_ConstraintsMixin::function:_recreate_constraint_in_model::function:map_arg)
  - Called from `model\constraints.py` line 481 (class:_ConstraintsMixin::function:_recreate_constraint_in_model)
  - Called from `model\constraints.py` line 484 (class:_ConstraintsMixin::function:_recreate_constraint_in_model)
- **`model.add_constraint_tags`** (41 calls)
  - Called from `model\constructor_tools.py` line 50 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 51 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 95 (function:NewLessOrEqualBoolVar)
- **`isinstance`** (34 calls)
  - Called from `model\constraints.py` line 56 (class:_ConstraintProxy::function:OnlyEnforceIf)
  - Called from `model\constraints.py` line 464 (class:_ConstraintsMixin::function:_recreate_constraint_in_model::function:map_arg)
  - Called from `model\constraints.py` line 466 (class:_ConstraintsMixin::function:_recreate_constraint_in_model::function:map_arg)
- **`tuple`** (29 calls)
  - Called from `model\constraints.py` line 121 (class:_ConstraintsMixin::function:AddAllDifferent)
  - Called from `model\constraints.py` line 135 (class:_ConstraintsMixin::function:AddElement)
  - Called from `model\constraints.py` line 147 (class:_ConstraintsMixin::function:AddCircuit)
- **`self._register_constraint`** (25 calls)
  - Called from `model\constraints.py` line 80 (class:_ConstraintsMixin::function:Add)
  - Called from `model\constraints.py` line 94 (class:_ConstraintsMixin::function:AddLinearConstraint)
  - Called from `model\constraints.py` line 107 (class:_ConstraintsMixin::function:AddLinearExpressionInDomain)
- **`len`** (23 calls)
  - Called from `model\constructor_tools.py` line 563 (function:NewMinSubjectToBools)
  - Called from `model\constructor_tools.py` line 563 (function:NewMinSubjectToBools)
  - Called from `model\constructor_tools.py` line 638 (function:NewMaxSubjectToBools)
- **`getattr`** (20 calls)
  - Called from `model\constraints.py` line 68 (class:_ConstraintProxy::function:__getattr__)
  - Called from `model\debug.py` line 133 (class:_DebugMixin::function:summary)
  - Called from `model\debug.py` line 145 (class:_DebugMixin::function:summary)
- **`model.Add`** (19 calls)
  - Called from `model\constraints.py` line 481 (class:_ConstraintsMixin::function:_recreate_constraint_in_model)
  - Called from `model\constructor_tools.py` line 47 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 48 (function:NewGreaterOrEqualBoolVar)
- **`model.Add.OnlyEnforceIf`** (16 calls)
  - Called from `model\constructor_tools.py` line 47 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 48 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 92 (function:NewLessOrEqualBoolVar)
- **`self._ensure_constraints`** (12 calls)
  - Called from `model\constraints.py` line 427 (class:_ConstraintsMixin::function:_register_constraint)
  - Called from `model\constraints.py` line 442 (class:_ConstraintsMixin::function:_register_constraint)
  - Called from `model\debug.py` line 129 (class:_DebugMixin::function:summary)
- **`hasattr`** (11 calls)
  - Called from `model\constraints.py` line 569 (class:_ConstraintsMixin::function:_ensure_constraints)
  - Called from `model\debug.py` line 45 (class:_DebugMixin::function:debug_infeasible)
  - Called from `model\debug.py` line 245 (class:_DebugMixin::function:_ensure_constraints)
- **`ValueError`** (11 calls)
  - Called from `model\constraints.py` line 428 (class:_ConstraintsMixin::function:_register_constraint)
  - Called from `model\constraints.py` line 557 (class:_ConstraintsMixin::function:_recreate_constraint_in_model)
  - Called from `model\debug.py` line 46 (class:_DebugMixin::function:debug_infeasible)
- **`model.NewBoolVar`** (11 calls)
  - Called from `model\constructor_tools.py` line 43 (function:NewGreaterOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 88 (function:NewLessOrEqualBoolVar)
  - Called from `model\constructor_tools.py` line 133 (function:NewGreaterBoolVar)
- **`self._map_expr_to_new_model`** (10 calls)
  - Called from `model\constraints.py` line 476 (class:_ConstraintsMixin::function:_recreate_constraint_in_model::function:map_arg)
  - Called from `model\io.py` line 264 (class:_IOMixin::function:_create_solving_model)
  - Called from `model\io.py` line 265 (class:_IOMixin::function:_create_solving_model)

## Detailed Module Documentation

### `model\constraints.py`

## Module Purpose
This module provides enhanced constraint management for Google OR-Tools CP-SAT by wrapping constraints with metadata for debugging, enable/disable functionality, and constraint recreation. It extends the base CpModel with tracking capabilities while maintaining full API compatibility.

## Public API

### ConstraintInfo Class
**Purpose**: Wrapper around a single constraint with metadata for debugging/enable/disable functionality.

**Usage pattern**:
```python
# Typically created internally by _ConstraintsMixin
from model.constraints import ConstraintInfo

# Not typically instantiated directly by users
```

**Method signatures**:
```python
def __init__(
    self,
    name: str,
    original_args: Any,
    constraint_type: str,
    ortools_ct: _cp.Constraint,
    enable_var: _cp.IntVar,
) -> None:
```

**Attributes**:
- `name`: Constraint identifier
- `original_args`: Original arguments used to create the constraint
- `constraint_type`: Type of constraint (e.g., "Linear", "AllDifferent")
- `ortools_ct`: The underlying OR-Tools constraint object
- `enable_var`: Boolean variable controlling constraint enforcement
- `user_enforcement_literals`: List of literals for conditional enforcement

### _ConstraintProxy Class
**Purpose**: Proxy returned by Add* methods to capture user calls like OnlyEnforceIf and WithName.

**Usage pattern**:
```python
# Returned by constraint creation methods
constraint = model.Add(x + y == 5)
constraint.OnlyEnforceIf([a, b])  # Returns self for chaining
constraint.WithName("my_constraint")
```

**Method signatures**:
```python
def OnlyEnforceIf(self, lits) -> "_ConstraintProxy":
    """Conditionally enforce constraint based on literals."""

def WithName(self, name: str) -> "_ConstraintProxy":
    """Set constraint name."""

def __getattr__(self, attr: str) -> Any:
    """Delegate unknown attributes to underlying constraint."""
```

### _ConstraintsMixin Class
**Purpose**: Mixin that extends CpModel with all constraint-creation helpers and metadata tracking.

**Usage pattern**:
```python
from model.constraints import _ConstraintsMixin
from ortools.sat.python import cp_model

class EnhancedModel(_ConstraintsMixin, cp_model.CpModel):
    pass

model = EnhancedModel()
```

**Key constraint creation methods** (all return `_ConstraintProxy`):
```python
# Linear constraints
Add(ct, name: Optional[str] = None)
AddLinearConstraint(linear_expr: _cp.LinearExprT, lb: int, ub: int, name: Optional[str] = None)
AddLinearExpressionInDomain(linear_expr: _cp.LinearExprT, domain: _cp.Domain, name: Optional[str] = None)

# Global constraints
AddAllDifferent(variables: Sequence[_cp.LinearExprT], name: Optional[str] = None)
AddElement(index: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], target: _cp.LinearExprT, name: Optional[str] = None)
AddCircuit(arcs: Sequence[Tuple[int, int, _cp.LiteralT]], name: Optional[str] = None)
AddMultipleCircuit(arcs: Sequence[Tuple[int, int, _cp.LiteralT]], name: Optional[str] = None)

# Table constraints
AddAllowedAssignments(variables: Sequence[_cp.IntVar], tuples_list: Sequence[Sequence[int]], name: Optional[str] = None)
AddForbiddenAssignments(variables: Sequence[_cp.IntVar], tuples_list: Sequence[Sequence[int]], name: Optional[str] = None)

# Automaton and inverse constraints
AddAutomaton(transition_variables: Sequence[_cp.IntVar], starting_state: int, 
             final_states: Sequence[int], transition_triples: Sequence[Tuple[int, int, int]], 
             name: Optional[str] = None)
AddInverse(variables: Sequence[_cp.IntVar], inverse_variables: Sequence[_cp.IntVar], name: Optional[str] = None)

# Reservoir constraint
AddReservoirConstraint(times: Sequence[_cp.LinearExprT], level_changes: Sequence[_cp.LinearExprT], 
                       min_level: int, max_level: int, name: Optional[str] = None)

# Expression constraints
AddMinEquality(target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None)
AddMaxEquality(target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None)
AddMultiplicationEquality(target: _cp.LinearExprT, variables: Sequence[_cp.LinearExprT], name: Optional[str] = None)
AddDivisionEquality(target: _cp.LinearExprT, numerator: _cp.LinearExprT, denominator: _cp.LinearExprT, name: Optional[str] = None)
AddAbsEquality(target: _cp.LinearExprT, variable: _cp.LinearExprT, name: Optional[str] = None)
AddModuloEquality(target: _cp.LinearExprT, variable: _cp.LinearExprT, modulo: _cp.LinearExprT, name: Optional[str] = None)

# Boolean constraints
AddBoolOr(literals: Sequence[_cp.LiteralT], name: Optional[str] = None)
AddBoolAnd(literals: Sequence[_cp.LiteralT], name: Optional[str] = None)
AddBoolXor(literals: Sequence[_cp.LiteralT], name: Optional[str] = None)
AddImplication(a: _cp.LiteralT, b: _cp.LiteralT, name: Optional[str] = None)

# Interval constraints
AddNoOverlap(intervals: Sequence[_cp.IntervalVar], name: Optional[str] = None)
AddNoOverlap2D(x_intervals: Sequence[_cp.IntervalVar], y_intervals: Sequence[_cp.IntervalVar], name: Optional[str] = None)
AddCumulative(intervals: Sequence[_cp.IntervalVar], demands: Sequence[_cp.LinearExprT], 
              capacity: _cp.LinearExprT, name: Optional[str] = None)
```

**Internal methods**:
```python
def _register_constraint(self, constraint: _cp.Constraint, original_args: Any, 
                        constraint_type: str, name: Optional[str], 
                        enforce_enable_var: bool = False) -> _cp.Constraint:
    """Register a constraint with full metadata."""

def _recreate_constraint_in_model(self, model: _cp.CpModel, constraint_info: ConstraintInfo, 
                                 var_mapping: Dict[str, Any]) -> None:
    """Recreate a constraint in another model using variable mapping."""

def _ensure_constraints(self) -> dict[str, ConstraintInfo]:
    """Lazy-initialize constraint registry."""
```

## Cross-References

### Imports from other modules
- `ortools.sat.python.cp_model` (as `_cp`): Core OR-Tools CP-SAT functionality
- `typing` types: `Any`, `List`, `Optional`, `Sequence`, `Tuple`, `Dict`

### Exports to other modules
- `ConstraintInfo`: For constraint metadata tracking
- `_ConstraintProxy`: For constraint method chaining
- `_ConstraintsMixin`: For extending CpModel with enhanced constraint capabilities

### Potential issues
- The module assumes `_cp` refers to `ortools.sat.python.cp_model` but this isn't explicitly shown in the imports
- Circular dependency risk if other modules depend heavily on constraint metadata
- No explicit error handling for missing `_constraints` attribute in mixin usage

## Integration Examples

### Basic usage with enhanced model
```python
from model.constraints import _ConstraintsMixin
from ortools.sat.python import cp_model

class TrackingModel(_ConstraintsMixin, cp_model.CpModel):
    pass

# Create model with constraint tracking
model = TrackingModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')

# Create constraint with metadata tracking
constraint = model.Add(x + y == 10)
constraint.OnlyEnforceIf([model.NewBoolVar('enable')])
constraint.WithName("sum_constraint")

# Access constraint metadata
constraint_info = model._constraints["sum_constraint"]
print(f"Constraint type: {constraint_info.constraint_type}")
```

### Constraint recreation for model copying
```python
# Copy constraints to a new model
new_model = TrackingModel()
var_mapping = {'x': new_x, 'y': new_y}  # Map old vars to new vars

for constraint_name, info in model._constraints.items():
    model._recreate_constraint_in_model(new_model, info, var_mapping)
```

### Debugging and analysis
```python
# Analyze all constraints in the model
for name, info in model._constraints.items():
    print(f"{name}: {info.constraint_type}")
    print(f"  Enabled by: {info.enable_var}")
    print(f"  User conditions: {info.user_enforcement_literals}")
```

The module provides a powerful foundation for constraint management, debugging, and model transformation while maintaining full compatibility with the standard OR-Tools API.

#### Complete API Reference

##### Class: `ConstraintInfo`

**Description:** Wrapper around a single constraint with metadata for debugging/enable/disable....

**Methods:**

- **`__init__`**(self, name: <_ast.Name object at 0x000001D6B13E09A0>, original_args: <_ast.Name object at 0x000001D6B13E0A00>, constraint_type: <_ast.Name object at 0x000001D6B13E0A60>, ortools_ct: <_ast.Attribute object at 0x000001D6B13E0AC0>, enable_var: <_ast.Attribute object at 0x000001D6B13E0B50>) -> <_ast.Constant object at 0x000001D6B11F6B80>
  - **Calls:** `set`

**Usage Pattern:**
```python
from model\constraints import ConstraintInfo
instance = ConstraintInfo(name, original_args, constraint_type, ortools_ct, enable_var)
```

##### Class: `_ConstraintProxy`

**Description:** Proxy returned by Add*/etc. to capture user calls like OnlyEnforceIf....

**Methods:**

- **`__init__`**(self, ct: <_ast.Attribute object at 0x000001D6B11F6D60>, info: <_ast.Name object at 0x000001D6B11F6DF0>) -> <_ast.Constant object at 0x000001D6B11F6F70>

- **`OnlyEnforceIf`**(self, lits) -> <_ast.Constant object at 0x000001D6B135C430>
  - **Calls:** `self._info.user_enforcement_literals.extend`, `self._ct.OnlyEnforceIf`, `isinstance`

- **`WithName`**(self, name: <_ast.Name object at 0x000001D6B135C4F0>) -> <_ast.Constant object at 0x000001D6B135C7F0>
  - **Calls:** `self._ct.WithName`

- **`__getattr__`**(self, attr: <_ast.Name object at 0x000001D6B135C8B0>) -> <_ast.Name object at 0x000001D6B135CAC0>
  - **Calls:** `getattr`

**Usage Pattern:**
```python
from model\constraints import _ConstraintProxy
instance = _ConstraintProxy(ct, info)
```

##### Class: `_ConstraintsMixin`

**Description:** All constraint-creation helpers....

**Inherits from:** `_cp.CpModel`

**Methods:**

- **`Add`**(self, ct, name: <_ast.Subscript object at 0x000001D6B135CCD0> = <_ast.Constant object at 0x000001D6B135CDF0>) -> <_ast.Attribute object at 0x000001D6B1215250>
  - Add a generic constraint....
  - **Calls:** `self._register_constraint`, `super.Add`, `super`

- **`AddLinearConstraint`**(self, linear_expr: <_ast.Attribute object at 0x000001D6B12153A0>, lb: <_ast.Name object at 0x000001D6B1215430>, ub: <_ast.Name object at 0x000001D6B12154C0>, name: <_ast.Subscript object at 0x000001D6B1215550> = <_ast.Constant object at 0x000001D6B1215640>) -> <_ast.Attribute object at 0x000001D6B1215B50>
  - **Calls:** `self._register_constraint`, `super.AddLinearConstraint`, `super`

- **`AddLinearExpressionInDomain`**(self, linear_expr: <_ast.Attribute object at 0x000001D6B1215C70>, domain: <_ast.Attribute object at 0x000001D6B1215D00>, name: <_ast.Subscript object at 0x000001D6B1215D90> = <_ast.Constant object at 0x000001D6B1215E50>) -> <_ast.Attribute object at 0x000001D6B120F370>
  - **Calls:** `self._register_constraint`, `super.AddLinearExpressionInDomain`, `super`

- **`AddAllDifferent`**(self, variables: <_ast.Subscript object at 0x000001D6B120F190>, name: <_ast.Subscript object at 0x000001D6B120F5B0> = <_ast.Constant object at 0x000001D6B120F6A0>) -> <_ast.Attribute object at 0x000001D6B120FB20>
  - **Calls:** `self._register_constraint`, `super.AddAllDifferent`, `tuple`, `super`

- **`AddElement`**(self, index: <_ast.Attribute object at 0x000001D6B120FC70>, variables: <_ast.Subscript object at 0x000001D6B120FD30>, target: <_ast.Attribute object at 0x000001D6B120FE50>, name: <_ast.Subscript object at 0x000001D6B120FEE0> = <_ast.Constant object at 0x000001D6B120FFD0>) -> <_ast.Attribute object at 0x000001D6B121B550>
  - **Calls:** `self._register_constraint`, `super.AddElement`, `tuple`, `super`

- **`AddCircuit`**(self, arcs: <_ast.Subscript object at 0x000001D6B121B670>, name: <_ast.Subscript object at 0x000001D6B121B910> = <_ast.Constant object at 0x000001D6B121BA00>) -> <_ast.Attribute object at 0x000001D6B121BEE0>
  - **Calls:** `self._register_constraint`, `super.AddCircuit`, `tuple`, `super`

- **`AddMultipleCircuit`**(self, arcs: <_ast.Subscript object at 0x000001D6B121BFD0>, name: <_ast.Subscript object at 0x000001D6B121D280> = <_ast.Constant object at 0x000001D6B121D370>) -> <_ast.Attribute object at 0x000001D6B121D7F0>
  - **Calls:** `self._register_constraint`, `super.AddMultipleCircuit`, `tuple`, `super`

- **`AddAllowedAssignments`**(self, variables: <_ast.Subscript object at 0x000001D6B121D910>, tuples_list: <_ast.Subscript object at 0x000001D6B121DA60>, name: <_ast.Subscript object at 0x000001D6B121DC40> = <_ast.Constant object at 0x000001D6B121DD30>) -> <_ast.Attribute object at 0x000001D6B121F430>
  - **Calls:** `self._register_constraint`, `super.AddAllowedAssignments`, `tuple`, `tuple`, `super` (+1 more)

- **`AddForbiddenAssignments`**(self, variables: <_ast.Subscript object at 0x000001D6B121F580>, tuples_list: <_ast.Subscript object at 0x000001D6B121F6D0>, name: <_ast.Subscript object at 0x000001D6B121F8B0> = <_ast.Constant object at 0x000001D6B121F9A0>) -> <_ast.Attribute object at 0x000001D6B1221070>
  - **Calls:** `self._register_constraint`, `super.AddForbiddenAssignments`, `tuple`, `tuple`, `super` (+1 more)

- **`AddAutomaton`**(self, transition_variables: <_ast.Subscript object at 0x000001D6B12211C0>, starting_state: <_ast.Name object at 0x000001D6B1221340>, final_states: <_ast.Subscript object at 0x000001D6B12213A0>, transition_triples: <_ast.Subscript object at 0x000001D6B12214C0>, name: <_ast.Subscript object at 0x000001D6B1221700> = <_ast.Constant object at 0x000001D6B12217F0>) -> <_ast.Attribute object at 0x000001D6B1221EB0>
  - **Calls:** `self._register_constraint`, `super.AddAutomaton`, `tuple`, `tuple`, `tuple` (+1 more)

- **`AddInverse`**(self, variables: <_ast.Subscript object at 0x000001D6B1221FD0>, inverse_variables: <_ast.Subscript object at 0x000001D6B1218130>, name: <_ast.Subscript object at 0x000001D6B1218280> = <_ast.Constant object at 0x000001D6B12183A0>) -> <_ast.Attribute object at 0x000001D6B1218940>
  - **Calls:** `self._register_constraint`, `super.AddInverse`, `tuple`, `tuple`, `super`

- **`AddReservoirConstraint`**(self, times: <_ast.Subscript object at 0x000001D6B1218A60>, level_changes: <_ast.Subscript object at 0x000001D6B1218BB0>, min_level: <_ast.Name object at 0x000001D6B1218CD0>, max_level: <_ast.Name object at 0x000001D6B1218D30>, name: <_ast.Subscript object at 0x000001D6B1218D90> = <_ast.Constant object at 0x000001D6B1218E80>) -> <_ast.Attribute object at 0x000001D6B1227520>
  - **Calls:** `self._register_constraint`, `super.AddReservoirConstraint`, `tuple`, `tuple`, `super`

- **`AddMinEquality`**(self, target: <_ast.Attribute object at 0x000001D6B1227640>, variables: <_ast.Subscript object at 0x000001D6B1227700>, name: <_ast.Subscript object at 0x000001D6B1227880> = <_ast.Constant object at 0x000001D6B1227970>) -> <_ast.Attribute object at 0x000001D6B1227E50>
  - **Calls:** `self._register_constraint`, `super.AddMinEquality`, `tuple`, `super`

- **`AddMaxEquality`**(self, target: <_ast.Attribute object at 0x000001D6B1227F70>, variables: <_ast.Subscript object at 0x000001D6B12276D0>, name: <_ast.Subscript object at 0x000001D6B122C130> = <_ast.Constant object at 0x000001D6B122C220>) -> <_ast.Attribute object at 0x000001D6B122C730>
  - **Calls:** `self._register_constraint`, `super.AddMaxEquality`, `tuple`, `super`

- **`AddMultiplicationEquality`**(self, target: <_ast.Attribute object at 0x000001D6B122C850>, variables: <_ast.Subscript object at 0x000001D6B122C8E0>, name: <_ast.Subscript object at 0x000001D6B122CA30> = <_ast.Constant object at 0x000001D6B122CB20>) -> <_ast.Attribute object at 0x000001D6B120D070>
  - **Calls:** `self._register_constraint`, `super.AddMultiplicationEquality`, `tuple`, `super`

- **`AddDivisionEquality`**(self, target: <_ast.Attribute object at 0x000001D6B120D1C0>, numerator: <_ast.Attribute object at 0x000001D6B120D250>, denominator: <_ast.Attribute object at 0x000001D6B120D2E0>, name: <_ast.Subscript object at 0x000001D6B120D370> = <_ast.Constant object at 0x000001D6B120D460>) -> <_ast.Attribute object at 0x000001D6B120D970>
  - **Calls:** `self._register_constraint`, `super.AddDivisionEquality`, `super`

- **`AddAbsEquality`**(self, target: <_ast.Attribute object at 0x000001D6B120DA90>, variable: <_ast.Attribute object at 0x000001D6B120DB20>, name: <_ast.Subscript object at 0x000001D6B120DBE0> = <_ast.Constant object at 0x000001D6B120DCD0>) -> <_ast.Attribute object at 0x000001D6B1205190>
  - **Calls:** `self._register_constraint`, `super.AddAbsEquality`, `super`

- **`AddModuloEquality`**(self, target: <_ast.Attribute object at 0x000001D6B12052E0>, variable: <_ast.Attribute object at 0x000001D6B1205370>, modulo: <_ast.Attribute object at 0x000001D6B1205400>, name: <_ast.Subscript object at 0x000001D6B1205490> = <_ast.Constant object at 0x000001D6B1205580>) -> <_ast.Attribute object at 0x000001D6B1205A90>
  - **Calls:** `self._register_constraint`, `super.AddModuloEquality`, `super`

- **`AddBoolOr`**(self, literals: <_ast.Subscript object at 0x000001D6B1205BB0>, name: <_ast.Subscript object at 0x000001D6B1205D00> = <_ast.Constant object at 0x000001D6B1205DF0>) -> <_ast.Attribute object at 0x000001D6B15CE2B0>
  - **Calls:** `self._register_constraint`, `super.AddBoolOr`, `tuple`, `super`

- **`AddBoolAnd`**(self, literals: <_ast.Subscript object at 0x000001D6B15CE3D0>, name: <_ast.Subscript object at 0x000001D6B15CE520> = <_ast.Constant object at 0x000001D6B15CE610>) -> <_ast.Attribute object at 0x000001D6B15CEAC0>
  - **Calls:** `self._register_constraint`, `super.AddBoolAnd`, `tuple`, `super`

- **`AddBoolXor`**(self, literals: <_ast.Subscript object at 0x000001D6B15CEBE0>, name: <_ast.Subscript object at 0x000001D6B15CED30> = <_ast.Constant object at 0x000001D6B15CEE80>) -> <_ast.Attribute object at 0x000001D6B15E22B0>
  - **Calls:** `self._register_constraint`, `super.AddBoolXor`, `tuple`, `super`

- **`AddImplication`**(self, a: <_ast.Attribute object at 0x000001D6B15E23A0>, b: <_ast.Attribute object at 0x000001D6B15E2460>, name: <_ast.Subscript object at 0x000001D6B15E24F0> = <_ast.Constant object at 0x000001D6B15E25E0>) -> <_ast.Attribute object at 0x000001D6B15E2A90>
  - **Calls:** `self._register_constraint`, `super.AddImplication`, `super`

- **`AddNoOverlap`**(self, intervals: <_ast.Subscript object at 0x000001D6B15E2BB0>, name: <_ast.Subscript object at 0x000001D6B15E2D00> = <_ast.Constant object at 0x000001D6B15E2DF0>) -> <_ast.Attribute object at 0x000001D6B15E82B0>
  - **Calls:** `self._register_constraint`, `super.AddNoOverlap`, `tuple`, `super`

- **`AddNoOverlap2D`**(self, x_intervals: <_ast.Subscript object at 0x000001D6B15E83D0>, y_intervals: <_ast.Subscript object at 0x000001D6B15E8520>, name: <_ast.Subscript object at 0x000001D6B15E8670> = <_ast.Constant object at 0x000001D6B15E8790>) -> <_ast.Attribute object at 0x000001D6B15E8CD0>
  - **Calls:** `self._register_constraint`, `super.AddNoOverlap2D`, `tuple`, `tuple`, `super`

- **`AddCumulative`**(self, intervals: <_ast.Subscript object at 0x000001D6B15E8E20>, demands: <_ast.Subscript object at 0x000001D6B15E8F70>, capacity: <_ast.Attribute object at 0x000001D6B15EB100>, name: <_ast.Subscript object at 0x000001D6B15EB190> = <_ast.Constant object at 0x000001D6B15EB280>) -> <_ast.Attribute object at 0x000001D6B15EB850>
  - **Calls:** `self._register_constraint`, `super.AddCumulative`, `tuple`, `tuple`, `super`

- **`_register_constraint`**(self, constraint: <_ast.Attribute object at 0x000001D6B15EB9A0>, original_args: <_ast.Name object at 0x000001D6B15EBA30>, constraint_type: <_ast.Name object at 0x000001D6B15EBA90>, name: <_ast.Subscript object at 0x000001D6B15EBAF0>, enforce_enable_var: <_ast.Name object at 0x000001D6B15EBC10> = <_ast.Constant object at 0x000001D6B15EBC40>) -> <_ast.Attribute object at 0x000001D6B15B2E20>
  - Register a constraint with full metadata....
  - **Calls:** `super.NewBoolVar`, `ConstraintInfo`, `_ConstraintProxy`, `constraint.OnlyEnforceIf`, `self._ensure_constraints` (+4 more)

- **`_recreate_constraint_in_model`**(self, model: <_ast.Attribute object at 0x000001D6B15B2F10>, constraint_info: <_ast.Name object at 0x000001D6B15B2FA0>, var_mapping: <_ast.Subscript object at 0x000001D6B15B8040>) -> <_ast.Constant object at 0x000001D6B12911F0>
  - Recreate a constraint in *model* using *var_mapping*.
Replays user-enforcement literals but **not** enable_var enforcement
(that is handled externally...
  - **Calls:** `isinstance`, `isinstance`, `self._map_expr_to_new_model`, `model.Add`, `new_ct.OnlyEnforceIf` (+78 more)

- **`_ensure_constraints`**(self) -> <_ast.Subscript object at 0x000001D6B1291730>
  - Lazy-initialise the constraint registry for mix-in safety....
  - **Calls:** `hasattr`

**Usage Pattern:**
```python
from model\constraints import _ConstraintsMixin
# Inherits from: _cp.CpModel
instance = _ConstraintsMixin()
```

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `from typing import Any`
- `from typing import List`
- `from typing import Optional`
- `from typing import Sequence`
- `from typing import Tuple`
- `from typing import Dict`

**Third Party:**
- `from __future__ import annotations`
- `ortools.sat.python.cp_model as _cp`

---

### `model\constructor_tools.py`

## Module Purpose
This module provides a comprehensive set of utility functions for creating boolean and integer variables in CP-SAT models that represent complex logical conditions and comparisons. It serves as a constructor toolkit for building sophisticated constraint relationships in optimization problems using Google's OR-Tools CP-SAT solver.

## Public API

### Comparison Boolean Variables

#### NewGreaterOrEqualBoolVar
```python
from model.constructor_tools import NewGreaterOrEqualBoolVar

bool_var = NewGreaterOrEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    threshold: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable >= threshold`, 0 otherwise.

#### NewLessOrEqualBoolVar
```python
from model.constructor_tools import NewLessOrEqualBoolVar

bool_var = NewLessOrEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    threshold: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable <= threshold`, 0 otherwise.

#### NewGreaterBoolVar
```python
from model.constructor_tools import NewGreaterBoolVar

bool_var = NewGreaterBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    threshold: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable > threshold`, 0 otherwise.

#### NewLessBoolVar
```python
from model.constructor_tools import NewLessBoolVar

bool_var = NewLessBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    threshold: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable < threshold`, 0 otherwise.

#### NewEqualBoolVar
```python
from model.constructor_tools import NewEqualBoolVar

bool_var = NewEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    value: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable == value`, 0 otherwise.

#### NewNotEqualBoolVar
```python
from model.constructor_tools import NewNotEqualBoolVar

bool_var = NewNotEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar,
    value: Union[_cp.IntVar, int],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when `variable != value`, 0 otherwise.

### Logical Operation Variables

#### NewAndBoolVar
```python
from model.constructor_tools import NewAndBoolVar

bool_var = NewAndBoolVar(
    model: EnhancedCpModel,
    variables: List[_cp.IntVar],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when ALL variables in the list are true.

#### NewOrBoolVar
```python
from model.constructor_tools import NewOrBoolVar

bool_var = NewOrBoolVar(
    model: EnhancedCpModel,
    variables: List[_cp.IntVar],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when ANY variable in the list is true.

### Interval and Containment Variables

#### NewPointInIntervalBoolVar
```python
from model.constructor_tools import NewPointInIntervalBoolVar

bool_var = NewPointInIntervalBoolVar(
    model: EnhancedCpModel,
    variable: Union[_cp.IntVar, int],
    interval: Union[Tuple[int, int], _cp.IntervalVar],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when a variable lies within an interval.

#### NewOverlapBoolVar
```python
from model.constructor_tools import NewOverlapBoolVar

bool_var = NewOverlapBoolVar(
    model: EnhancedCpModel,
    interval1: Union[_cp.IntervalVar, Tuple[int, int]],
    interval2: Union[_cp.IntervalVar, Tuple[int, int]],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when two intervals overlap.

#### NewContainedInBoolVar
```python
from model.constructor_tools import NewContainedInBoolVar

bool_var = NewContainedInBoolVar(
    model: EnhancedCpModel,
    interval1: Union[_cp.IntervalVar, Tuple[int, int]],
    interval2: Union[_cp.IntervalVar, Tuple[int, int]],
    name: str
) -> _cp.IntVar
```
Creates a boolean variable that is 1 when interval1 is contained within interval2.

### Conditional Aggregation Functions

#### NewMinSubjectToBools
```python
from model.constructor_tools import NewMinSubjectToBools

min_var, markers = NewMinSubjectToBools(
    model: EnhancedCpModel,
    values: Union[List[_cp.IntVar], List[int]],
    bools: List[_cp.IntVar],
    name: str,
    return_bool_markers: bool = False
) -> _cp.IntVar
```
Finds minimum value among variables where corresponding boolean is true.

#### NewMaxSubjectToBools
```python
from model.constructor_tools import NewMaxSubjectToBools

max_var, markers = NewMaxSubjectToBools(
    model: EnhancedCpModel,
    values: Union[List[_cp.IntVar], List[int]],
    bools: List[_cp.IntVar],
    name: str,
    return_bool_markers: bool = False
) -> _cp.IntVar
```
Finds maximum value among variables where corresponding boolean is true.

#### NewOrSubjectToBools
```python
from model.constructor_tools import NewOrSubjectToBools

bool_var = NewOrSubjectToBools(
    model: EnhancedCpModel,
    check_bools: List[_cp.IntVar],
    constraint_bools: List[_cp.IntVar],
    name: str
) -> _cp.IntVar
```
Logical OR operation applied to pairs of boolean variables subject to constraints.

#### NewAndSubjectToBools
```python
from model.constructor_tools import NewAndSubjectToBools

bool_var = NewAndSubjectToBools(
    model: EnhancedCpModel,
    check_bools: List[_cp.IntVar],
    constraint_bools: List[_cp.IntVar],
    name: str
) -> _cp.IntVar
```
Logical AND operation applied to pairs of boolean variables subject to constraints.

## Cross-References

### Imports from other modules
- **ortools.sat.python.cp_model** (as _cp): Core CP-SAT model components
- **typing**: Type annotations (Union, List, Tuple, TYPE_CHECKING)
- **model.EnhancedCpModel**: Enhanced model class with additional functionality

### Exports to other modules
This module exports all the constructor functions listed above, making them available for:
- Complex constraint building in scheduling problems
- Conditional optimization logic
- Interval and containment relationships
- Logical operation modeling

### Potential Issues
- **Circular dependency**: The module imports `EnhancedCpModel` from `model` but may be imported by other model components
- **Type checking**: Uses TYPE_CHECKING to avoid circular imports at runtime
- **Dependency on EnhancedCpModel**: Requires the enhanced model with `add_constraint_tags` method

## Integration Examples

### Example 1: Complex Scheduling Constraints
```python
from model.constructor_tools import NewOverlapBoolVar, NewContainedInBoolVar
from model import EnhancedCpModel

model = EnhancedCpModel()
task1 = model.NewIntervalVar(0, 5, 5, 'task1')
task2 = model.NewIntervalVar(3, 4, 8, 'task2')

# Check if tasks overlap
overlaps = NewOverlapBoolVar(model, task1, task2, 'tasks_overlap')

# Check if task1 is contained within task2
contained = NewContainedInBoolVar(model, task1, task2, 'task1_in_task2')
```

### Example 2: Conditional Optimization
```python
from model.constructor_tools import NewMinSubjectToBools, NewGreaterOrEqualBoolVar

model = EnhancedCpModel()
values = [model.NewIntVar(0, 100, f'val_{i}') for i in range(5)]
bools = [model.NewBoolVar(f'active_{i}') for i in range(5)]

# Create conditions where values must be >= 50 to be considered
conditions = [
    NewGreaterOrEqualBoolVar(model, val, 50, f'cond_{i}')
    for i, val in enumerate(values)
]

# Find minimum value among those that meet the condition
min_valid = NewMinSubjectToBools(model, values, conditions, 'min_valid_value')
```

### Example 3: Logical Combinations
```python
from model.constructor_tools import NewAndBoolVar, NewOrBoolVar

model = EnhancedCpModel()
conditions = [model.NewBoolVar(f'cond_{i}') for i in range(3)]

# All conditions must be true
all_true = NewAndBoolVar(model, conditions, 'all_conditions_met')

# Any condition is true
any_true = NewOrBoolVar(model, conditions, 'any_condition_met')
```

The module provides a powerful toolkit for building complex constraint relationships in CP-SAT models, enabling sophisticated optimization problems with conditional logic, interval relationships, and logical operations.

#### Complete API Reference

##### Functions

- **`NewGreaterOrEqualBoolVar`**(model: <_ast.Constant object at 0x000001D6B13E05B0>, variable: <_ast.Attribute object at 0x000001D6B13E05E0>, threshold: <_ast.Subscript object at 0x000001D6B13E0550>, name: <_ast.Name object at 0x000001D6B13E08B0>) -> <_ast.Attribute object at 0x000001D6B11F6940>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is greater or equal to a given threshold, and 0 when it is...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewGreaterOrEqualBoolVar(model, variable, threshold, ...)`

- **`NewLessOrEqualBoolVar`**(model: <_ast.Constant object at 0x000001D6B11F61C0>, variable: <_ast.Attribute object at 0x000001D6B11F6160>, threshold: <_ast.Subscript object at 0x000001D6B11F6DC0>, name: <_ast.Name object at 0x000001D6B135C190>) -> <_ast.Attribute object at 0x000001D6B126FD90>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is less than or equal to a given threshold, and 0 when it ...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewLessOrEqualBoolVar(model, variable, threshold, ...)`

- **`NewGreaterBoolVar`**(model: <_ast.Constant object at 0x000001D6B126FEB0>, variable: <_ast.Attribute object at 0x000001D6B126FF10>, threshold: <_ast.Subscript object at 0x000001D6B126FFA0>, name: <_ast.Name object at 0x000001D6B12910A0>) -> <_ast.Attribute object at 0x000001D6B128E6D0>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is strictly greater than a given threshold, and 0 when it ...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewGreaterBoolVar(model, variable, threshold, ...)`

- **`NewLessBoolVar`**(model: <_ast.Constant object at 0x000001D6B128E7C0>, variable: <_ast.Attribute object at 0x000001D6B128E820>, threshold: <_ast.Subscript object at 0x000001D6B128E8B0>, name: <_ast.Name object at 0x000001D6B128EA60>) -> <_ast.Attribute object at 0x000001D6B13DFB20>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is strictly less than a given threshold, and 0 when it is ...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewLessBoolVar(model, variable, threshold, ...)`

- **`NewEqualBoolVar`**(model: <_ast.Constant object at 0x000001D6B13DFC10>, variable: <_ast.Attribute object at 0x000001D6B13DFC70>, value: <_ast.Subscript object at 0x000001D6B13DFD00>, name: <_ast.Name object at 0x000001D6B13DFEE0>) -> <_ast.Attribute object at 0x000001D6B13E3F10>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is equal to a given value, and 0 when it is not equal to t...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewEqualBoolVar(model, variable, value, ...)`

- **`NewNotEqualBoolVar`**(model: <_ast.Constant object at 0x000001D6B13E6040>, variable: <_ast.Attribute object at 0x000001D6B13E60A0>, value: <_ast.Subscript object at 0x000001D6B13E6130>, name: <_ast.Name object at 0x000001D6B13E62E0>) -> <_ast.Attribute object at 0x000001D6B13EF370>
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is not equal to a given value, and 0 when it is equal to t...
  - **Calls:** `model.NewBoolVar`, `model.Add.OnlyEnforceIf`, `model.Add.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+3 more)
  - **Usage:** `NewNotEqualBoolVar(model, variable, value, ...)`

- **`NewAndBoolVar`**(model: <_ast.Constant object at 0x000001D6B13EF460>, variables: <_ast.Subscript object at 0x000001D6B13EF4C0>, name: <_ast.Name object at 0x000001D6B13EF610>) -> <_ast.Attribute object at 0x000001D6B13EC730>
  - Creates a boolean variable in a CP-SAT model that is 1 when all the specified boolean variables 
in the given list are true, and 0 when at least one o...
  - **Calls:** `model.NewBoolVar`, `model.AddBoolAnd.OnlyEnforceIf`, `model.AddBoolOr.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+4 more)
  - **Usage:** `NewAndBoolVar(model, variables, name)`

- **`NewOrBoolVar`**(model: <_ast.Constant object at 0x000001D6B13EC820>, variables: <_ast.Subscript object at 0x000001D6B13EC880>, name: <_ast.Name object at 0x000001D6B13ECA00>) -> <_ast.Attribute object at 0x000001D6B1212B50>
  - Creates a boolean variable in a CP-SAT model that is 1 when at least one of the specified boolean variables 
in the given list is true, and 0 when all...
  - **Calls:** `model.NewBoolVar`, `model.AddBoolOr.OnlyEnforceIf`, `model.AddBoolAnd.OnlyEnforceIf`, `model.add_constraint_tags`, `model.add_constraint_tags` (+4 more)
  - **Usage:** `NewOrBoolVar(model, variables, name)`

- **`NewPointInIntervalBoolVar`**(model: <_ast.Constant object at 0x000001D6B1212C40>, variable: <_ast.Subscript object at 0x000001D6B1212CA0>, interval: <_ast.Subscript object at 0x000001D6B1212B20>, name: <_ast.Name object at 0x000001D6B1215070>) -> <_ast.Attribute object at 0x000001D6B121A9D0>
  - Creates a boolean variable in a CP-SAT model that is 1 when a specified integer variable lies within a given interval, and 0 otherwise.

Parameters:
-...
  - **Calls:** `isinstance`, `isinstance`, `NewGreaterOrEqualBoolVar`, `NewLessOrEqualBoolVar`, `NewAndBoolVar` (+6 more)
  - **Usage:** `NewPointInIntervalBoolVar(model, variable, interval, ...)`

- **`NewOverlapBoolVar`**(model: <_ast.Constant object at 0x000001D6B121AAC0>, interval1: <_ast.Subscript object at 0x000001D6B121AB20>, interval2: <_ast.Subscript object at 0x000001D6B121ADC0>, name: <_ast.Name object at 0x000001D6B121E130>) -> <_ast.Attribute object at 0x000001D6B1225E50>
  - Creates a boolean variable in a CP-SAT model that is 1 when two intervals overlap, and 0 when they do not.

Parameters:
- model (EnhancedCpModel): The...
  - **Calls:** `model.NewBoolVar`, `NewPointInIntervalBoolVar`, `NewPointInIntervalBoolVar`, `NewPointInIntervalBoolVar`, `NewPointInIntervalBoolVar` (+16 more)
  - **Usage:** `NewOverlapBoolVar(model, interval1, interval2, ...)`

- **`NewContainedInBoolVar`**(model: <_ast.Constant object at 0x000001D6B1225F40>, interval1: <_ast.Subscript object at 0x000001D6B1225FA0>, interval2: <_ast.Subscript object at 0x000001D6B122A2B0>, name: <_ast.Name object at 0x000001D6B122A5B0>) -> <_ast.Attribute object at 0x000001D6B1205880>
  - Creates a boolean variable in a CP-SAT model that is 1 when the first interval is contained in the second interval, and 0 when it is not.

Parameters:...
  - **Calls:** `NewPointInIntervalBoolVar`, `NewPointInIntervalBoolVar`, `NewAndBoolVar`, `model.add_constraint_tags`, `model.add_constraint_tags` (+11 more)
  - **Usage:** `NewContainedInBoolVar(model, interval1, interval2, ...)`

- **`NewMinSubjectToBools`**(model: <_ast.Constant object at 0x000001D6B12059A0>, values: <_ast.Subscript object at 0x000001D6B1205A00>, bools: <_ast.Subscript object at 0x000001D6B1205460>, name: <_ast.Name object at 0x000001D6B1205E50>, return_bool_markers: <_ast.Name object at 0x000001D6B1205F10> = <_ast.Constant object at 0x000001D6B1205F40>) -> <_ast.Attribute object at 0x000001D6B15E2AC0>
  - Creates a new integer variable representing the minimum value among a list of integer variables, subject to boolean conditions.

Parameters:
- model (...
  - **Calls:** `max`, `min`, `model.NewIntVar`, `enumerate`, `NewOrBoolVar` (+24 more)
  - **Usage:** `NewMinSubjectToBools(model, values, bools, ...)`

- **`NewMaxSubjectToBools`**(model: <_ast.Constant object at 0x000001D6B15E27F0>, values: <_ast.Subscript object at 0x000001D6B15E2820>, bools: <_ast.Subscript object at 0x000001D6B15E2340>, name: <_ast.Name object at 0x000001D6B15E2460>, return_bool_markers: <_ast.Name object at 0x000001D6B15E24C0> = <_ast.Constant object at 0x000001D6B15E2550>) -> <_ast.Attribute object at 0x000001D6B15BF1F0>
  - Creates a new integer variable representing the maximum value among a list of integer variables, subject to boolean conditions.

Parameters:
- model (...
  - **Calls:** `max`, `min`, `model.NewIntVar`, `enumerate`, `NewOrBoolVar` (+24 more)
  - **Usage:** `NewMaxSubjectToBools(model, values, bools, ...)`

- **`NewOrSubjectToBools`**(model: <_ast.Constant object at 0x000001D6B15BF2E0>, check_bools: <_ast.Subscript object at 0x000001D6B15BF340>, constraint_bools: <_ast.Subscript object at 0x000001D6B15BF490>, name: <_ast.Name object at 0x000001D6B15BF5E0>) -> <_ast.Attribute object at 0x000001D6B15C9970>
  - Creates a boolean variable representing the logical OR operation applied to pairs of boolean variables subject to additional constraint boolean variab...
  - **Calls:** `range`, `NewOrBoolVar`, `len`, `len`, `len` (+4 more)
  - **Usage:** `NewOrSubjectToBools(model, check_bools, constraint_bools, ...)`

- **`NewAndSubjectToBools`**(model: <_ast.Constant object at 0x000001D6B15C9A60>, check_bools: <_ast.Subscript object at 0x000001D6B15C9AC0>, constraint_bools: <_ast.Subscript object at 0x000001D6B15C9C10>, name: <_ast.Name object at 0x000001D6B15C9D60>) -> <_ast.Attribute object at 0x000001D6B137D250>
  - Creates a boolean variable representing the logical AND operation applied to pairs of boolean variables subject to additional constraint boolean varia...
  - **Calls:** `range`, `model.NewIntVar`, `model.NewIntVar`, `model.Add`, `model.Add` (+14 more)
  - **Usage:** `NewAndSubjectToBools(model, check_bools, constraint_bools, ...)`

##### Constants

- **`M`** = `<_ast.Constant object at 0x000001D6B13E04F0>` (int) (line 8)

##### Called By

- **`NewAndBoolVar`** called by `model\constructors.py` (1 times)
  - Line 83 in class:EnhancedConstructorsMixin::function:NewAndBoolVar
- **`NewAndSubjectToBools`** called by `model\constructors.py` (1 times)
  - Line 151 in class:EnhancedConstructorsMixin::function:NewAndSubjectToBools
- **`NewContainedInBoolVar`** called by `model\constructors.py` (1 times)
  - Line 115 in class:EnhancedConstructorsMixin::function:NewContainedInBoolVar
- **`NewEqualBoolVar`** called by `model\constructors.py` (1 times)
  - Line 67 in class:EnhancedConstructorsMixin::function:NewEqualBoolVar
- **`NewGreaterBoolVar`** called by `model\constructors.py` (1 times)
  - Line 51 in class:EnhancedConstructorsMixin::function:NewGreaterBoolVar
- **`NewGreaterOrEqualBoolVar`** called by `model\constructors.py` (1 times)
  - Line 35 in class:EnhancedConstructorsMixin::function:NewGreaterOrEqualBoolVar
- **`NewLessBoolVar`** called by `model\constructors.py` (1 times)
  - Line 59 in class:EnhancedConstructorsMixin::function:NewLessBoolVar
- **`NewLessOrEqualBoolVar`** called by `model\constructors.py` (1 times)
  - Line 43 in class:EnhancedConstructorsMixin::function:NewLessOrEqualBoolVar
- **`NewMaxSubjectToBools`** called by `model\constructors.py` (1 times)
  - Line 134 in class:EnhancedConstructorsMixin::function:NewMaxSubjectToBools
- **`NewMinSubjectToBools`** called by `model\constructors.py` (1 times)
  - Line 125 in class:EnhancedConstructorsMixin::function:NewMinSubjectToBools
- **`NewNotEqualBoolVar`** called by `model\constructors.py` (1 times)
  - Line 75 in class:EnhancedConstructorsMixin::function:NewNotEqualBoolVar
- **`NewOrBoolVar`** called by `model\constructors.py` (1 times)
  - Line 90 in class:EnhancedConstructorsMixin::function:NewOrBoolVar
- **`NewOrSubjectToBools`** called by `model\constructors.py` (1 times)
  - Line 143 in class:EnhancedConstructorsMixin::function:NewOrSubjectToBools
- **`NewOverlapBoolVar`** called by `model\constructors.py` (1 times)
  - Line 107 in class:EnhancedConstructorsMixin::function:NewOverlapBoolVar
- **`NewPointInIntervalBoolVar`** called by `model\constructors.py` (1 times)
  - Line 99 in class:EnhancedConstructorsMixin::function:NewPointInIntervalBoolVar

##### Imports

**Standard Library:**
- `from typing import Union`
- `from typing import List`
- `from typing import Tuple`
- `from typing import TYPE_CHECKING`

**Third Party:**
- `from ortools.sat.python import cp_model as _cp`
- `from model import EnhancedCpModel`

---

### `model\constructors.py`

## Module Purpose
This module provides `EnhancedConstructorsMixin`, a mix-in class that exposes helper constructor methods for creating constraint-related boolean variables in Google OR-Tools CP-SAT models. It serves as a method-based interface to external constructor functions, making them available as instance methods on enhanced model classes.

## Public API

### EnhancedConstructorsMixin Class

**Usage pattern:**
```python
from model.constructors import EnhancedConstructorsMixin
from ortools.sat.python import cp_model

class EnhancedCpModel(cp_model.CpModel, EnhancedConstructorsMixin):
    pass

model = EnhancedCpModel()
bool_var = model.NewGreaterOrEqualBoolVar(variable, threshold, "name")
```

**Method signatures:**

1. **Comparison Constructors:**
```python
def NewGreaterOrEqualBoolVar(self, variable: _cp.IntVar, 
                           threshold: Union[_cp.IntVar, int], 
                           name: str) -> _cp.IntVar

def NewLessOrEqualBoolVar(self, variable: _cp.IntVar, 
                         threshold: Union[_cp.IntVar, int], 
                         name: str) -> _cp.IntVar

def NewGreaterBoolVar(self, variable: _cp.IntVar, 
                     threshold: Union[_cp.IntVar, int], 
                     name: str) -> _cp.IntVar

def NewLessBoolVar(self, variable: _cp.IntVar, 
                  threshold: Union[_cp.IntVar, int], 
                  name: str) -> _cp.IntVar

def NewEqualBoolVar(self, variable: _cp.IntVar, 
                   value: Union[_cp.IntVar, int], 
                   name: str) -> _cp.IntVar

def NewNotEqualBoolVar(self, variable: _cp.IntVar, 
                      value: Union[_cp.IntVar, int], 
                      name: str) -> _cp.IntVar
```

2. **Logical Constructors:**
```python
def NewAndBoolVar(self, variables: List[_cp.IntVar], name: str) -> _cp.IntVar
def NewOrBoolVar(self, variables: List[_cp.IntVar], name: str) -> _cp.IntVar
```

3. **Interval Constructors:**
```python
def NewPointInIntervalBoolVar(self, variable: Union[_cp.IntVar, int], 
                            interval: Union[Tuple[int, int], _cp.IntervalVar], 
                            name: str) -> _cp.IntVar

def NewOverlapBoolVar(self, interval1: Union[_cp.IntervalVar, Tuple[int, int]], 
                     interval2: Union[_cp.IntervalVar, Tuple[int, int]], 
                     name: str) -> _cp.IntVar

def NewContainedInBoolVar(self, interval1: Union[_cp.IntervalVar, Tuple[int, int]], 
                         interval2: Union[_cp.IntervalVar, Tuple[int, int]], 
                         name: str) -> _cp.IntVar
```

4. **Subject-To Constructors:**
```python
def NewMinSubjectToBools(self, values: Union[List[_cp.IntVar], List[int]], 
                        bools: List[_cp.IntVar], name: str, 
                        return_bool_markers: bool = False)

def NewMaxSubjectToBools(self, values: Union[List[_cp.IntVar], List[int]], 
                        bools: List[_cp.IntVar], name: str, 
                        return_bool_markers: bool = False)

def NewOrSubjectToBools(self, check_bools: List[_cp.IntVar], 
                       constraint_bools: List[_cp.IntVar], 
                       name: str) -> _cp.IntVar

def NewAndSubjectToBools(self, check_bools: List[_cp.IntVar], 
                        constraint_bools: List[_cp.IntVar], 
                        name: str) -> _cp.IntVar
```

**Dependencies:**
- `ortools.sat.python.cp_model` (as `_cp`)
- External constructor functions from `model.constructor_tools.py`

**Return types:**
- Most methods return `_cp.IntVar` (boolean variables)
- `NewMinSubjectToBools` and `NewMaxSubjectToBools` return unspecified results (likely tuples or custom objects)
- All return values are CP-SAT model variables that can be used in constraints

## Cross-References

**Imports from other modules:**
- From `ortools.sat.python.cp_model`: Core CP-SAT model components
- From `model.constructor_tools.py`: All 15 constructor functions (e.g., `NewGreaterOrEqualBoolVar`, `NewAndBoolVar`, etc.)
- From `typing`: Type annotations (`List`, `Union`, `Tuple`)

**Exports to other modules:**
- `EnhancedConstructorsMixin` class for inheritance by enhanced model classes
- No direct exports - designed to be mixed into custom model classes

**Potential issues:**
- Circular dependency risk: This module imports from `model.constructor_tools.py` which might import from this module
- Missing import for `_cp.IntervalVar` type (though it's used in type annotations)
- All constructor functions are imported from the same module they might be defined in

## Integration Examples

**Example 1: Creating an enhanced model with comparison constraints**
```python
from model.constructors import EnhancedConstructorsMixin
from ortools.sat.python import cp_model

class EnhancedModel(cp_model.CpModel, EnhancedConstructorsMixin):
    pass

model = EnhancedModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')

# Create boolean variables for constraints
x_ge_5 = model.NewGreaterOrEqualBoolVar(x, 5, 'x_ge_5')
y_eq_x = model.NewEqualBoolVar(y, x, 'y_eq_x')

# Use in model constraints
model.Add(x_ge_5 == 1)
model.Add(y_eq_x == 1)
```

**Example 2: Using logical and interval operations**
```python
# Create OR constraint over multiple conditions
conditions = [
    model.NewGreaterBoolVar(x, 8, 'x_gt_8'),
    model.NewLessBoolVar(y, 2, 'y_lt_2')
]
any_condition = model.NewOrBoolVar(conditions, 'any_condition')

# Check if value is in interval
in_range = model.NewPointInIntervalBoolVar(x, (3, 7), 'x_in_3_7')
```

**Example 3: Complex subject-to constraints**
```python
values = [1, 3, 5, 7, 9]
bools = [model.NewBoolVar(f'active_{i}') for i in range(5)]

# Find minimum value among active options
min_val, markers = model.NewMinSubjectToBools(
    values, bools, 'min_value', return_bool_markers=True
)
```

#### Complete API Reference

##### Class: `EnhancedConstructorsMixin`

**Description:** Mix-in that exposes helper constructors as *methods* on EnhancedCpModel
while keeping the implementation external....

**Methods:**

- **`NewGreaterOrEqualBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B13E09D0>, threshold: <_ast.Subscript object at 0x000001D6B13E0A30>, name: <_ast.Name object at 0x000001D6B13E0580>) -> <_ast.Attribute object at 0x000001D6B13E0550>
  - **Calls:** `NewGreaterOrEqualBoolVar`

- **`NewLessOrEqualBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B13E07F0>, threshold: <_ast.Subscript object at 0x000001D6B13E08B0>, name: <_ast.Name object at 0x000001D6B11F6880>) -> <_ast.Attribute object at 0x000001D6B11F67C0>
  - **Calls:** `NewLessOrEqualBoolVar`

- **`NewGreaterBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B11F6790>, threshold: <_ast.Subscript object at 0x000001D6B11F6670>, name: <_ast.Name object at 0x000001D6B11F6C70>) -> <_ast.Attribute object at 0x000001D6B11F6FA0>
  - **Calls:** `NewGreaterBoolVar`

- **`NewLessBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B11F6370>, threshold: <_ast.Subscript object at 0x000001D6B11F61C0>, name: <_ast.Name object at 0x000001D6B135C1C0>) -> <_ast.Attribute object at 0x000001D6B135C3D0>
  - **Calls:** `NewLessBoolVar`

- **`NewEqualBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B135C610>, value: <_ast.Subscript object at 0x000001D6B135C760>, name: <_ast.Name object at 0x000001D6B135C4C0>) -> <_ast.Attribute object at 0x000001D6B135C9D0>
  - **Calls:** `NewEqualBoolVar`

- **`NewNotEqualBoolVar`**(self, variable: <_ast.Attribute object at 0x000001D6B135C7C0>, value: <_ast.Subscript object at 0x000001D6B135CA30>, name: <_ast.Name object at 0x000001D6B135CCA0>) -> <_ast.Attribute object at 0x000001D6B135CFD0>
  - **Calls:** `NewNotEqualBoolVar`

- **`NewAndBoolVar`**(self, variables: <_ast.Subscript object at 0x000001D6B135CAF0>, name: <_ast.Name object at 0x000001D6B135C220>) -> <_ast.Attribute object at 0x000001D6B126FB50>
  - **Calls:** `NewAndBoolVar`

- **`NewOrBoolVar`**(self, variables: <_ast.Subscript object at 0x000001D6B126FDF0>, name: <_ast.Name object at 0x000001D6B126FF70>) -> <_ast.Attribute object at 0x000001D6B12911C0>
  - **Calls:** `NewOrBoolVar`

- **`NewPointInIntervalBoolVar`**(self, variable: <_ast.Subscript object at 0x000001D6B1291790>, interval: <_ast.Subscript object at 0x000001D6B12912E0>, name: <_ast.Name object at 0x000001D6B12914C0>) -> <_ast.Attribute object at 0x000001D6B12918E0>
  - **Calls:** `NewPointInIntervalBoolVar`

- **`NewOverlapBoolVar`**(self, interval1: <_ast.Subscript object at 0x000001D6B1291E50>, interval2: <_ast.Subscript object at 0x000001D6B137D130>, name: <_ast.Name object at 0x000001D6B137D4C0>) -> <_ast.Attribute object at 0x000001D6B137D670>
  - **Calls:** `NewOverlapBoolVar`

- **`NewContainedInBoolVar`**(self, interval1: <_ast.Subscript object at 0x000001D6B137D790>, interval2: <_ast.Subscript object at 0x000001D6B137DA30>, name: <_ast.Name object at 0x000001D6B137DD30>) -> <_ast.Attribute object at 0x000001D6B137DEE0>
  - **Calls:** `NewContainedInBoolVar`

- **`NewMinSubjectToBools`**(self, values: <_ast.Subscript object at 0x000001D6B137D310>, bools: <_ast.Subscript object at 0x000001D6B1290340>, name: <_ast.Name object at 0x000001D6B12904C0>, return_bool_markers: <_ast.Name object at 0x000001D6B1290520> = <_ast.Constant object at 0x000001D6B1290550>)
  - **Calls:** `NewMinSubjectToBools`

- **`NewMaxSubjectToBools`**(self, values: <_ast.Subscript object at 0x000001D6B1290880>, bools: <_ast.Subscript object at 0x000001D6B1290B80>, name: <_ast.Name object at 0x000001D6B1290CD0>, return_bool_markers: <_ast.Name object at 0x000001D6B1290D30> = <_ast.Constant object at 0x000001D6B1290D60>)
  - **Calls:** `NewMaxSubjectToBools`

- **`NewOrSubjectToBools`**(self, check_bools: <_ast.Subscript object at 0x000001D6B128A040>, constraint_bools: <_ast.Subscript object at 0x000001D6B128A190>, name: <_ast.Name object at 0x000001D6B128A2E0>) -> <_ast.Attribute object at 0x000001D6B128A490>
  - **Calls:** `NewOrSubjectToBools`

- **`NewAndSubjectToBools`**(self, check_bools: <_ast.Subscript object at 0x000001D6B128A5E0>, constraint_bools: <_ast.Subscript object at 0x000001D6B128A730>, name: <_ast.Name object at 0x000001D6B128A8B0>) -> <_ast.Attribute object at 0x000001D6B128AA60>
  - **Calls:** `NewAndSubjectToBools`

**Usage Pattern:**
```python
from model\constructors import EnhancedConstructorsMixin
instance = EnhancedConstructorsMixin()
```

##### Dependencies on Other Modules

- **`model\constructor_tools.py`**: uses `NewAndBoolVar, NewOrSubjectToBools, NewPointInIntervalBoolVar, NewMinSubjectToBools, NewOverlapBoolVar` (+10 more)

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `from typing import List`
- `from typing import Union`
- `from typing import Tuple`

**Third Party:**
- `from ortools.sat.python import cp_model as _cp`
- `from constructors import NewGreaterOrEqualBoolVar`
- `from constructors import NewLessOrEqualBoolVar`
- `from constructors import NewGreaterBoolVar`
- `from constructors import NewLessBoolVar`
- `from constructors import NewEqualBoolVar`
- `from constructors import NewNotEqualBoolVar`
- `from constructors import NewAndBoolVar`
- `from constructors import NewOrBoolVar`
- `from constructors import NewPointInIntervalBoolVar`
- `from constructors import NewOverlapBoolVar`
- `from constructors import NewContainedInBoolVar`
- `from constructors import NewMinSubjectToBools`
- `from constructors import NewMaxSubjectToBools`
- `from constructors import NewOrSubjectToBools`
- `from constructors import NewAndSubjectToBools`

---

### `model\debug.py`

## Module Purpose
This module provides debugging and introspection utilities for constraint programming models built on Google OR-Tools' CP-SAT solver. It extends the base CpModel class with functionality for identifying minimal infeasible subsets (MIS), constraint relaxation, model summarization, and constraint management.

## Public API

### _DebugMixin Class
**Usage pattern:**
```python
from model.debug import _DebugMixin
from ortools.sat.python import cp_model

class EnhancedCpModel(_DebugMixin, cp_model.CpModel):
    pass

model = EnhancedCpModel()
```

**Key Methods:**

1. **debug_infeasible()**
```python
def debug_infeasible(
    self,
    solver: Optional[_cp.CpSolver] = None,
    **solver_params,
) -> Dict[str, Any]:
```
- **Purpose**: Finds minimal set of constraints to disable to make model feasible
- **Returns**: Dict with keys: status, feasible, disabled_constraints, total_disabled, method
- **Dependencies**: ortools.sat.python.cp_model, random, typing

2. **summary()**
```python
def summary(self) -> Dict[str, Any]:
```
- **Purpose**: Provides comprehensive model overview
- **Returns**: Dict with constraint/variable counts, types, and status information

3. **validate_model()**
```python
def validate_model(self) -> Dict[str, Any]:
```
- **Purpose**: Basic diagnostics for model validation
- **Returns**: Dict with issues and warnings lists

4. **Constraint Management Methods:**
```python
def get_constraint_names(self) -> List[str]
def get_variable_names(self) -> List[str]
def get_constraint_info(self, name: str) -> ConstraintInfo
def get_constraints_by_type(self, constraint_type: str) -> List[str]
def get_constraints_by_tag(self, tag: str) -> List[str]
def get_enabled_constraints(self) -> List[str]
def get_disabled_constraints(self) -> List[str]
```

5. **Model Transformation Methods:**
```python
def create_relaxed_copy(self, relaxation_factor: float = 0.1) -> Any
def create_subset_copy(self, constraint_names: Sequence[str]) -> Any
```

## Cross-References

**Imports from other modules:**
- `ortools.sat.python.cp_model` (as _cp): Base CP-SAT functionality
- `constraints.ConstraintInfo`: Constraint metadata structure
- Standard library: random, typing (Any, Dict, List, Optional, Sequence)

**Exports to other modules:**
- `_DebugMixin` class designed to be mixed into CpModel subclasses
- Provides debugging infrastructure for constraint programming models

**Potential issues:**
- The mixin assumes the presence of `_constraints` attribute (created by `_ensure_constraints()`)
- Return types use `Any` for methods returning self-type (due to mixin pattern)
- Requires proper integration with a CpModel subclass

## Integration Examples

**Basic usage with model debugging:**
```python
from model.debug import _DebugMixin
from ortools.sat.python import cp_model

class DebuggableModel(_DebugMixin, cp_model.CpModel):
    pass

# Create and populate model
model = DebuggableModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')
model.Add(x + y > 20)  # Potentially infeasible constraint

# Debug infeasibility
result = model.debug_infeasible()
print(f"Minimal infeasible subset: {result['disabled_constraints']}")

# Get model summary
print(model.summary())

# Create relaxed version
relaxed_model = model.create_relaxed_copy(0.2)
```

**Constraint introspection:**
```python
# Get all constraint names
constraint_names = model.get_constraint_names()

# Get info about specific constraint
info = model.get_constraint_info('constraint_1')
print(f"Type: {info.constraint_type}, Enabled: {info.enabled}")

# Filter constraints by type
linear_constraints = model.get_constraints_by_type('linear')
```

**Model validation:**
```python
validation = model.validate_model()
if validation['issues']:
    print("Model has issues:", validation['issues'])
if validation['warnings']:
    print("Warnings:", validation['warnings'])
```

The module provides essential debugging capabilities for CP-SAT models, particularly useful for identifying why models are infeasible and understanding model structure through comprehensive introspection.

#### Complete API Reference

##### Class: `_DebugMixin`

**Description:** Introspection, MIS, relaxation & subset utilities....

**Inherits from:** `_cp.CpModel`

**Methods:**

- **`debug_infeasible`**(self, solver: <_ast.Subscript object at 0x000001D6B13E0E80> = <_ast.Constant object at 0x000001D6B13E0940>, solver_params) -> <_ast.Subscript object at 0x000001D6B137D820>
  - Find a minimal set of constraints to disable to make the model feasible.

Returns
-------
dict with keys:
    status, feasible, disabled_constraints, ...
  - **Calls:** `solver_params.items`, `self._solve`, `self.clone`, `self.get_enabled_constraints`, `mis_model.Minimize` (+19 more)

- **`summary`**(self) -> <_ast.Subscript object at 0x000001D6B15E3580>
  - One-stop overview of the model....
  - **Calls:** `self._ensure_constraints.values`, `getattr.values`, `set.union`, `len`, `len` (+17 more)

- **`validate_model`**(self) -> <_ast.Subscript object at 0x000001D6B15D0400>
  - Basic diagnostics: disabled constraints, multiple objectives, etc....
  - **Calls:** `self.get_disabled_constraints`, `warnings.append`, `len`, `len`

- **`get_constraint_names`**(self) -> <_ast.Subscript object at 0x000001D6B15D07C0>
  - **Calls:** `list`, `self._ensure_constraints.keys`, `self._ensure_constraints`

- **`get_variable_names`**(self) -> <_ast.Subscript object at 0x000001D6B15D0B20>
  - **Calls:** `list`, `getattr.keys`, `getattr`

- **`get_constraint_info`**(self, name: <_ast.Name object at 0x000001D6B15D0CD0>) -> <_ast.Name object at 0x000001D6B15C1190>
  - **Calls:** `self._ensure_constraints`, `ValueError`, `self._ensure_constraints`

- **`get_constraints_by_type`**(self, constraint_type: <_ast.Name object at 0x000001D6B15C12B0>) -> <_ast.Subscript object at 0x000001D6B15C1610>
  - **Calls:** `self._ensure_constraints.items`, `self._ensure_constraints`

- **`get_constraints_by_tag`**(self, tag: <_ast.Name object at 0x000001D6B15C17C0>) -> <_ast.Subscript object at 0x000001D6B15C1BB0>
  - **Calls:** `self._ensure_constraints.items`, `self._ensure_constraints`

- **`get_enabled_constraints`**(self) -> <_ast.Subscript object at 0x000001D6B15C1F10>
  - **Calls:** `self._ensure_constraints.items`, `self._ensure_constraints`

- **`get_disabled_constraints`**(self) -> <_ast.Subscript object at 0x000001D6B15CE9A0>
  - **Calls:** `self._ensure_constraints.items`, `self._ensure_constraints`

- **`create_relaxed_copy`**(self, relaxation_factor: <_ast.Name object at 0x000001D6B15CE850> = <_ast.Constant object at 0x000001D6B15CE9D0>) -> <_ast.Name object at 0x000001D6B15E7070>
  - Return a relaxed copy by randomly disabling `relaxation_factor`
fraction of currently-enabled constraints....
  - **Calls:** `self.clone`, `relaxed.get_enabled_constraints`, `int`, `ValueError`, `random.sample` (+2 more)

- **`create_subset_copy`**(self, constraint_names: <_ast.Subscript object at 0x000001D6B15E7160>) -> <_ast.Name object at 0x000001D6B15E77C0>
  - Return a copy with *only* the listed constraints enabled....
  - **Calls:** `self.clone`, `subset.disable_constraints`, `subset.enable_constraints`, `subset.get_constraint_names`, `list`

- **`_ensure_constraints`**(self) -> <_ast.Subscript object at 0x000001D6B15E7C40>
  - **Calls:** `hasattr`

**Usage Pattern:**
```python
from model\debug import _DebugMixin
# Inherits from: _cp.CpModel
instance = _DebugMixin()
```

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `from typing import Any`
- `from typing import Dict`
- `from typing import List`
- `from typing import Optional`
- `from typing import Sequence`

**Third Party:**
- `from __future__ import annotations`
- `random`
- `ortools.sat.python.cp_model as _cp`
- `from constraints import ConstraintInfo`

---

### `model\io.py`

## Module Purpose
This module provides model persistence capabilities for OR-Tools CP-SAT models, enabling complete serialization and deserialization of constraint programming models including both the underlying OR-Tools protocol buffers and Python-side metadata.

## Public API

### Class: `_IOMixin` (inherits from `_cp.CpModel`)

**Usage pattern:**
```python
from ortools.sat.python.cp_model import CpModel
from your_module import _IOMixin  # Assuming this is mixed into a model class

class EnhancedCpModel(_IOMixin, CpModel):
    pass

model = EnhancedCpModel()
```

**Method signatures:**

#### `export_to_file(filename: str) -> None`
Persists the entire EnhancedCpModel (proto + metadata) to disk as a ZIP archive.

**Parameters:**
- `filename`: Path to save the model file

**Dependencies:** Requires `zipfile`, `json`, and OR-Tools protobuf serialization

#### `import_from_file(filename: str) -> None`
Loads a model from a saved file, restoring both OR-Tools proto and metadata.

**Parameters:**
- `filename`: Path to load the model file from

**Dependencies:** Requires `zipfile`, `json`, and OR-Tools protobuf parsing

#### `_serialize_arg(arg: Any) -> Any`
Converts OR-Tools objects to JSON-serializable primitives (internal helper).

#### `_deserialize_arg(serialized_arg: Any, var_mapping: Dict[str, Any]) -> Any`
Re-hydrates primitives back to OR-Tools objects (internal helper).

#### `_create_solving_model() -> _cp.CpModel`
Builds a clean CpModel with only enabled constraints & objectives (internal helper).

## Cross-References

### Imports from other modules:
- **`ortools.sat.python.cp_model`** (as `_cp`): Base model functionality
- **`ortools.sat.cp_model_pb2`**: Protocol buffer definitions
- **`zipfile`**: Archive creation/reading
- **`json`**: Metadata serialization
- **`typing`**: Type annotations (Any, Dict, List, Tuple, Union)

### Exports to other modules:
This appears to be a mixin class designed to be inherited by model classes, providing persistence capabilities to enhanced CP-SAT models.

### Potential issues:
- **Circular dependencies**: The class inherits from `_cp.CpModel` but also likely needs to be mixed into classes that extend CpModel
- **Internal methods**: `_serialize_arg`, `_deserialize_arg`, and `_create_solving_model` are implementation details but are called by public methods
- **Type completeness**: The serialization/deserialization handles various OR-Tools types but may not cover all possible constraint expressions

## Integration Examples

**Example 1: Basic model persistence**
```python
from your_module import EnhancedCpModel

# Create and populate model
model = EnhancedCpModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')
model.Add(x + y <= 15)
model.Minimize(x + y)

# Save model
model.export_to_file('my_model.zip')

# Later, load model
new_model = EnhancedCpModel()
new_model.import_from_file('my_model.zip')
```

**Example 2: Integration with solving workflow**
```python
def solve_with_persistence(model, filename):
    if os.path.exists(filename):
        model.import_from_file(filename)
    else:
        # Build model from scratch
        build_complex_model(model)
        model.export_to_file(filename)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status, solver
```

**Example 3: Model debugging and analysis**
```python
# Save model for later analysis
model.export_to_file('debug_model.zip')

# Load in different session for analysis
analysis_model = EnhancedCpModel()
analysis_model.import_from_file('debug_model.zip')
print(f"Model has {len(analysis_model._variables)} variables")
print(f"Model has {len(analysis_model._constraints)} constraints")
```

The module provides complete round-trip serialization, preserving variable definitions, constraints, objectives, and metadata tags while maintaining Python object consistency.

#### Complete API Reference

##### Class: `_IOMixin`

**Description:** Model persistence helpers....

**Inherits from:** `_cp.CpModel`

**Methods:**

- **`export_to_file`**(self, filename: <_ast.Name object at 0x000001D6B11F6460>) -> <_ast.Constant object at 0x000001D6B137DF40>
  - Persist the *entire* EnhancedCpModel (proto + metadata) to disk.

The file is a ZIP archive with two entries:
    model.pb   – raw OR-Tools proto
    ...
  - **Calls:** `self._create_solving_model`, `temp_model.Proto.SerializeToString`, `getattr.items`, `getattr.items`, `getattr` (+13 more)

- **`import_from_file`**(self, filename: <_ast.Name object at 0x000001D6B1291370>) -> <_ast.Constant object at 0x000001D6B15DFF10>
  - Load a model from a saved file, restoring OR-Tools proto and metadata.
Re-creates constraints to guarantee Python-side consistency....
  - **Calls:** `self._clear_model`, `zipfile.ZipFile`, `zf.read`, `cp_model_pb2.CpModelProto`, `proto.ParseFromString` (+41 more)

- **`_serialize_arg`**(self, arg: <_ast.Name object at 0x000001D6B15BD040>) -> <_ast.Name object at 0x000001D6B1215220>
  - Convert OR-Tools objects to JSON-serialisable primitives....
  - **Calls:** `hasattr`, `isinstance`, `isinstance`, `isinstance`, `isinstance` (+16 more)

- **`_deserialize_arg`**(self, serialized_arg: <_ast.Name object at 0x000001D6B1215D90>, var_mapping: <_ast.Subscript object at 0x000001D6B1215E50>) -> <_ast.Name object at 0x000001D6B121EA30>
  - Re-hydrate primitives back to OR-Tools objects....
  - **Calls:** `isinstance`, `isinstance`, `isinstance`, `isinstance`, `serialized_arg.get` (+6 more)

- **`_create_solving_model`**(self) -> <_ast.Attribute object at 0x000001D6B1228AC0>
  - Build a *clean* CpModel with only enabled constraints & objectives.
Used by export_to_file and debug_infeasible....
  - **Calls:** `_cp.CpModel`, `getattr.items`, `getattr`, `getattr.values`, `getattr` (+20 more)

**Usage Pattern:**
```python
from model\io import _IOMixin
# Inherits from: _cp.CpModel
instance = _IOMixin()
```

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `json`
- `from typing import Any`
- `from typing import Dict`
- `from typing import List`
- `from typing import Tuple`
- `from typing import Union`

**Third Party:**
- `from __future__ import annotations`
- `zipfile`
- `ortools.sat.python.cp_model as _cp`
- `from ortools.sat import cp_model_pb2`

---

### `model\model.py`

## Module Purpose
This module provides `EnhancedCpModel`, a drop-in replacement for Google OR-Tools' `CpModel` with enhanced functionality including debugging capabilities, model cloning, and input/output operations. It extends the base constraint programming model with rich tracking of variables, constraints, and objectives.

## Public API

### EnhancedCpModel Class

**Usage pattern:**
```python
from model.model import EnhancedCpModel

model = EnhancedCpModel()
# Use as a regular CpModel but with enhanced features
x = model.new_int_var(0, 10, 'x')
model.add(x <= 5)
```

**Method signatures:**

- `__init__(self) -> None`: Initializes the enhanced model, inheriting from all mixins and the base CpModel.

- `clear_model(self) -> None`: Resets everything including OR-Tools proto, Python registries, and counters. Called internally by import/clone operations.

**Dependencies:**
- Requires `ortools.sat.python.cp_model` (aliased as `_cp`)
- Inherits from multiple mixins: `_VariablesMixin`, `_ConstraintsMixin`, `_ObjectivesMixin`, `_DebugMixin`, `_IOMixin`

**Return types:**
- Constructor returns an `EnhancedCpModel` instance
- `clear_model()` returns `None`

## Cross-References

**Imports from other modules:**
- `_VariablesMixin` from `model.variables` - Provides enhanced variable management
- `_ConstraintsMixin` from `model.constraints` - Provides enhanced constraint tracking
- `_ObjectivesMixin` from `model.objectives` - Provides objective function management
- `_DebugMixin` from `model.debug` - Provides debugging capabilities
- `_IOMixin` from `model.io` - Provides import/export functionality
- `CpModel` from `ortools.sat.python.cp_model` - Base constraint programming model

**Exports to other modules:**
- `EnhancedCpModel` is the main export and would be imported by application code
- The mixin classes are designed specifically for this composite class

**Potential issues:**
- The module uses relative imports (`model.variables`, `model.constraints`, etc.) which may cause issues if the package structure changes
- Multiple inheritance from several mixins could lead to method name conflicts
- The `clear_model()` method directly manipulates `__dict__` which is fragile

## Integration Examples

**Basic usage with enhanced debugging:**
```python
from model.model import EnhancedCpModel

model = EnhancedCpModel()
x = model.new_int_var(0, 10, 'x')
y = model.new_int_var(0, 10, 'y')
model.add(x + y == 10)

# Enhanced debugging capabilities
print(f"Variables created: {model._variable_counter}")
print(f"Constraints created: {model._constraint_counter}")

# Clear and reuse model
model.clear_model()
```

**Integration with OR-Tools solver:**
```python
from model.model import EnhancedCpModel
from ortools.sat.python import cp_model

model = EnhancedCpModel()
# Build model using enhanced features
x = model.new_int_var(0, 10, 'x')
model.maximize(x)

# Use standard OR-Tools solver
solver = cp_model.CpSolver()
status = solver.solve(model)  # EnhancedCpModel is compatible with base CpModel

# Access enhanced debugging info
if status == cp_model.OPTIMAL:
    print(f"Solution: {solver.value(x)}")
    print(f"Model stats: {model.get_model_stats()}")  # Assuming _DebugMixin provides this
```

**Model cloning and serialization:**
```python
from model.model import EnhancedCpModel

# Create and populate a model
model1 = EnhancedCpModel()
x = model1.new_int_var(0, 10, 'x')
model1.add(x >= 5)

# Clone the model (assuming _IOMixin provides cloning)
model2 = model1.clone()  # Creates identical copy with fresh state

# Export model (assuming _IOMixin provides export)
model_data = model1.export_to_dict()  # Serialize model state
```

The `EnhancedCpModel` maintains full compatibility with the base `CpModel` while adding rich tracking and management capabilities through its mixin composition.

#### Complete API Reference

##### Class: `EnhancedCpModel`

**Description:** Drop-in replacement for CpModel with rich debugging, cloning and I/O....

**Inherits from:** `_VariablesMixin`, `_ConstraintsMixin`, `_ObjectivesMixin`, `_DebugMixin`, `_IOMixin`, `_cp.CpModel`

**Methods:**

- **`__init__`**(self) -> <_ast.Constant object at 0x000001D6B135C580>
  - **Calls:** `super.__init__`, `super`

- **`clear_model`**(self) -> <_ast.Constant object at 0x000001D6B135CE80>
  - Reset *everything*: OR-Tools proto, Python registries, counters.
Called internally by import / clone....
  - **Calls:** `self.__dict__.update`, `self._variables.clear`, `self._constraints.clear`, `self._objectives.clear`, `_cp.CpModel`

**Usage Pattern:**
```python
from model\model import EnhancedCpModel
# Inherits from: _VariablesMixin, _ConstraintsMixin, _ObjectivesMixin, _DebugMixin, _IOMixin, _cp.CpModel
instance = EnhancedCpModel()
```

##### Dependencies on Other Modules

- **`model\io.py`**: uses `_IOMixin`
- **`model\variables.py`**: uses `_VariablesMixin`
- **`model\constraints.py`**: uses `_ConstraintsMixin`
- **`model\objectives.py`**: uses `_ObjectivesMixin`
- **`model\debug.py`**: uses `_DebugMixin`

##### Called By

No external callers found.

##### Imports

**Third Party:**
- `from variables import _VariablesMixin`
- `from constraints import _ConstraintsMixin`
- `from objectives import _ObjectivesMixin`
- `from debug import _DebugMixin`
- `from io import _IOMixin`
- `ortools.sat.python.cp_model as _cp`

---

### `model\objectives.py`

## Module Purpose
This module provides a multi-objective management system for constraint programming models. It extends OR-Tools' CpModel to support registering multiple objectives and dynamically enabling/disabling them, while ensuring only one objective is active at a time.

## Public API

### ObjectiveInfo Class
**Usage pattern**: Internal data class, not typically used directly by end users
```python
# Created automatically by _ObjectivesMixin methods
obj_info = ObjectiveInfo("Minimize", linear_expr, "my_objective")
```

**Method signatures**:
- `__init__(objective_type: str, linear_expr: _cp.LinearExprT, name: Optional[str] = None) -> None`

**Dependencies**: ortools.sat.python.cp_model (as _cp)

### _ObjectivesMixin Class
**Usage pattern**: Inherit from this class to add multi-objective capabilities
```python
from ortools.sat.python import cp_model

class MyModel(_ObjectivesMixin, cp_model.CpModel):
    def __init__(self):
        super().__init__()
```

**Method signatures**:
- `Minimize(obj: _cp.LinearExprT, name: Optional[str] = None) -> None`
- `Maximize(obj: _cp.LinearExprT, name: Optional[str] = None) -> None`
- `enable_objective(name: str) -> None`
- `disable_objective(name: str) -> None`
- `get_enabled_objective() -> Optional[ObjectiveInfo]`

**Dependencies**: Inherits from `_cp.CpModel`, uses `ObjectiveInfo`

**Return types**:
- `Minimize`/`Maximize`: None (register objectives)
- `enable_objective`/`disable_objective`: None
- `get_enabled_objective`: ObjectiveInfo or None

## Cross-References

**Imports from other modules**:
- `from ortools.sat.python.cp_model import *` (as _cp) - Core constraint programming functionality
- `from typing import List, Optional, Sequence` - Type annotations

**Exports to other modules**:
- `ObjectiveInfo` - Data container for objective metadata
- `_ObjectivesMixin` - Main functionality for multi-objective management

**Potential issues**:
- The mixin assumes proper multiple inheritance setup
- No explicit error handling for duplicate objective names
- Lazy initialization pattern may cause issues if `_ensure_objectives()` is called before objective registration

## Integration Examples

```python
from ortools.sat.python import cp_model
from model.objectives import _ObjectivesMixin

class MultiObjectiveModel(_ObjectivesMixin, cp_model.CpModel):
    def __init__(self):
        super().__init__()
        
    def solve_with_objective(self, objective_name: str):
        self.enable_objective(objective_name)
        solver = cp_model.CpSolver()
        status = solver.Solve(self)
        return status, solver

# Usage
model = MultiObjectiveModel()
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')

# Register multiple objectives
model.Minimize(x + y, "min_sum")
model.Maximize(x - y, "max_diff")

# Switch between objectives
model.enable_objective("min_sum")
status, solver = model.solve_with_objective("min_sum")
print(f"Min sum result: {solver.Value(x)}, {solver.Value(y)}")

model.enable_objective("max_diff")
status, solver = model.solve_with_objective("max_diff")
print(f"Max diff result: {solver.Value(x)}, {solver.Value(y)}")
```

```python
# Checking current objective
current_obj = model.get_enabled_objective()
if current_obj:
    print(f"Active objective: {current_obj.name} ({current_obj.objective_type})")
```

This module enables building constraint programming models that can easily switch between different optimization goals while maintaining the constraint satisfaction framework provided by OR-Tools.

#### Complete API Reference

##### Class: `ObjectiveInfo`

**Description:** Thin wrapper around an objective with metadata....

**Methods:**

- **`__init__`**(self, objective_type: <_ast.Name object at 0x000001D6B135C2E0>, linear_expr: <_ast.Attribute object at 0x000001D6B135C340>, name: <_ast.Subscript object at 0x000001D6B135CFA0> = <_ast.Constant object at 0x000001D6B135C700>) -> <_ast.Constant object at 0x000001D6B135CCA0>
  - **Calls:** `objective_type.lower`

**Usage Pattern:**
```python
from model\objectives import ObjectiveInfo
instance = ObjectiveInfo(objective_type, linear_expr, name)
```

##### Class: `_ObjectivesMixin`

**Description:** Objective creation and single-active-objective enforcement....

**Inherits from:** `_cp.CpModel`

**Methods:**

- **`Minimize`**(self, obj: <_ast.Attribute object at 0x000001D6B135CCD0>, name: <_ast.Subscript object at 0x000001D6B135CC40> = <_ast.Constant object at 0x000001D6B135C460>) -> <_ast.Constant object at 0x000001D6B137D9D0>
  - Register a minimization objective....
  - **Calls:** `self._ensure_objectives.append`, `super.Minimize`, `ObjectiveInfo`, `self._ensure_objectives`, `super`

- **`Maximize`**(self, obj: <_ast.Attribute object at 0x000001D6B137D580>, name: <_ast.Subscript object at 0x000001D6B137D520> = <_ast.Constant object at 0x000001D6B137D550>) -> <_ast.Constant object at 0x000001D6B137D130>
  - Register a maximization objective....
  - **Calls:** `self._ensure_objectives.append`, `super.Maximize`, `ObjectiveInfo`, `self._ensure_objectives`, `super`

- **`enable_objective`**(self, name: <_ast.Name object at 0x000001D6B137D430>) -> <_ast.Constant object at 0x000001D6B13E0310>
  - Enable exactly one objective by name; disable all others....
  - **Calls:** `self._ensure_objectives`, `ValueError`

- **`disable_objective`**(self, name: <_ast.Name object at 0x000001D6B13E0490>) -> <_ast.Constant object at 0x000001D6B13E0B50>
  - Disable one objective by name....
  - **Calls:** `self._ensure_objectives`, `ValueError`

- **`get_enabled_objective`**(self) -> <_ast.Subscript object at 0x000001D6B1291670>
  - Return the single enabled objective (or None)....
  - **Calls:** `len`, `RuntimeError`, `self._ensure_objectives`, `len`

- **`_ensure_objectives`**(self) -> <_ast.Subscript object at 0x000001D6B12911F0>
  - Lazy-initialise the objective registry for mix-in safety....
  - **Calls:** `hasattr`

**Usage Pattern:**
```python
from model\objectives import _ObjectivesMixin
# Inherits from: _cp.CpModel
instance = _ObjectivesMixin()
```

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `from typing import List`
- `from typing import Optional`
- `from typing import Sequence`

**Third Party:**
- `from __future__ import annotations`
- `ortools.sat.python.cp_model as _cp`

---

### `model\variables.py`

## Module Purpose
This module provides enhanced variable creation capabilities for OR-Tools CP-SAT models by wrapping OR-Tools variables with metadata tracking and name collision prevention. It extends the standard CpModel with additional functionality for managing variables and their metadata.

## Public API

### VariableInfo Class
**Usage pattern**: 
```python
from model.variables import VariableInfo

# Typically created internally by _VariablesMixin
var_info = VariableInfo(name, var_type, ortools_var, creation_args)
```

**Method signatures**:
- `__init__(self, name: str, var_type: str, ortools_var: Union[_cp.IntVar, _cp.IntervalVar], creation_args: Tuple[Any, ...]) -> None`

**Properties**:
- `name`: Variable identifier
- `var_type`: Type classification ("int", "bool", "interval", etc.)
- `ortools_var`: The underlying OR-Tools variable object
- `creation_args`: Arguments used to create the variable

### _VariablesMixin Class
**Usage pattern**: 
```python
from model.variables import _VariablesMixin
from ortools.sat.python.cp_model import CpModel

class MyModel(_VariablesMixin, CpModel):
    pass

model = MyModel()
```

**Method signatures**:

**Variable Creation Methods**:
- `NewIntVar(self, lb: int, ub: int, name: str) -> _cp.IntVar`
- `NewBoolVar(self, name: str) -> _cp.IntVar`
- `NewIntervalVar(self, start: _cp.LinearExprT, size: _cp.LinearExprT, end: _cp.LinearExprT, name: str) -> _cp.IntervalVar`
- `NewOptionalIntervalVar(self, start: _cp.LinearExprT, size: _cp.LinearExprT, end: _cp.LinearExprT, is_present: _cp.LiteralT, name: str) -> _cp.IntervalVar`
- `NewConstant(self, value: int) -> _cp.IntVar`

**Internal Utility Methods**:
- `_map_expr_to_new_model(self, expr: Any, var_mapping: Dict[str, Any]) -> Any`
- `_deep_map_expr(self, expr: Any, var_mapping: Dict[str, Any]) -> Any`
- `_ensure_variables(self) -> Dict[str, VariableInfo]`

**Dependencies**: Requires `ortools.sat.python.cp_model` as `_cp`

**Return types**: All variable creation methods return standard OR-Tools variable types (IntVar, IntervalVar) that can be used in constraints and expressions.

## Cross-References

### Imports from other modules
- `ortools.sat.python.cp_model` (aliased as `_cp`): Base CP-SAT model functionality
- `typing`: Type annotations (Tuple, Union, Dict, Any)
- `__future__`: Annotations for forward references

### Exports to other modules
- `VariableInfo`: Metadata container for variables
- `_VariablesMixin`: Mixin class for enhanced variable management
- Variable creation methods with name collision protection
- Expression mapping utilities for model transformation

### Potential issues
- **Circular dependencies**: The module imports from OR-Tools but doesn't appear to have circular dependencies within the project
- **Mixin usage**: Requires proper multiple inheritance with CpModel
- **Type safety**: Extensive use of `Any` type in expression mapping methods

## Integration Examples

### Basic Usage with Custom Model
```python
from model.variables import _VariablesMixin
from ortools.sat.python.cp_model import CpModel

class EnhancedModel(_VariablesMixin, CpModel):
    """Model with enhanced variable tracking"""
    
    def get_variable_info(self, name: str) -> VariableInfo:
        return self._ensure_variables().get(name)

# Create and use enhanced model
model = EnhancedModel()
x = model.NewIntVar(0, 10, "x")  # Automatically tracked
y = model.NewBoolVar("y")        # Name collision protected

# Access variable metadata
var_info = model.get_variable_info("x")
print(f"Variable {var_info.name} of type {var_info.var_type}")
```

### Creating Constants with Automatic Naming
```python
model = EnhancedModel()
const_5 = model.NewConstant(5)    # Creates "_const_5" or similar
const_10 = model.NewConstant(10)  # Creates "_const_10"

# Constants can be used in constraints like regular variables
model.Add(x + const_5 == const_10)
```

### Expression Mapping Between Models
```python
# Create two models
model1 = EnhancedModel()
model2 = EnhancedModel()

# Create variables in first model
var1 = model1.NewIntVar(0, 10, "var1")

# Create mapping and transfer expression
var_mapping = {"var1": model2.NewIntVar(0, 10, "var1_copy")}
expr = var1 + 5
mapped_expr = model2._map_expr_to_new_model(expr, var_mapping)

# mapped_expr now uses model2's variable
model2.Add(mapped_expr <= 15)
```

### Interval Variable Creation
```python
model = EnhancedModel()
start = model.NewIntVar(0, 100, "start_time")
duration = 5
end = model.NewIntVar(0, 100, "end_time")

# Create interval variable with automatic tracking
interval = model.NewIntervalVar(start, duration, end, "task_interval")

# Optional interval with presence condition
is_present = model.NewBoolVar("task_present")
optional_interval = model.NewOptionalIntervalVar(
    start, duration, end, is_present, "optional_task"
)
```

The module provides a robust foundation for building constraint programming models with enhanced variable management capabilities while maintaining full compatibility with standard OR-Tools functionality.

#### Complete API Reference

##### Class: `VariableInfo`

**Description:** Rich wrapper around OR-Tools variables with metadata....

**Methods:**

- **`__init__`**(self, name: <_ast.Name object at 0x000001D6B137D1F0>, var_type: <_ast.Name object at 0x000001D6B137DE20>, ortools_var: <_ast.Subscript object at 0x000001D6B137D760>, creation_args: <_ast.Subscript object at 0x000001D6B137DD60>) -> <_ast.Constant object at 0x000001D6B137DBB0>

**Usage Pattern:**
```python
from model\variables import VariableInfo
instance = VariableInfo(name, var_type, ortools_var, creation_args)
```

##### Class: `_VariablesMixin`

**Description:** All variable-creation helpers....

**Inherits from:** `_cp.CpModel`

**Methods:**

- **`NewIntVar`**(self, lb: <_ast.Name object at 0x000001D6B137DD30>, ub: <_ast.Name object at 0x000001D6B137DCD0>, name: <_ast.Name object at 0x000001D6B137DFD0>) -> <_ast.Attribute object at 0x000001D6B13E0490>
  - Create a new integer variable with bounds [lb, ub]....
  - **Calls:** `super.NewIntVar`, `VariableInfo`, `getattr`, `ValueError`, `self._ensure_variables` (+1 more)

- **`NewBoolVar`**(self, name: <_ast.Name object at 0x000001D6B13E0910>) -> <_ast.Attribute object at 0x000001D6B135CD90>
  - Create a new boolean variable....
  - **Calls:** `super.NewBoolVar`, `VariableInfo`, `getattr`, `ValueError`, `self._ensure_variables` (+1 more)

- **`NewIntervalVar`**(self, start: <_ast.Attribute object at 0x000001D6B135CDF0>, size: <_ast.Attribute object at 0x000001D6B135C280>, end: <_ast.Attribute object at 0x000001D6B135C2E0>, name: <_ast.Name object at 0x000001D6B135C7F0>) -> <_ast.Attribute object at 0x000001D6B1291280>
  - Create a new interval variable....
  - **Calls:** `super.NewIntervalVar`, `VariableInfo`, `getattr`, `ValueError`, `self._ensure_variables` (+1 more)

- **`NewOptionalIntervalVar`**(self, start: <_ast.Attribute object at 0x000001D6B1291940>, size: <_ast.Attribute object at 0x000001D6B12918B0>, end: <_ast.Attribute object at 0x000001D6B1291D30>, is_present: <_ast.Attribute object at 0x000001D6B1291820>, name: <_ast.Name object at 0x000001D6B1291040>) -> <_ast.Attribute object at 0x000001D6B11F6EE0>
  - Create a new optional interval variable....
  - **Calls:** `super.NewOptionalIntervalVar`, `VariableInfo`, `getattr`, `ValueError`, `self._ensure_variables` (+1 more)

- **`NewConstant`**(self, value: <_ast.Name object at 0x000001D6B11F6940>) -> <_ast.Attribute object at 0x000001D6B15E6F40>
  - Create a new constant with a unique name....
  - **Calls:** `self._ensure_variables`, `super.NewConstant`, `VariableInfo`, `super.get_variable_by_name`, `super` (+1 more)

- **`_map_expr_to_new_model`**(self, expr: <_ast.Name object at 0x000001D6B15E30D0>, var_mapping: <_ast.Subscript object at 0x000001D6B15E3160>) -> <_ast.Name object at 0x000001D6B15E3FD0>
  - Return expr with all variables remapped via var_mapping....
  - **Calls:** `isinstance`, `hasattr`, `self._deep_map_expr`, `hasattr`, `expr.Name` (+6 more)

- **`_deep_map_expr`**(self, expr: <_ast.Name object at 0x000001D6B15E8070>, var_mapping: <_ast.Subscript object at 0x000001D6B15E80D0>) -> <_ast.Name object at 0x000001D6B15DB340>
  - Recursively rebuild linear expressions with remapped variables....
  - **Calls:** `isinstance`, `hasattr`, `hasattr`, `expr.IsConstant`, `int` (+4 more)

- **`_ensure_variables`**(self) -> <_ast.Subscript object at 0x000001D6B15DB880>
  - Lazy-initialise the variable registry for mix-in safety....
  - **Calls:** `hasattr`

**Usage Pattern:**
```python
from model\variables import _VariablesMixin
# Inherits from: _cp.CpModel
instance = _VariablesMixin()
```

##### Called By

No external callers found.

##### Imports

**Standard Library:**
- `from typing import Tuple`
- `from typing import Union`
- `from typing import Dict`
- `from typing import Any`

**Third Party:**
- `from __future__ import annotations`
- `ortools.sat.python.cp_model as _cp`

---

