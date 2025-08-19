# Repository Context

Generated on: La fecha actual es: 19/08/2025 
Escriba la nueva fecha: (dd-mm-aa)
Files analyzed: 8

## Repository Architecture Overview

**Core Purpose:** This repository implements an enhanced wrapper framework around Google OR-Tools' CP-SAT constraint programming solver, providing advanced constraint management, debugging capabilities, serialization, and metadata tracking while maintaining full compatibility with the underlying OR-Tools API.

**Architectural Style:** The architecture follows a **layered wrapper pattern** with a **monolithic core** that extends the base CpModel functionality. It uses a **mixin-based composition** approach where specialized functionality is separated into focused modules that enhance the core model class.

**Key Components:**
- **Core Model** (`model.py`): The central wrapper class that extends CpModel and integrates all functionality
- **Constraint Management** (`constraints.py`): Handles constraint registration, tagging, and metadata tracking
- **Variable Management** (`variables.py`): Provides enhanced variable creation with metadata preservation
- **Constructor Tools** (`constructor_tools.py`, `constructors.py`): Factory methods for creating common constraint patterns and boolean indicators
- **Debug Utilities** (`debug.py`): Implements infeasibility analysis through Minimal Infeasible Subset detection
- **Serialization** (`io.py`): Handles complete model persistence and restoration with metadata
- **Objective Management** (`objectives.py`): Manages multiple optimization objectives with metadata tracking

**Data Flow:** The system follows a **builder pattern** flow: Users create variables through the enhanced wrapper, define constraints using constructor tools, register objectives, and then serialize or solve the model. Constraints and variables maintain metadata throughout their lifecycle, and the debugging layer can analyze failed models to identify root causes of infeasibility.

**Integration Points:** All components integrate through the central `Model` class which inherits from CpModel and mixes in functionality from various modules. The constraint system uses a registration mechanism (`_register_constraint`, `_ensure_constraints`) to track metadata. The serialization system coordinates with all other components to persist and restore complete model state including variables, constraints, objectives, and their associated metadata.

**Design Patterns:**
- **Wrapper/Decorator Pattern:** Extensive wrapping of OR-Tools primitives to add metadata
- **Factory Method Pattern:** Constructor tools provide factory methods for common constraint types
- **Builder Pattern:** Fluent interface for constraint construction
- **Registry Pattern:** Constraint and objective tracking through registration systems
- **Mixin Pattern:** Modular functionality added through class composition
- **Strategy Pattern:** Multiple objective handling with interchangeable optimization strategies

The architecture emphasizes **metadata preservation** throughout the modeling lifecycle, enabling advanced debugging, serialization, and introspection capabilities that are not available in the base OR-Tools implementation.

## Most Referenced Symbols

- `super` (43 references)
- `map_arg` (43 references)
- `model.add_constraint_tags` (41 references)
- `isinstance` (34 references)
- `tuple` (29 references)
- `self._register_constraint` (25 references)
- `len` (23 references)
- `OnlyEnforceIf` (20 references)
- `getattr` (19 references)
- `model.Add` (19 references)

## Detailed File Analysis

### `model\constraints.py`

**Complexity Score:** 50 | **Lines:** 571 | **Size:** 20442 bytes

**Primary Purpose:** This file implements a comprehensive constraint management system for CP-SAT modeling, providing a wrapper around OR-Tools' constraint programming functionality with enhanced metadata tracking and constraint recreation capabilities.

**Key Responsibilities:**
- Provides a mixin class (`_ConstraintsMixin`) that extends OR-Tools' `CpModel` with constraint creation methods
- Manages constraint metadata through `ConstraintInfo` objects for debugging and enable/disable functionality
- Implements constraint proxies (`_ConstraintProxy`) to capture user configuration calls like `OnlyEnforceIf` and `WithName`
- Supports constraint recreation in different models through `_recreate_constraint_in_model` with variable mapping
- Handles 25+ different constraint types including linear, boolean, circuit, automaton, and scheduling constraints
- Maintains enforcement literals and constraint registration for advanced constraint management

**Architecture Role:** This serves as a core middleware component between the user-facing API and the underlying OR-Tools CP-SAT engine. It acts as an enhancement layer that adds metadata tracking, constraint recreation capabilities, and a unified interface for constraint management while maintaining compatibility with the base OR-Tools functionality.

**Notable Patterns:**
- **Decorator Pattern:** The `_ConstraintProxy` wraps OR-Tools constraints to intercept configuration calls
- **Mixin Pattern:** `_ConstraintsMixin` extends OR-Tools' `CpModel` without direct inheritance
- **Registry Pattern:** Maintains a registry of constraints with metadata for management and recreation
- **Factory Pattern:** Multiple `Add*` methods act as factories for different constraint types
- **Visitor Pattern:** The `map_arg` function recursively maps variables through nested structures during constraint recreation
- **Proxy Pattern:** Constraint methods return proxies that capture user configuration before creating actual constraints

#### Module Interactions

**Internal Dependencies:**
- Depends on `model\debug.py` (likely for logging/debugging utilities)
- The number "2" suggests this is the second dependency, possibly indicating a specific import pattern or version requirement

**External Dependencies:**
- **ortools.sat.python.cp_model**: Primary dependency for constraint programming and optimization (Google OR-Tools CP-SAT solver)
- **typing**: Extensive use of type hints for function signatures and data structures
- **__future__**: Likely for Python version compatibility features

**Integration Patterns:**
- **Constraint Definition Hub**: Acts as a central module for defining optimization constraints using OR-Tools
- **Type-Safe Interface**: Heavy typing suggests it provides well-defined interfaces for other modules to interact with constraint logic
- **Debug Integration**: Integrates with debugging utilities to log constraint violations or optimization progress
- **Solver Abstraction**: Likely wraps OR-Tools functionality with application-specific constraints and validation logic
- **Configuration-Driven**: Probably designed to accept constraint configurations from other modules rather than hardcoding business rules

#### Classes

- **`ConstraintInfo`** (line 17)
  - Wrapper around a single constraint with metadata for debugging/enable/disable.
  - Methods: __init__

- **`_ConstraintProxy`** (line 48)
  - Proxy returned by Add*/etc. to capture user calls like OnlyEnforceIf.
  - Methods: __init__, OnlyEnforceIf, WithName, __getattr__

- **`_ConstraintsMixin`** (line 71) extends _cp.CpModel
  - All constraint-creation helpers.
  - Methods: Add, AddLinearConstraint, AddLinearExpressionInDomain, AddAllDifferent, AddElement, ...+23 more

#### Functions

- **`__init__`** (line 30) - Args: self, name, original_args, constraint_type, ortools_ct, enable_var
  - 
- **`__init__`** (line 51) - Args: self, ct, info
  - 
- **`OnlyEnforceIf`** (line 55) - Args: self, lits
  - 
- **`WithName`** (line 62) - Args: self, name
  - 
- **`__getattr__`** (line 67) - Args: self, attr
  - 
- **`Add`** (line 78) - Args: self, ct, name
  - Add a generic constraint.
- **`AddLinearConstraint`** (line 87) - Args: self, linear_expr, lb, ub, name
  - 
- **`AddLinearExpressionInDomain`** (line 101) - Args: self, linear_expr, domain, name
  - 
- **`AddAllDifferent`** (line 114) - Args: self, variables, name
  - 
- **`AddElement`** (line 126) - Args: self, index, variables, target, name
  - 
- **`AddCircuit`** (line 140) - Args: self, arcs, name
  - 
- **`AddMultipleCircuit`** (line 152) - Args: self, arcs, name
  - 
- **`AddAllowedAssignments`** (line 164) - Args: self, variables, tuples_list, name
  - 
- **`AddForbiddenAssignments`** (line 177) - Args: self, variables, tuples_list, name
  - 
- **`AddAutomaton`** (line 190) - Args: self, transition_variables, starting_state, final_states, transition_triples, name
  - 
- **`AddInverse`** (line 212) - Args: self, variables, inverse_variables, name
  - 
- **`AddReservoirConstraint`** (line 225) - Args: self, times, level_changes, min_level, max_level, name
  - 
- **`AddMinEquality`** (line 243) - Args: self, target, variables, name
  - 
- **`AddMaxEquality`** (line 256) - Args: self, target, variables, name
  - 
- **`AddMultiplicationEquality`** (line 269) - Args: self, target, variables, name
  - 
- **`AddDivisionEquality`** (line 282) - Args: self, target, numerator, denominator, name
  - 
- **`AddAbsEquality`** (line 296) - Args: self, target, variable, name
  - 
- **`AddModuloEquality`** (line 309) - Args: self, target, variable, modulo, name
  - 
- **`AddBoolOr`** (line 324) - Args: self, literals, name
  - 
- **`AddBoolAnd`** (line 336) - Args: self, literals, name
  - 
- **`AddBoolXor`** (line 348) - Args: self, literals, name
  - 
- **`AddImplication`** (line 360) - Args: self, a, b, name
  - 
- **`AddNoOverlap`** (line 374) - Args: self, intervals, name
  - 
- **`AddNoOverlap2D`** (line 386) - Args: self, x_intervals, y_intervals, name
  - 
- **`AddCumulative`** (line 399) - Args: self, intervals, demands, capacity, name
  - 
- **`_register_constraint`** (line 416) - Args: self, constraint, original_args, constraint_type, name, enforce_enable_var
  - Register a constraint with full metadata.
- **`_recreate_constraint_in_model`** (line 448) - Args: self, model, constraint_info, var_mapping
  - Recreate a constraint in *model* using *var_mapping*.
Replays user-enforcement literals but **not** enable_var enforcement
(that is handled externally...
- **`map_arg`** (line 462) - Args: arg
  - Maps variables or nested structures.
- **`_ensure_constraints`** (line 567) - Args: self
  - Lazy-initialise the constraint registry for mix-in safety.
#### Internal Dependencies

- `model\debug.py` (2 references)

---

### `model\constructor_tools.py`

**Complexity Score:** 62 | **Lines:** 774 | **Size:** 31960 bytes

**Primary Purpose:** This file provides a comprehensive set of constraint programming tools for creating boolean indicator variables that represent various logical and relational conditions within a CP-SAT model, enabling complex constraint formulations.

**Key Responsibilities:**
- Creating boolean indicator variables for relational comparisons (≥, ≤, >, <, =, ≠)
- Implementing logical operations (AND, OR) as boolean variables
- Handling interval relationships (point containment, overlap, containment)
- Creating conditional min/max variables subject to boolean constraints
- Implementing complex logical operations with constraint dependencies
- Managing constraint tagging and metadata for debugging and analysis

**Architecture Role:** This serves as a core utility module in a constraint programming system, providing foundational building blocks for higher-level constraint formulations. It acts as an abstraction layer between the raw CP-SAT API and domain-specific constraint logic, making complex constraint expressions more manageable and reusable.

**Notable Patterns:**
- **Builder Pattern:** Each function constructs complex constraint relationships through a consistent interface
- **Indicator Variable Pattern:** Extensive use of boolean variables to represent complex conditions
- **Constraint Tagging:** Systematic approach to labeling constraints for debugging and analysis
- **Template Method Pattern:** Consistent structure across similar constraint types (e.g., all comparison functions follow the same two-constraint pattern)
- **Domain Abstraction:** Handles both primitive values and interval objects through type checking and adaptation
- **Composite Pattern:** Functions that combine multiple simpler constraints into complex logical expressions

#### Module Interactions

**Internal Dependencies:**
- Directly depends on `model\constructors.py` (line 18), suggesting it extends or works closely with core constructor functionality
- Part of the `model` package, indicating it provides specialized tooling for model construction

**External Dependencies:**
- **ortools.sat.python**: Primary dependency - provides constraint programming and SAT solver capabilities from Google OR-Tools
- **typing**: Used extensively for type annotations, indicating strong type safety focus

**Integration Patterns:**
- Acts as a specialized utility layer for model construction, likely providing advanced constraint formulation or optimization tools
- Integrates with OR-Tools SAT solver for constraint satisfaction problems
- Serves as an extension point for the core constructors, providing additional construction capabilities
- Pattern suggests this module handles complex constraint logic while `constructors.py` manages basic model structure

#### Functions

- **`NewGreaterOrEqualBoolVar`** (line 10) - Args: model, variable, threshold, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is greater or equal to a given threshold, and 0 when it is...
- **`NewLessOrEqualBoolVar`** (line 55) - Args: model, variable, threshold, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is less than or equal to a given threshold, and 0 when it ...
- **`NewGreaterBoolVar`** (line 100) - Args: model, variable, threshold, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is strictly greater than a given threshold, and 0 when it ...
- **`NewLessBoolVar`** (line 145) - Args: model, variable, threshold, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is strictly less than a given threshold, and 0 when it is ...
- **`NewEqualBoolVar`** (line 190) - Args: model, variable, value, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is equal to a given value, and 0 when it is not equal to t...
- **`NewNotEqualBoolVar`** (line 238) - Args: model, variable, value, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
is not equal to a given value, and 0 when it is equal to t...
- **`NewAndBoolVar`** (line 283) - Args: model, variables, name
  - Creates a boolean variable in a CP-SAT model that is 1 when all the specified boolean variables 
in the given list are true, and 0 when at least one o...
- **`NewOrBoolVar`** (line 326) - Args: model, variables, name
  - Creates a boolean variable in a CP-SAT model that is 1 when at least one of the specified boolean variables 
in the given list is true, and 0 when all...
- **`NewPointInIntervalBoolVar`** (line 369) - Args: model, variable, interval, name
  - Creates a boolean variable in a CP-SAT model that is 1 when a specified integer variable lies within a given interval, and 0 otherwise.

Parameters:
-...
- **`NewOverlapBoolVar`** (line 422) - Args: model, interval1, interval2, name
  - Creates a boolean variable in a CP-SAT model that is 1 when two intervals overlap, and 0 when they do not.

Parameters:
- model (EnhancedCpModel): The...
- **`NewContainedInBoolVar`** (line 485) - Args: model, interval1, interval2, name
  - Creates a boolean variable in a CP-SAT model that is 1 when the first interval is contained in the second interval, and 0 when it is not.

Parameters:...
- **`NewMinSubjectToBools`** (line 537) - Args: model, values, bools, name, return_bool_markers
  - Creates a new integer variable representing the minimum value among a list of integer variables, subject to boolean conditions.

Parameters:
- model (...
- **`NewMaxSubjectToBools`** (line 612) - Args: model, values, bools, name, return_bool_markers
  - Creates a new integer variable representing the maximum value among a list of integer variables, subject to boolean conditions.

Parameters:
- model (...
- **`NewOrSubjectToBools`** (line 687) - Args: model, check_bools, constraint_bools, name
  - Creates a boolean variable representing the logical OR operation applied to pairs of boolean variables subject to additional constraint boolean variab...
- **`NewAndSubjectToBools`** (line 726) - Args: model, check_bools, constraint_bools, name
  - Creates a boolean variable representing the logical AND operation applied to pairs of boolean variables subject to additional constraint boolean varia...
#### Internal Dependencies

- `model\constructors.py` (18 references)

---

### `model\constructors.py`

**Complexity Score:** 17 | **Lines:** 151 | **Size:** 4438 bytes

**Primary Purpose:** This file provides a mixin class that extends OR-Tools CP-SAT models with helper methods for creating common constraint patterns and boolean variables, keeping the implementation separate from the core model class.

**Key Responsibilities:**
- Creates boolean variables representing comparison constraints (greater/less than, equal/not equal)
- Implements logical operations (AND, OR) as boolean variables
- Handles interval-related constraints (point in interval, overlap, containment)
- Provides subject-to constraints that conditionally apply based on boolean markers
- Acts as a facade pattern for underlying constructor functions

**Architecture Role:** This is a utility module that enhances the core OR-Tools CP-SAT functionality. It serves as an extension mechanism (via mixin) rather than a core component, allowing for cleaner separation of concerns while providing syntactic sugar for common constraint patterns.

**Notable Patterns:**
- **Mixin Pattern:** The `EnhancedConstructorsMixin` class is designed to be inherited by model classes to add functionality
- **Facade Pattern:** Wraps multiple standalone constructor functions into a unified interface
- **Consistent API:** All methods follow similar signatures with `self`, input parameters, and `name` for the resulting variable
- **Type Hints:** Uses typing annotations for better code documentation and IDE support
- **External Implementation:** The actual logic is implemented elsewhere (imported functions), keeping this as a clean interface layer

#### Module Interactions

**Internal Dependencies:**
- Strong dependency on `model\constructor_tools.py` (15 imports), indicating this module likely provides core utility functions and base classes for constructor operations
- Part of a constructor-focused subsystem within the model layer

**External Dependencies:**
- **ortools.sat.python**: Primary dependency for constraint programming and optimization (likely for scheduling, resource allocation, or combinatorial problems)
- **typing**: Extensive type hinting usage suggests strong emphasis on type safety and interface definition

**Integration Patterns:**
- **Utility Consumer**: Heavily relies on constructor_tools for shared functionality and abstractions
- **OR-Tools Integration Layer**: Acts as a bridge between the application's constructor concepts and Google's OR-Tools constraint programming framework
- **Type-Driven Interface**: Extensive typing suggests this module defines clear contracts for constructor operations that other modules must adhere to
- **Specialized Constructor Factory**: Likely provides various constructor implementations that leverage OR-Tools for complex constraint satisfaction problems

#### Classes

- **`EnhancedConstructorsMixin`** (line 22)
  - Mix-in that exposes helper constructors as *methods* on EnhancedCpModel
while keeping the implementation external.
  - Methods: NewGreaterOrEqualBoolVar, NewLessOrEqualBoolVar, NewGreaterBoolVar, NewLessBoolVar, NewEqualBoolVar, ...+10 more

#### Functions

- **`NewGreaterOrEqualBoolVar`** (line 29) - Args: self, variable, threshold, name
  - 
- **`NewLessOrEqualBoolVar`** (line 37) - Args: self, variable, threshold, name
  - 
- **`NewGreaterBoolVar`** (line 45) - Args: self, variable, threshold, name
  - 
- **`NewLessBoolVar`** (line 53) - Args: self, variable, threshold, name
  - 
- **`NewEqualBoolVar`** (line 61) - Args: self, variable, value, name
  - 
- **`NewNotEqualBoolVar`** (line 69) - Args: self, variable, value, name
  - 
- **`NewAndBoolVar`** (line 78) - Args: self, variables, name
  - 
- **`NewOrBoolVar`** (line 85) - Args: self, variables, name
  - 
- **`NewPointInIntervalBoolVar`** (line 93) - Args: self, variable, interval, name
  - 
- **`NewOverlapBoolVar`** (line 101) - Args: self, interval1, interval2, name
  - 
- **`NewContainedInBoolVar`** (line 109) - Args: self, interval1, interval2, name
  - 
- **`NewMinSubjectToBools`** (line 118) - Args: self, values, bools, name, return_bool_markers
  - 
- **`NewMaxSubjectToBools`** (line 127) - Args: self, values, bools, name, return_bool_markers
  - 
- **`NewOrSubjectToBools`** (line 137) - Args: self, check_bools, constraint_bools, name
  - 
- **`NewAndSubjectToBools`** (line 145) - Args: self, check_bools, constraint_bools, name
  - 
#### Internal Dependencies

- `model\constructor_tools.py` (15 references)

---

### `model\debug.py`

**Complexity Score:** 28 | **Lines:** 247 | **Size:** 8990 bytes

**Primary Purpose:** This file provides debugging and introspection utilities for constraint programming models, specifically focused on identifying infeasibility causes through Minimal Infeasible Subset (MIS) detection and model analysis.

**Key Responsibilities:**
- Debug infeasible models by finding minimal sets of constraints to disable
- Provide comprehensive model summaries and validation diagnostics
- Enable constraint introspection by type, tag, and status (enabled/disabled)
- Create relaxed model copies by randomly disabling constraints
- Generate subset models with specific constraint configurations
- Manage constraint state tracking and validation

**Architecture Role:** This serves as a diagnostic utility layer for constraint programming models, extending the base CP model functionality with debugging capabilities. It acts as a mixin class (`_DebugMixin`) that enhances model objects with introspection and debugging features without modifying core solving logic.

**Notable Patterns:**
- **Mixin Pattern:** Uses inheritance to extend `_cp.CpModel` with debugging capabilities while maintaining separation of concerns
- **Constraint State Management:** Implements sophisticated tracking of enabled/disabled constraints through boolean variables and state dictionaries
- **Model Cloning:** Uses `clone()` pattern to create modified copies for debugging without affecting original models
- **Lazy Evaluation:** `_ensure_constraints()` method suggests deferred constraint processing or validation
- **Visitor Pattern Elements:** Methods like `get_constraints_by_type()` and `get_constraints_by_tag()` provide filtered views of model components
- **Diagnostic Interface:** Unified API for model analysis through summary, validation, and constraint information methods

#### Module Interactions

**Internal Dependencies:**
- Directly depends on `model\constraints.py` (with a weight of 10, indicating strong coupling)
- Likely uses constraint definitions and utilities from the constraints module
- Part of the model layer, suggesting integration with other model components

**External Dependencies:**
- **ortools.sat.python.cp_model**: Core dependency for constraint programming and SAT solving
- **random**: Used for generating test data or randomizing debug scenarios
- **typing**: Extensive type hinting usage (imported 5 times), indicating strong type safety focus

**Integration Patterns:**
- **Debug/Testing Utility**: Functions as a debugging module for constraint models
- **Model Validation**: Likely validates constraint implementations from `constraints.py`
- **Solution Analysis**: Probably analyzes and verifies CP-SAT model solutions
- **Development Tool**: Serves as a development aid rather than production code, providing verification and diagnostic capabilities for the constraint programming system

#### Classes

- **`_DebugMixin`** (line 22) extends _cp.CpModel
  - Introspection, MIS, relaxation & subset utilities.
  - Methods: debug_infeasible, summary, validate_model, get_constraint_names, get_variable_names, ...+8 more

#### Functions

- **`debug_infeasible`** (line 28) - Args: self, solver
  - Find a minimal set of constraints to disable to make the model feasible.

Returns
-------
dict with keys:
    status, feasible, disabled_constraints, ...
- **`summary`** (line 126) - Args: self
  - One-stop overview of the model.
- **`validate_model`** (line 152) - Args: self
  - Basic diagnostics: disabled constraints, multiple objectives, etc.
- **`get_constraint_names`** (line 174) - Args: self
  - 
- **`get_variable_names`** (line 177) - Args: self
  - 
- **`get_constraint_info`** (line 180) - Args: self, name
  - 
- **`get_constraints_by_type`** (line 185) - Args: self, constraint_type
  - 
- **`get_constraints_by_tag`** (line 192) - Args: self, tag
  - 
- **`get_enabled_constraints`** (line 197) - Args: self
  - 
- **`get_disabled_constraints`** (line 202) - Args: self
  - 
- **`create_relaxed_copy`** (line 210) - Args: self, relaxation_factor
  - Return a relaxed copy by randomly disabling `relaxation_factor`
fraction of currently-enabled constraints.
- **`create_subset_copy`** (line 229) - Args: self, constraint_names
  - Return a copy with *only* the listed constraints enabled.
- **`_ensure_constraints`** (line 244) - Args: self
  - 
#### Internal Dependencies

- `model\constraints.py` (10 references)

---

### `model\io.py`

**Complexity Score:** 18 | **Lines:** 296 | **Size:** 12670 bytes

**Primary Purpose:** This file provides serialization/deserialization capabilities for an enhanced CP-SAT model, allowing complete model persistence to disk and restoration with full metadata preservation.

**Key Responsibilities:**
- Exporting complete model state (proto + metadata) to ZIP archives containing both binary and JSON data
- Importing and reconstructing models from saved files with full object consistency
- Converting OR-Tools objects to JSON-serializable primitives and back
- Creating clean solving models with only enabled constraints and objectives
- Managing variable mapping and expression translation between model instances

**Architecture Role:** Serves as a persistence layer and I/O utility module for the enhanced CP-SAT modeling system. It acts as a bridge between the high-level modeling interface and the underlying OR-Tools infrastructure, enabling model checkpointing, sharing, and debugging capabilities.

**Notable Patterns:**
- **Composite Pattern:** Handles serialization/deserialization of complex expression trees through recursive processing
- **Adapter Pattern:** Translates between OR-Tools native objects and JSON-serializable representations
- **Factory Pattern:** Recreates constraints and objectives dynamically using metadata and reflection
- **Strategy Pattern:** Different serialization/deserialization methods for various OR-Tools object types
- **Metadata Preservation:** Maintains Python-side consistency through comprehensive metadata storage alongside the raw proto

#### Module Interactions

**Internal Dependencies:** None - This module appears to be a standalone utility for model I/O operations with no dependencies on other files in the codebase.

**External Dependencies:**
- `ortools.sat.python.cp_model` & `ortools.sat`: Google OR-Tools constraint programming library for loading and working with CP-SAT models
- `json`: For serializing/deserializing model data
- `zipfile`: For compressed model storage/retrieval
- `typing`: For type annotations and static type checking

**Integration Patterns:** This module serves as a **serialization facade** that encapsulates model persistence logic. It likely provides:
- Save/load functionality for OR-Tools CP-SAT models with metadata
- ZIP-based packaging of models with associated configuration/data
- JSON serialization for human-readable model representation
- Type-safe interfaces for model exchange between components

The module acts as a **boundary object** between the optimization engine and storage systems, abstracting the complexity of model serialization while maintaining type safety through extensive typing annotations.

#### Classes

- **`_IOMixin`** (line 21) extends _cp.CpModel
  - Model persistence helpers.
  - Methods: export_to_file, import_from_file, _serialize_arg, _deserialize_arg, _create_solving_model

#### Functions

- **`export_to_file`** (line 27) - Args: self, filename
  - Persist the *entire* EnhancedCpModel (proto + metadata) to disk.

The file is a ZIP archive with two entries:
    model.pb   – raw OR-Tools proto
    ...
- **`import_from_file`** (line 82) - Args: self, filename
  - Load a model from a saved file, restoring OR-Tools proto and metadata.
Re-creates constraints to guarantee Python-side consistency.
- **`_serialize_arg`** (line 195) - Args: self, arg
  - Convert OR-Tools objects to JSON-serialisable primitives.
- **`_deserialize_arg`** (line 220) - Args: self, serialized_arg, var_mapping
  - Re-hydrate primitives back to OR-Tools objects.
- **`_create_solving_model`** (line 242) - Args: self
  - Build a *clean* CpModel with only enabled constraints & objectives.
Used by export_to_file and debug_infeasible.
---

### `model\model.py`

**Complexity Score:** 7 | **Lines:** 42 | **Size:** 1259 bytes

**Primary Purpose:** This file implements an enhanced wrapper class for Google OR-Tools' CP-SAT model that extends the base CpModel with additional functionality for debugging, cloning, and I/O operations.

**Key Responsibilities:**
- Provides a drop-in replacement for `CpModel` with extended capabilities
- Combines multiple mixins for variables, constraints, objectives, debugging, and I/O functionality
- Manages internal state clearing through the `clear_model()` method
- Maintains separate registries for variables, constraints, and objectives

**Architecture Role:** This serves as a core facade/adapter component that sits between the OR-Tools library and the application code, providing enhanced functionality while maintaining compatibility with the original API. It acts as a central coordination point that aggregates multiple specialized mixins.

**Notable Patterns:**
- **Multiple Inheritance/Mixin Pattern:** Uses multiple inheritance to combine functionality from specialized mixin classes (`_VariablesMixin`, `_ConstraintsMixin`, etc.)
- **Facade Pattern:** Provides a simplified interface to complex OR-Tools functionality while adding custom features
- **Registry Pattern:** Maintains separate internal registries (`_variables`, `_constraints`, `_objectives`) for tracking model components
- **Composition over Inheritance:** While using inheritance, it delegates to the underlying `_cp.CpModel` instance rather than replacing its functionality entirely

#### Module Interactions

**Internal Dependencies:**
- Directly imports `variables`, `constraints`, and `objectives` from the same `model` package, indicating tight coupling with these core modeling components
- Depends on `debug` and `io` modules, suggesting this is a central orchestration module that handles model construction, debugging, and input/output operations
- Serves as the main entry point that coordinates variable creation, constraint application, and objective definition

**External Dependencies:**
- `ortools.sat.python.cp_model`: Primary external dependency for constraint programming, providing the core CP-SAT solver capabilities
- This module acts as a wrapper/interface layer between the internal modeling components and Google's OR-Tools framework

**Integration Patterns:**
- **Orchestrator Pattern:** Acts as the central coordinator that brings together variables, constraints, and objectives to build complete optimization models
- **Facade Pattern:** Provides a simplified interface to the complex OR-Tools CP-SAT model, abstracting internal implementation details
- **Builder Pattern:** Likely constructs optimization models incrementally by aggregating components from the dependent modules
- This module serves as the main integration point where domain-specific modeling logic meets the general-purpose solver infrastructure

#### Classes

- **`EnhancedCpModel`** (line 9) extends _VariablesMixin, _ConstraintsMixin, _ObjectivesMixin, _DebugMixin, _IOMixin, _cp.CpModel
  - Drop-in replacement for CpModel with rich debugging, cloning and I/O.
  - Methods: __init__, clear_model

#### Functions

- **`__init__`** (line 19) - Args: self
  - 
- **`clear_model`** (line 27) - Args: self
  - Reset *everything*: OR-Tools proto, Python registries, counters.
Called internally by import / clone.
#### Internal Dependencies

- `model\constraints.py` (2 references)
- `model\variables.py` (1 references)
- `model\objectives.py` (1 references)

---

### `model\objectives.py`

**Complexity Score:** 18 | **Lines:** 89 | **Size:** 3318 bytes

**Primary Purpose:** This file provides objective management functionality for constraint programming models, allowing registration and control of multiple optimization objectives with metadata while enforcing single-active-objective constraints.

**Key Responsibilities:**
- Defines `ObjectiveInfo` class to wrap optimization objectives with metadata (type, expression, name)
- Implements `_ObjectivesMixin` that extends CP model capabilities with objective management
- Provides methods for registering minimization (`Minimize`) and maximization (`Maximize`) objectives
- Enables single-objective enforcement through `enable_objective`/`disable_objective` methods
- Maintains lazy-initialized objective registry for mix-in safety
- Validates objective state consistency (exactly one enabled objective)

**Architecture Role:** Acts as a core extension module that enhances the OR-Tools CP-SAT model with multi-objective management capabilities. It serves as a mixin that can be composed with the base `CpModel` to provide advanced objective handling without modifying the core library.

**Notable Patterns:**
- **Mixin Pattern:** `_ObjectivesMixin` extends base CP model functionality through inheritance while maintaining separation of concerns
- **Lazy Initialization:** Uses `_ensure_objectives()` to defer initialization until first use for mixin safety
- **Metadata Wrapper:** `ObjectiveInfo` follows the Decorator pattern to augment objectives with additional information
- **Single Responsibility:** Each method handles a specific aspect of objective management (registration, activation, querying)
- **Validation Logic:** Enforces business rules (exactly one enabled objective) with proper error handling

#### Module Interactions

**Internal Dependencies:** None - This module has no dependencies on other files in the codebase.

**External Dependencies:**
- `ortools.sat.python.cp_model`: Core dependency for constraint programming, providing the CP-SAT solver and modeling objects (variables, constraints, objective functions)
- `typing`: Used for type annotations to improve code clarity and maintainability

**Integration Patterns:**
This module serves as a pure utility/helper module focused on objective function construction. It likely:
- Provides factory functions or classes that create objective expressions for optimization problems
- Returns `cp_model` objective objects that can be directly set on CP-SAT model instances
- Follows a stateless pattern - functions take model and parameters, return objective expressions
- Acts as a standalone component that can be called by any optimization service needing objective definitions
- Enables separation of objective logic from core model building and solving infrastructure

#### Classes

- **`ObjectiveInfo`** (line 17)
  - Thin wrapper around an objective with metadata.
  - Methods: __init__

- **`_ObjectivesMixin`** (line 33) extends _cp.CpModel
  - Objective creation and single-active-objective enforcement.
  - Methods: Minimize, Maximize, enable_objective, disable_objective, get_enabled_objective, ...+1 more

#### Functions

- **`__init__`** (line 21) - Args: self, objective_type, linear_expr, name
  - 
- **`Minimize`** (line 39) - Args: self, obj, name
  - Register a minimization objective.
- **`Maximize`** (line 44) - Args: self, obj, name
  - Register a maximization objective.
- **`enable_objective`** (line 52) - Args: self, name
  - Enable exactly one objective by name; disable all others.
- **`disable_objective`** (line 65) - Args: self, name
  - Disable one objective by name.
- **`get_enabled_objective`** (line 73) - Args: self
  - Return the single enabled objective (or None).
- **`_ensure_objectives`** (line 85) - Args: self
  - Lazy-initialise the objective registry for mix-in safety.
---

### `model\variables.py`

**Complexity Score:** 24 | **Lines:** 165 | **Size:** 6161 bytes

**Primary Purpose:** This file provides a rich wrapper around OR-Tools CP-SAT variables with metadata tracking and enhanced variable creation capabilities, extending the base CpModel functionality.

**Key Responsibilities:**
- Creating and managing different types of constraint programming variables (IntVar, BoolVar, IntervalVar, OptionalIntervalVar)
- Maintaining variable metadata and registry through VariableInfo class
- Providing expression mapping utilities for variable remapping between models
- Ensuring proper variable initialization and mix-in safety
- Handling constant creation with unique naming

**Architecture Role:** This serves as a core extension module that enhances the OR-Tools CP-SAT model with additional functionality. It acts as a mixin class (_VariablesMixin) designed to be composed with the base CpModel, providing metadata tracking and advanced variable management capabilities.

**Notable Patterns:**
- **Mixin Pattern:** _VariablesMixin extends _cp.CpModel functionality without traditional inheritance
- **Lazy Initialization:** _ensure_variables() method for on-demand registry setup
- **Factory Methods:** Multiple New*Var methods following a consistent creation pattern
- **Visitor Pattern:** _deep_map_expr implements recursive expression traversal for variable remapping
- **Metadata Attachment:** VariableInfo class stores creation context alongside OR-Tools variables
- **Expression Processing:** Comprehensive handling of linear expressions with GetVar, GetVars, and WeightedSum operations

#### Module Interactions

**Internal Dependencies:** None - This file has no dependencies on other files within the codebase.

**External Dependencies:**
- `ortools.sat.python.cp_model`: Core dependency for constraint programming, providing the `CpModel` class for defining optimization models and variables
- `typing`: Used for type annotations (List, Optional, Union) to enhance code clarity and maintainability

**Integration Patterns:**
This module serves as a **variable factory/utility layer** for OR-Tools constraint programming. It likely:
- Creates and manages CP-SAT model variables with consistent configurations
- Provides typed wrapper functions around `model.NewIntVar()` and similar methods
- Standardizes variable creation patterns across the codebase
- Returns properly typed variable objects for use in constraint definitions elsewhere

The module acts as a **pure utility** - it doesn't maintain state but provides functions that other constraint/model building modules would call to instantiate variables consistently.

#### Classes

- **`VariableInfo`** (line 16)
  - Rich wrapper around OR-Tools variables with metadata.
  - Methods: __init__

- **`_VariablesMixin`** (line 33) extends _cp.CpModel
  - All variable-creation helpers.
  - Methods: NewIntVar, NewBoolVar, NewIntervalVar, NewOptionalIntervalVar, NewConstant, ...+3 more

#### Functions

- **`__init__`** (line 20) - Args: self, name, var_type, ortools_var, creation_args
  - 
- **`NewIntVar`** (line 39) - Args: self, lb, ub, name
  - Create a new integer variable with bounds [lb, ub].
- **`NewBoolVar`** (line 49) - Args: self, name
  - Create a new boolean variable.
- **`NewIntervalVar`** (line 59) - Args: self, start, size, end, name
  - Create a new interval variable.
- **`NewOptionalIntervalVar`** (line 78) - Args: self, start, size, end, is_present, name
  - Create a new optional interval variable.
- **`NewConstant`** (line 98) - Args: self, value
  - Create a new constant with a unique name.
- **`_map_expr_to_new_model`** (line 120) - Args: self, expr, var_mapping
  - Return expr with all variables remapped via var_mapping.
- **`_deep_map_expr`** (line 135) - Args: self, expr, var_mapping
  - Recursively rebuild linear expressions with remapped variables.
- **`_ensure_variables`** (line 161) - Args: self
  - Lazy-initialise the variable registry for mix-in safety.
---

