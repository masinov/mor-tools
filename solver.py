from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import time

# Assuming model.py is in the same directory or accessible in the python path
from model import EnhancedCpModel
from ortools.sat.python import cp_model
from ortools.sat import sat_parameters_pb2

class EnhancedCpSolver(cp_model.CpSolver):
    """
    Enhanced CpSolver that integrates with EnhancedCpModel.

    Features:
      - Model-aware tracking: Stores full EnhancedCpModel instances and metadata.
      - Rich solve history: Each record contains a replayable model clone and summary.
      - Advanced replay: Replay solves from history or create model clones for modification.
      - Integrated experiments: Methods for constraint necessity testing and tag-based solving.
      - Safe hinting: Uses the model's variable registry for reliable hint application.
      - Direct MIS integration: Wrapper for the model's debug_infeasible method.
    """

    def __init__(self):
        super().__init__()
        self._last_model: Optional[EnhancedCpModel] = None
        self._last_model_metadata: Optional[Dict[str, Any]] = None

        # Public history: each entry is a dict with detailed solve information
        self.solve_history: List[Dict[str, Any]] = []

        # Internal caches for "last" and auto-hinting
        self._last_solution_values: Dict[str, Union[int, float]] = {}
        self._auto_hint_enabled: bool = False
        self._auto_hint_source_index: Optional[int] = None  # None => default to "last"
        self._auto_hint_seed_values: Optional[Dict[str, Union[int, float]]] = None

    # --------------------------------
    # Configuration
    # --------------------------------

    def enable_auto_hint(self):
        """Enable seeding next solves from a record (defaults to 'last' if no source set)."""
        self._auto_hint_enabled = True

    def disable_auto_hint(self):
        """Disable auto-hinting."""
        self._auto_hint_enabled = False

    def set_auto_hint_source(self, index: Optional[int]):
        """
        Choose which history record to use as the auto-hint source.
        - index=None => fall back to 'last' record available.
        - index >= 0 => use that specific history index (if it exists at solve time).
        """
        self._auto_hint_source_index = index
        self._auto_hint_seed_values = None  # clear manual seed cache

    # --------------------------------
    # Main Solving Methods
    # --------------------------------

    def Solve(
        self,
        model: EnhancedCpModel,
        *,
        pre_relaxed: bool = False,
        relaxation_factor: float = 0.0
    ) -> int:
        """
        Solve the given EnhancedCpModel, tracking results and metadata in history.

        Args:
            model: The EnhancedCpModel instance to solve.
            pre_relaxed: If True, automatically create a relaxed copy of the model
                         before solving. Defaults to False.
            relaxation_factor: The fraction of constraints to disable if pre_relaxed
                               is True (0.0 to 1.0). Defaults to 0.0.

        Returns:
            The solver status code.

        Raises:
            TypeError: If the provided model is not an instance of EnhancedCpModel.
        """
        if not isinstance(model, EnhancedCpModel):
            raise TypeError("EnhancedCpSolver requires an EnhancedCpModel instance.")

        # Apply pre-relaxation if requested
        if pre_relaxed:
            model_to_solve = model.create_relaxed_copy(relaxation_factor)
        else:
            model_to_solve = model
            
        self._apply_auto_hints(model_to_solve)

        return self._internal_solve(model_to_solve)

    def SolveWithParameters(
        self,
        model: EnhancedCpModel,
        parameters: sat_parameters_pb2.SatParameters
    ) -> int:
        """
        Solve the model with explicit parameters, also tracked in history.
        """
        self.parameters.CopyFrom(parameters)
        return self.Solve(model)

    def _internal_solve(
        self,
        model: EnhancedCpModel,
        **solver_params
    ) -> int:
        """
        Internal core solving logic that performs the solve and records history.
        """
        # Snapshot current parameters to restore them later if they are modified
        original_params = self._snapshot_parameters()
        
        # Apply any temporary solver parameters
        for param_name, value in solver_params.items():
            if hasattr(self.parameters, param_name):
                setattr(self.parameters, param_name, value)

        # The actual solve call, respecting the model's enabled/disabled constraints
        solving_model = model._create_solving_model()
        start_time = time.time()
        status = super().Solve(solving_model)
        elapsed = time.time() - start_time
        
        # Restore original parameters
        self.parameters.CopyFrom(original_params)

        # Cache model and metadata
        self._last_model = model
        self._last_model_metadata = model.summary()

        # Collect variable values if a solution was found
        self._last_solution_values = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for name, var_info in model._variables.items():
                if not name.startswith('_'): # Exclude internal variables like _enable_
                    try:
                        # Use the variable from the original model for value extraction
                        self._last_solution_values[name] = self.Value(var_info.ortools_var)
                    except Exception:
                        # Silently ignore if variable is not accessible or error occurs
                        pass

        # Build and store the history record
        record: Dict[str, Any] = {
            "status": status,
            "status_name": self._status_name(status),
            "solve_time": elapsed,
            "objective_value": (
                self.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None
            ),
            "variables": dict(self._last_solution_values),
            "parameters": self._snapshot_parameters(),
            "timestamp": time.time(),
            "model_id": id(model),
            # New rich metadata fields
            "model_clone": model.clone(),
            "model_summary": self._last_model_metadata,
            "enabled_constraints": model.get_enabled_constraints(),
            "disabled_constraints": model.get_disabled_constraints(),
        }
        self.solve_history.append(record)

        return status

    # --------------------------------
    # Hinting Helpers
    # --------------------------------

    def _apply_hints_to_model(
        self,
        model: EnhancedCpModel,
        values: Dict[str, Union[int, float]]
    ) -> int:
        """Applies hints to a model using its variable registry."""
        hints_added = 0
        if not values:
            return 0
            
        for var_name, val in values.items():
            info = model._variables.get(var_name)
            # Only hint integer and boolean variables
            if info and info.var_type in ("IntVar", "BoolVar"):
                try:
                    model.AddHint(info.ortools_var, int(val))
                    hints_added += 1
                except Exception:
                    # Silently ignore if hint is invalid (e.g., out of domain)
                    pass
        return hints_added

    def _apply_auto_hints(self, model: EnhancedCpModel):
        """Internal helper to apply hints based on the current auto-hint configuration."""
        if not self._auto_hint_enabled:
            return

        hint_values = None
        # 1. Highest priority: explicit manual seed
        if self._auto_hint_seed_values:
            hint_values = self._auto_hint_seed_values
        else:
            # 2. From chosen history index
            src_idx = self._auto_hint_source_index
            if src_idx is not None and 0 <= src_idx < len(self.solve_history):
                hint_values = self.solve_history[src_idx].get("variables", {})
            # 3. Fallback: last solution
            elif self._last_solution_values:
                hint_values = self._last_solution_values
        
        if hint_values:
            self._apply_hints_to_model(model, hint_values)

    def set_manual_auto_hint_seed(self, values: Dict[str, Union[int, float]]):
        """
        Provide an explicit manual seed dict for auto-hinting (highest priority when enabled).
        Pass an empty dict or None to clear.
        """
        self._auto_hint_seed_values = dict(values) if values else None

    # --------------------------------
    # Replay and Model Creation Utilities
    # --------------------------------

    def replay_last(self, *, with_hints: bool = False) -> int:
        """
        Creates a clone of the last solved model and solves it again.

        Args:
            with_hints: If True, the solution values from the last solve are
                        applied as hints to the cloned model.

        Returns:
            The solver status of the replayed solve.
        """
        if self._last_model is None:
            raise ValueError("No last model to replay")

        model = self._last_model.clone()  # Preserves enable/disable & tags
        if with_hints:
            self._apply_auto_hints(model)
            
        return self._internal_solve(model)

    def replay_from_history(self, index: int, *, with_hints: bool = False) -> int:
        """
        Takes a model clone from a specific history record and re-solves it.

        Args:
            index: The index of the record in `solve_history`.
            with_hints: If True, the solution from that history record is
                        applied as a hint.

        Returns:
            The solver status of the replayed solve.
        """
        if not (0 <= index < len(self.solve_history)):
            raise IndexError("History index out of range.")
        
        record = self.solve_history[index]
        model: EnhancedCpModel = record["model_clone"].clone() # Clone again to not modify history
        
        if with_hints:
            vals = record.get("variables", {})
            if vals:
                self._apply_hints_to_model(model, vals)
                
        return self._internal_solve(model)

    def create_replay_model(self, index: int) -> EnhancedCpModel:
        """
        Returns a ready-to-solve clone of a model from a history entry
        without solving it, allowing for further modification.

        Args:
            index: The index of the record in `solve_history`.

        Returns:
            A new EnhancedCpModel instance cloned from the history record.
        """
        if not (0 <= index < len(self.solve_history)):
            raise IndexError("History index out of range.")
            
        return self.solve_history[index]["model_clone"].clone()

    # --------------------------------
    # Model Relaxation & Sub-setting Wrappers
    # --------------------------------

    def relaxed_copy(self, relaxation_factor: float = 0.1) -> EnhancedCpModel:
        """
        Creates a relaxed copy of the *last solved model*.
        
        Args:
            relaxation_factor: Fraction of constraints to disable (0.0 to 1.0).

        Returns:
            A new EnhancedCpModel with some constraints disabled.
        """
        if self._last_model is None:
            raise ValueError("Solve a model first to create a relaxed copy.")
        return self._last_model.create_relaxed_copy(relaxation_factor)

    def subset_copy(self, constraint_names: List[str]) -> EnhancedCpModel:
        """
        Creates a copy of the *last solved model* with only a subset of constraints enabled.

        Args:
            constraint_names: A list of constraint names to keep enabled.

        Returns:
            A new EnhancedCpModel with only the specified constraints enabled.
        """
        if self._last_model is None:
            raise ValueError("Solve a model first to create a subset copy.")
        return self._last_model.create_subset_copy(constraint_names)

    # --------------------------------
    # Experimentation and Debugging
    # --------------------------------

    def test_constraint_necessity(
        self,
        constraint_names: List[str],
        **solver_params
    ) -> Dict[str, Any]:
        """
        Tests if disabling each listed constraint makes an infeasible model feasible.
        This is run on the *last solved model*.

        Args:
            constraint_names: List of names of constraints to test.
            **solver_params: Solver parameters, e.g., `timeout_per_test=10.0`.

        Returns:
            A dictionary mapping each constraint name to its test result.
        """
        if self._last_model is None:
            raise ValueError("No model loaded. Solve a model first.")

        base_params = self._snapshot_parameters()
        timeout = solver_params.pop("timeout_per_test", 10.0)
        
        results = {}
        for name in constraint_names:
            if name not in self._last_model._constraints:
                results[name] = {"status": "NOT_FOUND", "error": "Constraint not in model."}
                continue

            test_model = self._last_model.clone()
            test_model.disable_constraint(name)
            
            # Use a temporary solver to avoid polluting history
            temp_solver = cp_model.CpSolver()
            temp_solver.parameters.CopyFrom(base_params)
            temp_solver.parameters.max_time_in_seconds = timeout
            
            # Create the final solving model from the modified clone
            solving_model = test_model._create_solving_model()
            status = temp_solver.Solve(solving_model)
            
            is_feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            results[name] = {
                "status": status,
                "status_name": self._status_name(status),
                "necessary": not is_feasible,
                "objective_value": temp_solver.ObjectiveValue() if is_feasible else None,
            }

        return results

    def disable_tagged_and_solve(
        self,
        model: EnhancedCpModel,
        tag: str,
        **solver_params
    ) -> Dict[str, Any]:
        """
        Clones a model, disables all constraints with a given tag, and solves it.

        Args:
            model: The base EnhancedCpModel.
            tag: The tag identifying which constraints to disable.
            **solver_params: Solver parameters for this specific run.

        Returns:
            A dictionary with the results of the solve.
        """
        clone = model.clone()
        clone.disable_constraints_by_tag(tag)
        status = self._internal_solve(clone, **solver_params)
        
        return {
            "tag": tag,
            "disabled_constraints": [
                c for c in clone.get_constraint_names() if tag in clone._constraints[c].tags
            ],
            "status": status,
            "objective_value": self.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        }

    def enable_only_tagged_and_solve(
        self,
        model: EnhancedCpModel,
        tags: List[str],
        **solver_params
    ) -> Dict[str, Any]:
        """
        Clones a model, disables ALL constraints, then re-enables only those
        matching the given tags, and solves it.

        Args:
            model: The base EnhancedCpModel.
            tags: A list of tags. Constraints with any of these tags will be enabled.
            **solver_params: Solver parameters for this specific run.

        Returns:
            A dictionary with the results of the solve.
        """
        clone = model.clone()
        clone.disable_constraints(clone.get_constraint_names())
        for t in tags:
            clone.enable_constraints_by_tag(t)
            
        status = self._internal_solve(clone, **solver_params)
        
        return {
            "enabled_tags": tags,
            "enabled_constraints": clone.get_enabled_constraints(),
            "status": status,
            "objective_value": self.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        }

    def find_minimal_infeasible_subset(
        self,
        model: EnhancedCpModel,
        **solver_params
    ) -> Dict[str, Any]:
        """
        Delegates to the model's built-in MIS (Minimal Infeasible Subset) finder.

        Args:
            model: The infeasible EnhancedCpModel.
            **solver_params: Parameters to pass to the MIS solver.

        Returns:
            A dictionary with debugging information from the model.
        """
        return model.debug_infeasible(self, **solver_params)

    # --------------------------------
    # History and State Access
    # --------------------------------
    def history(self) -> List[Dict[str, Any]]:
        """Returns the solve history."""
        return self.solve_history

    def last_result(self) -> Optional[Dict[str, Any]]:
        """Returns the last solve record (or None)."""
        return self.solve_history[-1] if self.solve_history else None

    def clear_history(self):
        """Clear public history and reset caches for 'last'."""
        self.solve_history.clear()
        self._last_solution_values.clear()
        self._last_model = None
        self._last_model_metadata = None

    def last_model(self) -> Optional[EnhancedCpModel]:
        """Returns the last EnhancedCpModel instance that was solved."""
        return self._last_model

    # --------------------------------
    # Utilities
    # --------------------------------

    def _snapshot_parameters(self) -> sat_parameters_pb2.SatParameters:
        """Creates a snapshot of the current solver parameters."""
        p = sat_parameters_pb2.SatParameters()
        p.CopyFrom(self.parameters)
        return p

    @staticmethod
    def _status_name(status: int) -> str:
        """Converts a status code to its string representation."""
        # Status constants are in cp_model module as integers
        status_names = {
            0: "UNKNOWN",
            1: "MODEL_INVALID", 
            2: "FEASIBLE",
            3: "INFEASIBLE",
            4: "OPTIMAL"
        }
        return status_names.get(status, f"UNKNOWN_STATUS_{status}")