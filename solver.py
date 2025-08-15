from ortools.sat.python import cp_model
from typing import List, Dict, Any, Optional, Union
import copy
import time


class EnhancedCpSolver(cp_model.CpSolver):
    """
    Enhanced CpSolver with:
      - Model tracking (stores last solved model and/or proto)
      - Public solve history (list of dicts): `solve_history`
      - Add hints from last or any past resolution
      - Optional auto-hinting from last or a chosen history index
      - Optional per-record proto storage

    Notes:
      * `solve_history` is a public list; you can inspect or clear it directly.
      * For memory safety in large runs, you can disable proto snapshots in history.
    """

    def __init__(self):
        super().__init__()
        self._last_model: Optional[cp_model.CpModel] = None
        self._last_proto: Optional[cp_model.CpModelProto] = None

        # PUBLIC history: each entry is a dict with keys:
        #   status, status_name, solve_time, objective_value, variables, parameters,
        #   timestamp, model_id, model_proto (optional, if enabled)
        self.solve_history: List[Dict[str, Any]] = []

        # internal caches for “last” and auto-hinting
        self._last_solution_values: Dict[str, Union[int, float]] = {}
        self._auto_hint_enabled: bool = False
        self._auto_hint_source_index: Optional[int] = None  # None => default to "last"
        self._auto_hint_seed_values: Optional[Dict[str, Union[int, float]]] = None

        # control whether to store a proto snapshot per history record
        self._store_proto_in_history: bool = True

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

    def set_store_proto_in_history(self, enabled: bool):
        """Enable/disable storing a `model_proto` snapshot in each history entry."""
        self._store_proto_in_history = enabled

    # --------------------------------
    # Main solving with tracking
    # --------------------------------
    def Solve(self, model: cp_model.CpModel):
        """
        Solve the given model, tracking the proto, model, and storing results in `solve_history`.
        Applies auto-hints if enabled (from chosen history record, manual seed, or last).
        """
        # Choose auto-hint source if enabled
        if self._auto_hint_enabled:
            hinted = False
            # 1) highest priority: explicit manual seed values (if set)
            if self._auto_hint_seed_values:
                self._apply_hints(model, self._auto_hint_seed_values)
                hinted = True
            else:
                # 2) from chosen history index (if set and valid at this moment)
                src_idx = self._auto_hint_source_index
                if src_idx is not None and 0 <= src_idx < len(self.solve_history):
                    src_vars = self.solve_history[src_idx].get("variables", {})
                    if src_vars:
                        self._apply_hints(model, src_vars)
                        hinted = True
                # 3) fallback: last solution values (classic behavior)
                if not hinted and self._last_solution_values:
                    self._apply_hints(model, self._last_solution_values)

        # Keep copies of model and proto
        self._last_model = model
        try:
            self._last_proto = copy.deepcopy(model.Proto())
        except Exception:
            self._last_proto = None

        start_time = time.time()
        status = super().Solve(model)
        elapsed = time.time() - start_time

        # Collect variable values (by name) if feasible
        self._last_solution_values = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Prefer EnhancedCpModel's registry when available
            if hasattr(model, "_variables") and isinstance(getattr(model, "_variables"), dict):
                for name, var in model._variables.items():
                    try:
                        self._last_solution_values[name] = self.Value(var)
                    except Exception:
                        pass
            else:
                # Best-effort: rely on proto names if user kept consistent naming
                # (Only works if you can retrieve named vars; otherwise skip.)
                proto = model.Proto()
                # No safe public API to map proto->IntVar object; skip in plain CpModel.
                # Users of plain CpModel can add hints manually.
                _ = proto  # silence linter; intentionally unused

        # Build history record
        record: Dict[str, Any] = {
            "status": status,
            "status_name": self._status_name(status),
            "solve_time": elapsed,
            "objective_value": (
                self.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None
            ),
            "variables": dict(self._last_solution_values),
            "parameters": copy.deepcopy(self.parameters),
            "timestamp": time.time(),
            "model_id": id(model),
        }
        if self._store_proto_in_history:
            try:
                record["model_proto"] = copy.deepcopy(model.Proto())
            except Exception:
                record["model_proto"] = None

        # Append to PUBLIC history
        self.solve_history.append(record)

        return status

    def SolveWithParameters(self, model: cp_model.CpModel, parameters: cp_model.CpSolverParameters):
        """
        Same as Solve but with explicit parameters, also tracked in history.
        """
        self.parameters.CopyFrom(parameters)
        return self.Solve(model)

    # --------------------------------
    # Hinting helpers
    # --------------------------------
    def add_hints_from_last(self, model: cp_model.CpModel):
        """
        Manually add hints from the last solution to the given model.
        """
        if self._last_solution_values:
            self._apply_hints(model, self._last_solution_values)

    def add_hints_from_history(self, model: cp_model.CpModel, index: int):
        """
        Add hints to `model` from a specific history record (by index).
        """
        if index < 0 or index >= len(self.solve_history):
            raise IndexError("History index out of range.")
        values = self.solve_history[index].get("variables", {})
        if values:
            self._apply_hints(model, values)

    def add_hints_from_values(self, model: cp_model.CpModel, values: Dict[str, Union[int, float]]):
        """
        Add hints to `model` from a plain {var_name: value} dict.
        """
        if values:
            self._apply_hints(model, values)

    def set_manual_auto_hint_seed(self, values: Dict[str, Union[int, float]]):
        """
        Provide an explicit manual seed dict for auto-hinting (highest priority when enabled).
        Pass {} or None to clear.
        """
        self._auto_hint_seed_values = dict(values) if values else None

    def _apply_hints(self, model: cp_model.CpModel, previous_values: Dict[str, Union[int, float]]):
        """
        Add hints to the given model for variables present in previous_values.
        Prefers EnhancedCpModel's _variables registry. Falls back silently otherwise.
        """
        if not previous_values:
            return

        # If using EnhancedCpModel, we can reliably resolve variables by name
        if hasattr(model, "_variables") and isinstance(getattr(model, "_variables"), dict):
            for name, val in previous_values.items():
                var = model._variables.get(name)
                if var is not None:
                    try:
                        model.AddHint(var, int(val))
                    except Exception:
                        # Ignore type/domain mismatches silently for hinting
                        pass
        else:
            # Plain CpModel: no safe public way to retrieve variables by name.
            # To use hinting here, the caller should pass actual variables via EnhancedCpModel.
            pass

    # --------------------------------
    # History helpers (kept for backwards compatibility)
    # --------------------------------
    def history(self) -> List[Dict[str, Any]]:
        """Return a deep copy of the solve history (backwards-compatible helper)."""
        return copy.deepcopy(self.solve_history)

    def last_result(self) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the last solve record (or None)."""
        return copy.deepcopy(self.solve_history[-1]) if self.solve_history else None

    def clear_history(self):
        """Clear public history and reset caches for 'last'."""
        self.solve_history.clear()
        self._last_solution_values.clear()

    # --------------------------------
    # Model/proto access
    # --------------------------------
    def last_model(self) -> Optional[cp_model.CpModel]:
        return self._last_model

    def last_proto(self) -> Optional[cp_model.CpModelProto]:
        return self._last_proto

    # --------------------------------
    # Replay
    # --------------------------------
    def replay_last(self, with_hints: bool = False):
        """
        Replay the last solve. If with_hints=True, reapply previous solution values as hints.
        """
        if not self._last_model:
            raise ValueError("No model stored from previous solve.")
        model_copy = copy.deepcopy(self._last_model)
        if with_hints and self._last_solution_values:
            self._apply_hints(model_copy, self._last_solution_values)
        return self.Solve(model_copy)

    # --------------------------------
    # Utilities
    # --------------------------------
    @staticmethod
    def _status_name(status: int) -> str:
        if status == cp_model.OPTIMAL:
            return "OPTIMAL"
        if status == cp_model.FEASIBLE:
            return "FEASIBLE"
        if status == cp_model.INFEASIBLE:
            return "INFEASIBLE"
        if status == cp_model.MODEL_INVALID:
            return "MODEL_INVALID"
        if status == cp_model.UNKNOWN:
            return "UNKNOWN"
        return f"STATUS_{status}"
