from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Union, Any
import uuid

class EnhancedCpModel(cp_model.CpModel):
    """
    Enhanced CP-SAT model that extends CpModel with:
      - Constraint registration, enabling/disabling
      - Named objectives and switching
      - Submodel support (with context manager)
      - Debug solver for minimal infeasibility analysis

    NOTE:
        The built-in `_debug_solver` is **only for debugging**.
        Use your own CpSolver instance for normal solving.
    """

    def __init__(self):
        super().__init__()

        # Debug-only solver
        self._debug_solver = cp_model.CpSolver()

        # Constraint registry: id -> {name, enable_var, constraint, ortools_constraint, enabled}
        self._constraint_registry: Dict[str, Dict] = {}
        self._constraint_counter = 0
        self._constraint_names: set = set()

        # Objectives: name -> {expression, is_minimization}
        self._objectives: Dict[str, Dict] = {}
        self._objective_counter = 0
        self._objective_names: set = set()

        # Variable registry: name -> variable object
        self._variables: Dict[str, Any] = {}

        # Submodels: name -> constraint_names
        self._submodels: Dict[str, List[str]] = {}
        self._active_submodel: Optional[str] = None

    # -------------------------------
    # Variable registration
    # -------------------------------
    def NewIntVar(self, lb, ub, name: str):
        var = super().NewIntVar(lb, ub, name)
        self._variables[name] = var
        return var

    def NewBoolVar(self, name: str):
        var = super().NewBoolVar(name)
        self._variables[name] = var
        return var

    # -------------------------------
    # Constraint management
    # -------------------------------
    def Add(self, constraint, name: Optional[str] = None):
        constraint_id = f"{name or 'constraint'}_{self._constraint_counter}"
        self._constraint_counter += 1

        if name is None:
            name = constraint_id
        elif name in self._constraint_names:
            raise ValueError(f"Constraint name '{name}' already exists")

        self._constraint_names.add(name)

        enable_var = self.NewBoolVar(f"enable_{name}")
        constraint.OnlyEnforceIf(enable_var)

        # Enable by default
        super().Add(enable_var == 1)

        result = super().Add(constraint)

        self._constraint_registry[constraint_id] = {
            'name': name,
            'constraint': constraint,
            'ortools_constraint': result,
            'enable_var': enable_var,
            'enabled': True
        }

        return result

    def enable_constraint(self, name: str):
        cid = self._find_constraint_by_name(name)
        if cid:
            self._constraint_registry[cid]['enabled'] = True
            super().Add(self._constraint_registry[cid]['enable_var'] == 1)

    def disable_constraint(self, name: str):
        cid = self._find_constraint_by_name(name)
        if cid:
            self._constraint_registry[cid]['enabled'] = False
            super().Add(self._constraint_registry[cid]['enable_var'] == 0)

    def enable_constraints(self, names: List[str]):
        for n in names:
            self.enable_constraint(n)

    def disable_constraints(self, names: List[str]):
        for n in names:
            self.disable_constraint(n)

    def get_constraint_names(self) -> List[str]:
        return [info['name'] for info in self._constraint_registry.values()]

    def _find_constraint_by_name(self, name: str) -> Optional[str]:
        for cid, info in self._constraint_registry.items():
            if info['name'] == name:
                return cid
        return None

    # -------------------------------
    # Objective management
    # -------------------------------
    def Minimize(self, expression, name: Optional[str] = None):
        if name is None:
            name = f"objective_{self._objective_counter}"
            self._objective_counter += 1
        elif name in self._objective_names:
            raise ValueError(f"Objective name '{name}' already exists")

        self._objective_names.add(name)
        self._objectives[name] = {
            'expression': expression,
            'is_minimization': True
        }
        super().Minimize(expression)

    def Maximize(self, expression, name: Optional[str] = None):
        if name is None:
            name = f"objective_{self._objective_counter}"
            self._objective_counter += 1
        elif name in self._objective_names:
            raise ValueError(f"Objective name '{name}' already exists")

        self._objective_names.add(name)
        self._objectives[name] = {
            'expression': expression,
            'is_minimization': False
        }
        super().Maximize(expression)

    def set_objective(self, name: str):
        if name not in self._objectives:
            raise ValueError(f"Objective '{name}' not registered")
        obj = self._objectives[name]
        if obj['is_minimization']:
            self.Minimize(obj['expression'])
        else:
            self.Maximize(obj['expression'])

    def get_objective_names(self) -> List[str]:
        return list(self._objectives.keys())

    # -------------------------------
    # Submodel support
    # -------------------------------
    def create_submodel(self, name: str, constraint_names: List[str]):
        valid = self.get_constraint_names()
        invalid = [n for n in constraint_names if n not in valid]
        if invalid:
            raise ValueError(f"Invalid constraint names: {invalid}")
        self._submodels[name] = constraint_names

    def activate_submodel(self, name: str):
        if name not in self._submodels:
            raise ValueError(f"Submodel '{name}' not found")
        active_constraints = set(self._submodels[name])
        for _, info in self._constraint_registry.items():
            if info['name'] in active_constraints:
                self.enable_constraint(info['name'])
            else:
                self.disable_constraint(info['name'])
        self._active_submodel = name

    def deactivate_submodel(self):
        for _, info in self._constraint_registry.items():
            self.enable_constraint(info['name'])
        self._active_submodel = None

    class _SubmodelContext:
        def __init__(self, model, name):
            self.model = model
            self.name = name
            self.prev_submodel = model._active_submodel
        def __enter__(self):
            self.model.activate_submodel(self.name)
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.prev_submodel:
                self.model.activate_submodel(self.prev_submodel)
            else:
                self.model.deactivate_submodel()

    def submodel(self, name: str):
        return self._SubmodelContext(self, name)

    # -------------------------------
    # Debugging tools
    # -------------------------------
    def set_debug_solver_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._debug_solver.parameters, k, v)

    def find_minimal_infeasible_constraints(self) -> List[str]:
        disable_vars = []
        names = []
        for _, info in self._constraint_registry.items():
            disable_var = self.NewBoolVar(f"disable_{info['name']}")
            super().Add(disable_var + info['enable_var'] == 1)
            disable_vars.append(disable_var)
            names.append(info['name'])
        self.Minimize(sum(disable_vars))
        status = self._debug_solver.Solve(self)
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return []
        return [names[i] for i, dv in enumerate(disable_vars)
                if self._debug_solver.Value(dv) == 1]

    def debug_solve(self) -> Dict[str, Any]:
        status = self._debug_solver.Solve(self)
        debug_info = {
            'status': status,
            'feasible': status in [cp_model.OPTIMAL, cp_model.FEASIBLE],
            'solve_time': self._debug_solver.WallTime(),
            'disabled_constraints': []
        }
        if not debug_info['feasible']:
            disabled = self.find_minimal_infeasible_constraints()
            debug_info['disabled_constraints'] = disabled
            for c in disabled:
                self.disable_constraint(c)
            final_status = self._debug_solver.Solve(self)
            debug_info.update({
                'final_status': final_status,
                'final_feasible': final_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
            })
        return debug_info

    # -------------------------------
    # Summaries
    # -------------------------------
    def summary(self):
        return {
            "constraints": {
                name: "enabled" if info['enabled'] else "disabled"
                for _, info in self._constraint_registry.items()
                for name in [info['name']]
            },
            "objectives": list(self._objectives.keys()),
            "variables": list(self._variables.keys()),
            "submodels": list(self._submodels.keys()),
            "active_submodel": self._active_submodel
        }
