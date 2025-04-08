from abc import ABC, abstractmethod
from ortools.sat.python import cp_model


class Model(ABC):
    """
    Abstract base class for ortools models.
    """

    def __init__(self):
        self.model = None
        self.solver = None

        self.variables = {}
        self.constraints = {} 
        self.objective = None
        self.objective_coefficients = {}

        self.history = {}

    @abstractmethod
    def __build_variables(self, _filter: list = None):
        pass

    @abstractmethod
    def __build_constraints(self, _filter: list = None):
        pass

    def __build_objective(self, constructor_dict: dict = None):
        mode = constructor_dict["mode"]

        self.objective = sum(
            [
                self.objective_coefficients[mode][var_name] * self.variables[var_name]
                for var_name in self.objective_coefficients[mode].keys()
            ]
        )

        if constructor_dict["target"] == "minimize":
            self.model.Minimize(self.objective)
        elif constructor_dict["target"] == "maximize":
            self.model.Maximize(self.objective)
    
    def __build_model(
            self, 
            variables_filter: list = None, 
            constraints_filter: list = None,
            objective_dict: dict = None
        ):
        self.solver = cp_model.CpSolver()
        self.model = cp_model.CpModel()

        self.__build_variables(variables_filter)
        self.__build_constraints(constraints_filter)
        self.__build_objective(objective_dict)
        pass

    @abstractmethod
    def debug(self):
        pass

    @abstractmethod
    def solve(self):
        pass