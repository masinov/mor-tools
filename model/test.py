# test_debug_infeasible.py
from model import EnhancedCpModel
from ortools.sat.python import cp_model


def build_infeasible_model() -> EnhancedCpModel:
    model = EnhancedCpModel()

    # Two integer vars with overlapping constraints that make infeasible
    x = model.NewIntVar(0, 5, "x")
    y = model.NewIntVar(0, 5, "y")

    # Add contradictory constraints
    model.Add(x + y == 3).WithName("sum_eq_3")
    model.Add(x + y == 4).WithName("sum_eq_4")

    return model


if __name__ == "__main__":
    model = build_infeasible_model()

    # Quick summary
    print("=== Model summary ===")
    print(model.summary())

    # Run infeasibility analysis
    solver = cp_model.CpSolver()
    result = model.debug_infeasible(solver, max_time_in_seconds=5)

    print("\n=== debug_infeasible result ===")
    for k, v in result.items():
        print(f"{k:20}: {v}")
