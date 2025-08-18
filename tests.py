#!/usr/bin/env python3
"""
Comprehensive test battery for EnhancedCpSolver and EnhancedCpModel.

This test suite covers:
- Basic model creation and variable management
- Constraint creation and management
- Enable/disable functionality
- Model solving and solution extraction
- History tracking and replay functionality
- Auto-hinting capabilities
- Model cloning and copying
- Advanced debugging features (MIS, constraint necessity testing)
- Tag-based constraint management
- Relaxation and subset functionality
- Error handling and edge cases
"""

import pytest
import time
import random
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model
from ortools.sat import sat_parameters_pb2

# Import our enhanced classes (assuming they're in separate files)
try:
    from model import EnhancedCpModel
    from solver import EnhancedCpSolver
except ImportError:
    # If running standalone, the classes should be defined above or imported differently
    print("Warning: Could not import EnhancedCpModel and EnhancedCpSolver from modules")
    print("Make sure the classes are available in the current namespace")


class TestEnhancedCpModel:
    """Test suite for EnhancedCpModel functionality."""
    
    def test_basic_model_creation(self):
        """Test basic model creation and properties."""
        model = EnhancedCpModel()
        assert len(model) == 0
        assert model.get_constraint_names() == []
        assert model.get_variable_names() == []
        summary = model.summary()
        assert summary["total_constraints"] == 0
        assert summary["total_variables"] == 0
        print("✓ Basic model creation test passed")

    def test_variable_creation(self):
        """Test creation of different variable types."""
        model = EnhancedCpModel()
        
        # Integer variables
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(-5, 15, "y")
        
        # Boolean variables
        b1 = model.NewBoolVar("b1")
        b2 = model.NewBoolVar("b2")
        
        # Constants
        c = model.NewConstant(42)
        
        # Interval variables
        start = model.NewIntVar(0, 10, "start")
        size = model.NewIntVar(1, 5, "size")
        end = model.NewIntVar(1, 15, "end")
        interval = model.NewIntervalVar(start, size, end, "interval")
        
        # Verify variables are tracked
        assert len(model.get_variable_names()) >= 6  # At least the named ones
        assert "x" in model.get_variable_names()
        assert "b1" in model.get_variable_names()
        
        # Test variable info retrieval
        x_info = model.get_variable_info("x")
        assert x_info.var_type == "IntVar"
        assert x_info.creation_args == (0, 10, "x")
        
        b1_info = model.get_variable_info("b1")
        assert b1_info.var_type == "BoolVar"
        
        print("✓ Variable creation test passed")

    def test_constraint_creation_and_management(self):
        """Test constraint creation and enable/disable functionality."""
        model = EnhancedCpModel()
        
        # Create variables
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        z = model.NewIntVar(0, 10, "z")
        
        # Create constraints with explicit names
        c1 = model.Add(x + y <= 10, "sum_constraint")
        c2 = model.AddAllDifferent([x, y, z], "all_diff")
        c3 = model.Add(x >= 2, "x_lower_bound")
        
        # Verify constraints are tracked
        assert len(model.get_constraint_names()) == 3
        assert "sum_constraint" in model.get_constraint_names()
        assert "all_diff" in model.get_constraint_names()
        
        # Test constraint info
        c1_info = model.get_constraint_info("sum_constraint")
        assert c1_info.constraint_type == "Generic"
        assert c1_info.enabled == True
        
        # Test enable/disable
        assert "sum_constraint" in model.get_enabled_constraints()
        model.disable_constraint("sum_constraint")
        assert "sum_constraint" in model.get_disabled_constraints()
        assert "sum_constraint" not in model.get_enabled_constraints()
        
        model.enable_constraint("sum_constraint")
        assert "sum_constraint" in model.get_enabled_constraints()
        
        print("✓ Constraint creation and management test passed")

    def test_constraint_tags(self):
        """Test constraint tagging functionality."""
        model = EnhancedCpModel()
        
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        
        # Create constraints
        c1 = model.Add(x + y <= 10, "c1")
        c2 = model.Add(x >= 2, "c2")
        c3 = model.Add(y <= 8, "c3")
        
        # Add tags
        model.add_constraint_tag("c1", "bounds")
        model.add_constraint_tag("c2", "bounds")
        model.add_constraint_tag("c2", "lower_bounds")
        model.add_constraint_tag("c3", "upper_bounds")
        
        # Test tag-based operations
        bounds_constraints = model.get_constraints_by_tag("bounds")
        assert "c1" in bounds_constraints
        assert "c2" in bounds_constraints
        assert "c3" not in bounds_constraints
        
        # Test enable/disable by tag
        model.disable_constraints_by_tag("bounds")
        assert "c1" in model.get_disabled_constraints()
        assert "c2" in model.get_disabled_constraints()
        assert "c3" in model.get_enabled_constraints()
        
        model.enable_constraints_by_tag("lower_bounds")
        assert "c2" in model.get_enabled_constraints()
        
        print("✓ Constraint tagging test passed")

    def test_model_cloning(self):
        """Test model cloning functionality."""
        model = EnhancedCpModel()
        
        # Create a model with variables and constraints
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x + y <= 10, "sum_constraint")
        model.add_constraint_tag("sum_constraint", "test_tag")
        model.disable_constraint("sum_constraint")
        model.Minimize(x + y, "minimize_sum")
        
        # Clone the model
        cloned = model.clone()
        
        # Verify clone has same structure
        assert len(cloned.get_constraint_names()) == len(model.get_constraint_names())
        assert len(cloned.get_variable_names()) == len(model.get_variable_names())
        assert cloned.get_disabled_constraints() == model.get_disabled_constraints()
        
        # Verify clone is independent
        cloned.enable_constraint("sum_constraint")
        assert "sum_constraint" in cloned.get_enabled_constraints()
        assert "sum_constraint" in model.get_disabled_constraints()
        
        # Verify tags are preserved
        assert "test_tag" in cloned.get_constraints_by_tag("test_tag")
        
        print("✓ Model cloning test passed")

    def test_objectives(self):
        """Test objective management."""
        model = EnhancedCpModel()
        
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        
        # Add objectives
        model.Minimize(x + y, "minimize_sum")
        model.Maximize(x - y, "maximize_diff")
        
        # Test objective info
        objectives = model.get_enabled_objectives()
        assert len(objectives) == 2
        
        # Test disable/enable
        model.disable_objective("minimize_sum")
        objectives = model.get_enabled_objectives()
        assert len(objectives) == 1
        assert objectives[0].name == "maximize_diff"
        
        print("✓ Objectives test passed")


class TestEnhancedCpSolver:
    """Test suite for EnhancedCpSolver functionality."""
    
    def create_simple_feasible_model(self) -> EnhancedCpModel:
        """Create a simple feasible model for testing."""
        model = EnhancedCpModel()
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x + y <= 10, "sum_constraint")
        model.Add(x >= 2, "x_lower")
        model.Add(y >= 1, "y_lower")
        model.Minimize(x + y, "minimize_sum")
        return model
    
    def create_infeasible_model(self) -> EnhancedCpModel:
        """Create an infeasible model for testing."""
        model = EnhancedCpModel()
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x + y <= 5, "sum_constraint")
        model.Add(x >= 8, "x_lower")
        model.Add(y >= 3, "y_lower")  # 8 + 3 > 5, so infeasible
        return model

    def test_basic_solving(self):
        """Test basic solving functionality."""
        solver = EnhancedCpSolver()
        model = self.create_simple_feasible_model()
        
        # Solve the model
        status = solver.Solve(model)
        
        # Check result
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Check history
        assert len(solver.history()) == 1
        last_result = solver.last_result()
        assert last_result is not None
        assert last_result["status"] == status
        assert "variables" in last_result
        assert "x" in last_result["variables"]
        
        print("✓ Basic solving test passed")

    def test_solve_with_parameters(self):
        """Test solving with custom parameters."""
        solver = EnhancedCpSolver()
        model = self.create_simple_feasible_model()
        
        # Create custom parameters
        params = sat_parameters_pb2.SatParameters()
        params.max_time_in_seconds = 10.0
        params.num_search_workers = 1
        
        # Solve with parameters
        status = solver.SolveWithParameters(model, params)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Verify parameters were used
        last_result = solver.last_result()
        stored_params = last_result["parameters"]
        assert stored_params.max_time_in_seconds == 10.0
        
        print("✓ Solve with parameters test passed")

    def test_constraint_enable_disable_solving(self):
        """Test that enable/disable affects solving."""
        solver = EnhancedCpSolver()
        model = self.create_infeasible_model()
        
        # First solve - should be infeasible
        status = solver.Solve(model)
        assert status == cp_model.INFEASIBLE
        
        # Disable one conflicting constraint
        model.disable_constraint("x_lower")
        status = solver.Solve(model)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        print("✓ Constraint enable/disable solving test passed")

    def test_history_tracking(self):
        """Test comprehensive history tracking."""
        solver = EnhancedCpSolver()
        model = self.create_simple_feasible_model()
        
        # Solve multiple times with different configurations
        status1 = solver.Solve(model)
        
        model.disable_constraint("y_lower")
        status2 = solver.Solve(model)
        
        model.enable_constraint("y_lower")
        model.add_constraint_tag("sum_constraint", "test_tag")
        status3 = solver.Solve(model)
        
        # Check history
        history = solver.history()
        assert len(history) == 3
        
        # Check each record has required fields
        for i, record in enumerate(history):
            assert "status" in record
            assert "status_name" in record
            assert "solve_time" in record
            assert "variables" in record
            assert "model_clone" in record
            assert "model_summary" in record
            assert "enabled_constraints" in record
            assert "disabled_constraints" in record
            assert "timestamp" in record
            assert "model_id" in record
            
            # Verify model_clone is functional
            clone = record["model_clone"]
            assert isinstance(clone, EnhancedCpModel)
            
        print("✓ History tracking test passed")

    def test_replay_functionality(self):
        """Test model replay from history."""
        solver = EnhancedCpSolver()
        model = self.create_simple_feasible_model()
        
        # Solve original
        status1 = solver.Solve(model)
        original_vars = dict(solver.last_result()["variables"])
        
        # Modify model and solve again
        model.disable_constraint("y_lower")
        status2 = solver.Solve(model)
        
        # Replay from history
        replay_status = solver.replay_from_history(0)  # Replay first solve
        assert replay_status == status1
        
        # Replay last model
        last_replay_status = solver.replay_last()
        assert last_replay_status == status2
        
        # Test creating replay model without solving
        replay_model = solver.create_replay_model(0)
        assert isinstance(replay_model, EnhancedCpModel)
        assert "sum_constraint" in replay_model.get_enabled_constraints()
        
        print("✓ Replay functionality test passed")

    def test_auto_hinting(self):
        """Test auto-hinting functionality."""
        solver = EnhancedCpSolver()
        model = self.create_simple_feasible_model()
        
        # Solve without hints
        solver.Solve(model)
        first_solution = dict(solver.last_result()["variables"])
        
        # Enable auto-hinting
        solver.enable_auto_hint()
        
        # Create a similar model
        model2 = self.create_simple_feasible_model()
        solver.Solve(model2)  # Should use hints from first solve
        
        # Test manual hint seed
        manual_hints = {"x": 5, "y": 3}
        solver.set_manual_auto_hint_seed(manual_hints)
        solver.Solve(model2)
        
        # Disable auto-hinting
        solver.disable_auto_hint()
        solver.Solve(model2)  # Should not use hints
        
        print("✓ Auto-hinting test passed")

    def test_relaxation_and_subsets(self):
        """Test relaxation and subset functionality."""
        solver = EnhancedCpSolver()
        model = self.create_infeasible_model()
        
        # Solve infeasible model
        status = solver.Solve(model)
        assert status == cp_model.INFEASIBLE
        
        # Create relaxed copy
        relaxed = solver.relaxed_copy(relaxation_factor=0.5)
        status_relaxed = solver.Solve(relaxed)
        # Relaxed should be more likely to be feasible (though not guaranteed)
        
        # Create subset copy
        subset = solver.subset_copy(["sum_constraint"])
        status_subset = solver.Solve(subset)
        assert status_subset in [cp_model.OPTIMAL, cp_model.FEASIBLE]  # Only sum constraint, should be feasible
        
        print("✓ Relaxation and subsets test passed")

    def test_constraint_necessity_testing(self):
        """Test constraint necessity analysis."""
        solver = EnhancedCpSolver()
        model = self.create_infeasible_model()
        
        # Solve to establish baseline
        status = solver.Solve(model)
        assert status == cp_model.INFEASIBLE
        
        # Test constraint necessity
        constraint_names = ["sum_constraint", "x_lower", "y_lower"]
        results = solver.test_constraint_necessity(constraint_names)
        
        assert len(results) == 3
        for name in constraint_names:
            assert name in results
            assert "status" in results[name]
            assert "necessary" in results[name]
            # At least one constraint should be identified as non-necessary for feasibility
        
        # At least one constraint removal should make it feasible
        necessity_results = [results[name]["necessary"] for name in constraint_names]
        assert not all(necessity_results), "At least one constraint should not be necessary"
        
        print("✓ Constraint necessity testing passed")

    def test_tagged_solving(self):
        """Test tag-based solving functionality."""
        solver = EnhancedCpSolver()
        model = EnhancedCpModel()
        
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        
        model.Add(x + y <= 10, "sum")
        model.Add(x >= 2, "x_bound")
        model.Add(y >= 1, "y_bound")
        
        # Add tags
        model.add_constraint_tag("sum", "core")
        model.add_constraint_tag("x_bound", "bounds")
        model.add_constraint_tag("y_bound", "bounds")
        
        model.Minimize(x + y)
        
        # Test disabling tagged constraints
        result1 = solver.disable_tagged_and_solve(model, "bounds")
        assert result1["status"] in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        assert len(result1["disabled_constraints"]) == 2
        
        # Test enabling only tagged constraints
        result2 = solver.enable_only_tagged_and_solve(model, ["core"])
        assert result2["status"] in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        assert "sum" in result2["enabled_constraints"]
        
        print("✓ Tagged solving test passed")

    def test_minimal_infeasible_subset(self):
        """Test MIS (Minimal Infeasible Subset) finding."""
        solver = EnhancedCpSolver()
        model = self.create_infeasible_model()
        
        # Find minimal infeasible subset
        mis_result = solver.find_minimal_infeasible_subset(model)
        
        assert "feasible" in mis_result
        assert "disabled_constraints" in mis_result
        assert "total_disabled" in mis_result
        assert "method" in mis_result
        
        if mis_result["feasible"]:
            # If MIS found a solution, there should be disabled constraints
            assert mis_result["total_disabled"] > 0
            assert len(mis_result["disabled_constraints"]) > 0
        
        print("✓ Minimal infeasible subset test passed")

    def test_pre_relaxed_solving(self):
        """Test pre-relaxed solving functionality."""
        solver = EnhancedCpSolver()
        model = self.create_infeasible_model()
        
        # Solve with pre-relaxation
        status = solver.Solve(model, pre_relaxed=True, relaxation_factor=0.5)
        # Pre-relaxed should be more likely to find a solution
        
        # Check that history shows relaxed model was used
        last_result = solver.last_result()
        model_summary = last_result["model_summary"]
        assert model_summary["disabled_constraints"] > 0  # Some constraints should be disabled
        
        print("✓ Pre-relaxed solving test passed")

    def test_error_handling(self):
        """Test error handling and edge cases."""
        solver = EnhancedCpSolver()
        
        # Test with non-EnhancedCpModel
        regular_model = cp_model.CpModel()
        x = regular_model.NewIntVar(0, 10, "x")
        regular_model.Add(x >= 5)
        
        with pytest.raises(TypeError):
            solver.Solve(regular_model)
        
        # Test replay without history
        with pytest.raises(ValueError):
            solver.replay_last()
        
        # Test history access with invalid index
        with pytest.raises(IndexError):
            solver.replay_from_history(999)
        
        # Test constraint operations on non-existent constraints
        model = EnhancedCpModel()
        with pytest.raises(ValueError):
            model.enable_constraint("non_existent")
        
        print("✓ Error handling test passed")


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow(self):
        """Test a complete workflow using all major features."""
        solver = EnhancedCpSolver()
        
        # Create complex model
        model = EnhancedCpModel()
        
        # Variables
        x = model.NewIntVar(0, 100, "x")
        y = model.NewIntVar(0, 100, "y")
        z = model.NewIntVar(0, 100, "z")
        
        # Constraints with tags
        model.Add(x + y + z <= 150, "budget")
        model.Add(x >= 20, "x_min")
        model.Add(y >= 30, "y_min")
        model.Add(z >= 10, "z_min")
        model.AddAllDifferent([x, y, z], "all_different")
        
        # Add tags
        model.add_constraint_tag("budget", "capacity")
        model.add_constraint_tag("x_min", "bounds")
        model.add_constraint_tag("y_min", "bounds")
        model.add_constraint_tag("z_min", "bounds")
        model.add_constraint_tag("all_different", "uniqueness")
        
        # Objective
        model.Maximize(2*x + 3*y + z, "profit")
        
        # Enable auto-hinting
        solver.enable_auto_hint()
        
        # Solve original model
        status1 = solver.Solve(model)
        assert status1 in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        original_objective = solver.ObjectiveValue()
        
        # Experiment with constraint subsets
        core_result = solver.enable_only_tagged_and_solve(model, ["capacity", "bounds"])
        relaxed_result = solver.disable_tagged_and_solve(model, "uniqueness")
        
        # Test constraint necessity
        necessity_results = solver.test_constraint_necessity(
            ["budget", "x_min", "y_min", "all_different"]
        )
        
        # Create variations and solve
        model.disable_constraint("all_different")
        status2 = solver.Solve(model)
        
        # Replay original
        replay_status = solver.replay_from_history(0, with_hints=True)
        
        # Verify comprehensive history
        history = solver.history()
        assert len(history) >= 4  # Original + experiments + modified + replay
        
        # Verify all solves have proper metadata
        for record in history:
            assert "status" in record
            assert "model_summary" in record
            assert "enabled_constraints" in record
            assert record["model_summary"]["total_constraints"] > 0
        
        print("✓ Full workflow integration test passed")

    def test_performance_and_memory(self):
        """Test performance with larger models and memory management."""
        solver = EnhancedCpSolver()
        
        # Create larger model
        model = EnhancedCpModel()
        n = 20  # Size parameter
        
        # Create variables
        vars = []
        for i in range(n):
            vars.append(model.NewIntVar(0, 100, f"x_{i}"))
        
        # Create many constraints
        for i in range(n-1):
            model.Add(vars[i] + vars[i+1] <= 150, f"pair_{i}")
            model.add_constraint_tag(f"pair_{i}", "adjacent")
        
        # Add sum constraint
        model.Add(sum(vars) <= 500, "total_sum")
        model.Minimize(sum(vars))
        
        # Solve multiple times
        start_time = time.time()
        for iteration in range(5):
            if iteration > 0:
                # Modify model slightly each iteration
                model.disable_constraints_by_tag("adjacent")
                random.shuffle(list(model.get_constraint_names()))
                enabled = list(model.get_constraint_names())[:len(model.get_constraint_names())//2]
                model.enable_constraints(enabled)
            
            status = solver.Solve(model)
            assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.INFEASIBLE]
        
        solve_time = time.time() - start_time
        
        # Verify history management
        assert len(solver.history()) == 5
        
        # Test history cleanup
        solver.clear_history()
        assert len(solver.history()) == 0
        
        print(f"✓ Performance test passed (5 solves in {solve_time:.2f}s)")


def run_all_tests():
    """Run all tests in the battery."""
    print("🧪 Starting Enhanced CP-SAT Test Battery")
    print("=" * 50)
    
    # Model tests
    print("\n📋 Testing EnhancedCpModel...")
    model_tests = TestEnhancedCpModel()
    model_tests.test_basic_model_creation()
    model_tests.test_variable_creation()
    model_tests.test_constraint_creation_and_management()
    model_tests.test_constraint_tags()
    model_tests.test_model_cloning()
    model_tests.test_objectives()
    
    # Solver tests
    print("\n🔧 Testing EnhancedCpSolver...")
    solver_tests = TestEnhancedCpSolver()
    solver_tests.test_basic_solving()
    solver_tests.test_solve_with_parameters()
    solver_tests.test_constraint_enable_disable_solving()
    solver_tests.test_history_tracking()
    solver_tests.test_replay_functionality()
    solver_tests.test_auto_hinting()
    solver_tests.test_relaxation_and_subsets()
    solver_tests.test_constraint_necessity_testing()
    solver_tests.test_tagged_solving()
    solver_tests.test_minimal_infeasible_subset()
    solver_tests.test_pre_relaxed_solving()
    # solver_tests.test_error_handling()  # Uncomment if using pytest
    
    # Integration tests
    print("\n🔗 Testing Integration...")
    integration_tests = TestIntegration()
    integration_tests.test_full_workflow()
    integration_tests.test_performance_and_memory()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("Enhanced CP-SAT implementation is working correctly.")


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("🚀 Quick Smoke Test")
    print("-" * 20)
    
    # Test basic functionality
    try:
        # Create model
        model = EnhancedCpModel()
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x + y <= 10, "sum")
        model.Minimize(x + y)
        
        # Create solver and solve
        solver = EnhancedCpSolver()
        status = solver.Solve(model)
        
        # Basic checks
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        assert len(solver.history()) == 1
        assert "x" in solver.last_result()["variables"]
        
        print("✓ Smoke test passed - basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


if __name__ == "__main__":
    # Run quick smoke test first
    if run_quick_smoke_test():
        # If smoke test passes, run full battery
        print("\n" + "=" * 60)
        run_all_tests()
    else:
        print("Skipping full test battery due to smoke test failure")