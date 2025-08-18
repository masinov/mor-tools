from typing import List, Tuple, Optional, Dict, Any
from ortools.sat.python import cp_model as _cp
from model import EnhancedCpModel  # Assuming the provided code is in this module


class RectanglePackingOptimizer:
    """
    Optimization model for packing rectangles to cover points while avoiding forbidden intervals.
    
    Problem Description:
    - max_rectangle_number large rectangles of size 1000x2000
    - Each large rectangle contains 18 small rectangles
    - Small rectangles have minimum size 300x300 (or can be disabled with 0x0)
    - Small rectangles can be rotated (w,h or h,w)
    - All points must be covered by small rectangles
    - Small rectangles cannot overlap in their mother rectangle space
    - Small rectangles can overlap in point space
    - Small rectangles' y-coordinates cannot be in forbidden intervals
    - Objective: minimize number of active large rectangles
    """
    
    def __init__(self, 
                 points: List[Tuple[int, int]], 
                 forbidden_y_intervals: List[Tuple[int, int]], 
                 max_rectangle_number: int):
        """
        Initialize the optimizer with problem parameters.
        
        Args:
            points: List of (x, y) points that must be covered
            forbidden_y_intervals: List of (y_min, y_max) intervals to avoid
            max_rectangle_number: Maximum number of large rectangles available
        """
        self.points = points
        self.forbidden_y_intervals = forbidden_y_intervals
        self.max_rectangle_number = max_rectangle_number
        self.small_rectangles_per_large = 18
        
        # Rectangle dimensions
        self.large_rect_width = 1000
        self.large_rect_height = 2000
        self.min_small_size = 300
        
        # Model and variables
        self.model = None
        self.variables = {}
        
        # Initialize the model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the complete optimization model."""
        self.model = EnhancedCpModel()
        self._build_variables()
        self._build_constraints()
    
    def _build_variables(self) -> None:
        """Build all optimization variables."""
        self._build_rectangle_activation_variables()
        self._build_small_rectangle_geometry_variables()
        self._build_small_rectangle_position_variables()
        self._build_point_assignment_variables()
    
    def _build_constraints(self) -> None:
        """Build all constraints."""
        self._build_rectangle_activation_constraints()
        self._build_geometry_constraints()
        self._build_non_overlap_constraints_in_mother_rectangle()
        self._build_point_coverage_constraints()
        self._build_forbidden_interval_constraints()
        self._build_objective()
    
    def _build_rectangle_activation_variables(self) -> None:
        """Build variables for rectangle activation status."""
        # Large rectangle activation
        self.variables['large_rect_active'] = {}
        for i in range(self.max_rectangle_number):
            self.variables['large_rect_active'][i] = self.model.NewBoolVar(f'large_rect_{i}_active')
        
        # Small rectangle activation
        self.variables['small_rect_active'] = {}
        for i in range(self.max_rectangle_number):
            self.variables['small_rect_active'][i] = {}
            for j in range(self.small_rectangles_per_large):
                self.variables['small_rect_active'][i][j] = self.model.NewBoolVar(f'small_rect_{i}_{j}_active')
    
    def _build_small_rectangle_geometry_variables(self) -> None:
        """Build variables for small rectangle dimensions and rotation."""
        # Dimensions in mother rectangle space
        self.variables['small_rect_width_mother'] = {}
        self.variables['small_rect_height_mother'] = {}
        
        # Dimensions in point space (after potential rotation)
        self.variables['small_rect_width_point'] = {}
        self.variables['small_rect_height_point'] = {}
        
        # Rotation variables
        self.variables['small_rect_rotated'] = {}
        
        for i in range(self.max_rectangle_number):
            self.variables['small_rect_width_mother'][i] = {}
            self.variables['small_rect_height_mother'][i] = {}
            self.variables['small_rect_width_point'][i] = {}
            self.variables['small_rect_height_point'][i] = {}
            self.variables['small_rect_rotated'][i] = {}
            
            for j in range(self.small_rectangles_per_large):
                # Base dimensions (can be 0 if inactive, or >= 300 if active)
                self.variables['small_rect_width_mother'][i][j] = self.model.NewIntVar(
                    0, self.large_rect_width, f'small_rect_{i}_{j}_width_mother')
                self.variables['small_rect_height_mother'][i][j] = self.model.NewIntVar(
                    0, self.large_rect_height, f'small_rect_{i}_{j}_height_mother')
                
                # Point space dimensions (after rotation)
                max_coord = max(max(p[0] for p in self.points) + self.large_rect_width,
                               max(p[1] for p in self.points) + self.large_rect_height)
                
                self.variables['small_rect_width_point'][i][j] = self.model.NewIntVar(
                    0, max_coord, f'small_rect_{i}_{j}_width_point')
                self.variables['small_rect_height_point'][i][j] = self.model.NewIntVar(
                    0, max_coord, f'small_rect_{i}_{j}_height_point')
                
                # Rotation flag
                self.variables['small_rect_rotated'][i][j] = self.model.NewBoolVar(f'small_rect_{i}_{j}_rotated')
    
    def _build_small_rectangle_position_variables(self) -> None:
        """Build position variables for small rectangles."""
        # Positions in mother rectangle space
        self.variables['small_rect_x_mother'] = {}
        self.variables['small_rect_y_mother'] = {}
        
        # Positions in point space
        self.variables['small_rect_x_point'] = {}
        self.variables['small_rect_y_point'] = {}
        
        # Create 2D interval variables for non-overlap constraints
        self.variables['small_rect_interval_x_mother'] = {}
        self.variables['small_rect_interval_y_mother'] = {}
        
        for i in range(self.max_rectangle_number):
            self.variables['small_rect_x_mother'][i] = {}
            self.variables['small_rect_y_mother'][i] = {}
            self.variables['small_rect_x_point'][i] = {}
            self.variables['small_rect_y_point'][i] = {}
            self.variables['small_rect_interval_x_mother'][i] = {}
            self.variables['small_rect_interval_y_mother'][i] = {}
            
            for j in range(self.small_rectangles_per_large):
                # Mother rectangle positions
                self.variables['small_rect_x_mother'][i][j] = self.model.NewIntVar(
                    0, self.large_rect_width, f'small_rect_{i}_{j}_x_mother')
                self.variables['small_rect_y_mother'][i][j] = self.model.NewIntVar(
                    0, self.large_rect_height, f'small_rect_{i}_{j}_y_mother')
                
                # Point space positions
                min_coord = min(min(p[0] for p in self.points) - self.large_rect_width,
                               min(p[1] for p in self.points) - self.large_rect_height)
                max_coord = max(max(p[0] for p in self.points) + self.large_rect_width,
                               max(p[1] for p in self.points) + self.large_rect_height)
                
                self.variables['small_rect_x_point'][i][j] = self.model.NewIntVar(
                    min_coord, max_coord, f'small_rect_{i}_{j}_x_point')
                self.variables['small_rect_y_point'][i][j] = self.model.NewIntVar(
                    min_coord, max_coord, f'small_rect_{i}_{j}_y_point')
                
                # Create interval variables for 2D non-overlap in mother rectangle
                x_end = self.model.NewIntVar(0, self.large_rect_width, f'small_rect_{i}_{j}_x_end_mother')
                y_end = self.model.NewIntVar(0, self.large_rect_height, f'small_rect_{i}_{j}_y_end_mother')
                
                # x_end = x_start + width
                self.model.Add(x_end == self.variables['small_rect_x_mother'][i][j] + 
                              self.variables['small_rect_width_mother'][i][j], 
                              name=f'x_end_constraint_{i}_{j}')
                self.model.Add(y_end == self.variables['small_rect_y_mother'][i][j] + 
                              self.variables['small_rect_height_mother'][i][j], 
                              name=f'y_end_constraint_{i}_{j}')
                
                # Create interval variables
                self.variables['small_rect_interval_x_mother'][i][j] = self.model.NewOptionalIntervalVar(
                    self.variables['small_rect_x_mother'][i][j],
                    self.variables['small_rect_width_mother'][i][j],
                    x_end,
                    self.variables['small_rect_active'][i][j],
                    f'small_rect_{i}_{j}_interval_x_mother'
                )
                
                self.variables['small_rect_interval_y_mother'][i][j] = self.model.NewOptionalIntervalVar(
                    self.variables['small_rect_y_mother'][i][j],
                    self.variables['small_rect_height_mother'][i][j],
                    y_end,
                    self.variables['small_rect_active'][i][j],
                    f'small_rect_{i}_{j}_interval_y_mother'
                )
    
    def _build_point_assignment_variables(self) -> None:
        """Build variables for point-to-rectangle assignment."""
        self.variables['point_covered_by'] = {}
        for p_idx, point in enumerate(self.points):
            self.variables['point_covered_by'][p_idx] = {}
            for i in range(self.max_rectangle_number):
                self.variables['point_covered_by'][p_idx][i] = {}
                for j in range(self.small_rectangles_per_large):
                    self.variables['point_covered_by'][p_idx][i][j] = self.model.NewBoolVar(
                        f'point_{p_idx}_covered_by_{i}_{j}')
    
    def _build_rectangle_activation_constraints(self) -> None:
        """Build constraints linking rectangle activation."""
        for i in range(self.max_rectangle_number):
            # Large rectangle is active if any small rectangle is active
            small_rects = [self.variables['small_rect_active'][i][j] 
                          for j in range(self.small_rectangles_per_large)]
            
            # large_rect_active[i] == 1 iff at least one small rectangle is active
            self.model.Add(self.variables['large_rect_active'][i] <= sum(small_rects), 
                          name=f'large_rect_activation_upper_{i}')
            
            # If any small rectangle is active, large rectangle must be active
            for j in range(self.small_rectangles_per_large):
                self.model.Add(self.variables['large_rect_active'][i] >= 
                              self.variables['small_rect_active'][i][j], 
                              name=f'large_rect_activation_lower_{i}_{j}')
    
    def _build_geometry_constraints(self) -> None:
        """Build constraints for rectangle geometry and rotation."""
        for i in range(self.max_rectangle_number):
            for j in range(self.small_rectangles_per_large):
                active = self.variables['small_rect_active'][i][j]
                w_mother = self.variables['small_rect_width_mother'][i][j]
                h_mother = self.variables['small_rect_height_mother'][i][j]
                w_point = self.variables['small_rect_width_point'][i][j]
                h_point = self.variables['small_rect_height_point'][i][j]
                rotated = self.variables['small_rect_rotated'][i][j]
                
                # If inactive, dimensions are 0
                self.model.Add(w_mother == 0).OnlyEnforceIf(active.Not())
                self.model.Add(h_mother == 0).OnlyEnforceIf(active.Not())
                self.model.Add(w_point == 0).OnlyEnforceIf(active.Not())
                self.model.Add(h_point == 0).OnlyEnforceIf(active.Not())
                
                # If active, minimum dimensions in mother space
                self.model.Add(w_mother >= self.min_small_size).OnlyEnforceIf(active)
                self.model.Add(h_mother >= self.min_small_size).OnlyEnforceIf(active)
                
                # Rotation logic: if not rotated, point dimensions = mother dimensions
                self.model.Add(w_point == w_mother).OnlyEnforceIf(active, rotated.Not())
                self.model.Add(h_point == h_mother).OnlyEnforceIf(active, rotated.Not())
                
                # If rotated, point dimensions are swapped
                self.model.Add(w_point == h_mother).OnlyEnforceIf(active, rotated)
                self.model.Add(h_point == w_mother).OnlyEnforceIf(active, rotated)
                
                # Rectangle must fit in mother rectangle
                self.model.Add(self.variables['small_rect_x_mother'][i][j] + w_mother <= self.large_rect_width)
                self.model.Add(self.variables['small_rect_y_mother'][i][j] + h_mother <= self.large_rect_height)
    
    def _build_non_overlap_constraints_in_mother_rectangle(self) -> None:
        """Build non-overlap constraints within each mother rectangle using 2D intervals."""
        for i in range(self.max_rectangle_number):
            x_intervals = []
            y_intervals = []
            
            for j in range(self.small_rectangles_per_large):
                x_intervals.append(self.variables['small_rect_interval_x_mother'][i][j])
                y_intervals.append(self.variables['small_rect_interval_y_mother'][i][j])
            
            # Add 2D no overlap constraint
            self.model.AddNoOverlap2D(x_intervals, y_intervals, 
                                     name=f'no_overlap_mother_{i}')
    
    def _build_point_coverage_constraints(self) -> None:
        """Build constraints ensuring all points are covered."""
        for p_idx, (px, py) in enumerate(self.points):
            # Each point must be covered by exactly one rectangle
            coverage_vars = []
            
            for i in range(self.max_rectangle_number):
                for j in range(self.small_rectangles_per_large):
                    coverage_var = self.variables['point_covered_by'][p_idx][i][j]
                    coverage_vars.append(coverage_var)
                    
                    active = self.variables['small_rect_active'][i][j]
                    x_point = self.variables['small_rect_x_point'][i][j]
                    y_point = self.variables['small_rect_y_point'][i][j]
                    w_point = self.variables['small_rect_width_point'][i][j]
                    h_point = self.variables['small_rect_height_point'][i][j]
                    
                    # If point is covered by this rectangle, rectangle must be active
                    self.model.Add(active >= coverage_var, 
                                  name=f'coverage_implies_active_{p_idx}_{i}_{j}')
                    
                    # If covered, point must be within rectangle bounds
                    self.model.Add(x_point <= px).OnlyEnforceIf(coverage_var)
                    self.model.Add(y_point <= py).OnlyEnforceIf(coverage_var)
                    self.model.Add(x_point + w_point >= px + 1).OnlyEnforceIf(coverage_var)
                    self.model.Add(y_point + h_point >= py + 1).OnlyEnforceIf(coverage_var)
            
            # Each point must be covered by at least one rectangle
            self.model.Add(sum(coverage_vars) >= 1, name=f'point_{p_idx}_covered')
    
    def _build_forbidden_interval_constraints(self) -> None:
        """Build constraints to avoid forbidden y-intervals."""
        # Create allowed domain for y coordinates
        if not self.forbidden_y_intervals:
            return
        
        # Sort forbidden intervals
        sorted_intervals = sorted(self.forbidden_y_intervals)
        
        # Create domain excluding forbidden intervals
        min_y = min(p[1] for p in self.points) - self.large_rect_height
        max_y = max(p[1] for p in self.points) + self.large_rect_height
        
        allowed_ranges = []
        current_y = min_y
        
        for start, end in sorted_intervals:
            if current_y < start:
                allowed_ranges.append((current_y, start - 1))
            current_y = max(current_y, end + 1)
        
        if current_y <= max_y:
            allowed_ranges.append((current_y, max_y))
        
        # Create domain from allowed ranges
        domain = _cp.Domain.FromFlatIntervals([item for range_tuple in allowed_ranges 
                                             for item in range_tuple])
        
        # Apply domain to all y-coordinates in point space
        for i in range(self.max_rectangle_number):
            for j in range(self.small_rectangles_per_large):
                y_point = self.variables['small_rect_y_point'][i][j]
                h_point = self.variables['small_rect_height_point'][i][j]
                active = self.variables['small_rect_active'][i][j]
                
                # Rectangle's y-coordinate must be in allowed domain when active
                self.model.AddLinearExpressionInDomain(y_point, domain)\
                    .OnlyEnforceIf(active)
                
                # Rectangle's top edge must also avoid forbidden intervals
                y_top = self.model.NewIntVar(min_y, max_y + self.large_rect_height, 
                                           f'y_top_{i}_{j}')
                self.model.Add(y_top == y_point + h_point - 1).OnlyEnforceIf(active)
                self.model.AddLinearExpressionInDomain(y_top, domain)\
                    .OnlyEnforceIf(active)
    
    def _build_objective(self) -> None:
        """Build the objective to minimize active large rectangles."""
        total_active_large_rects = sum(self.variables['large_rect_active'][i] 
                                     for i in range(self.max_rectangle_number))
        self.model.Minimize(total_active_large_rects, name='minimize_active_rectangles')
    
    def solve(self, solver: Optional[_cp.CpSolver] = None, **solver_params) -> Tuple[int, Dict[str, Any]]:
        """
        Solve the optimization model.
        
        Args:
            solver: Optional solver instance
            **solver_params: Additional solver parameters
            
        Returns:
            Tuple of (status, solution_dict)
        """
        solver = solver or _cp.CpSolver()
        
        # Set default parameters for better performance
        default_params = {
            'max_time_in_seconds': 300.0,
            'enumerate_all_solutions': False
        }
        default_params.update(solver_params)
        
        # Apply parameters
        for param_name, value in default_params.items():
            if hasattr(solver.parameters, param_name):
                setattr(solver.parameters, param_name, value)
        
        status = solver.Solve(self.model)
        
        solution = {}
        if status in [_cp.OPTIMAL, _cp.FEASIBLE]:
            solution = self._extract_solution(solver)
        
        return status, solution
    
    def _extract_solution(self, solver: _cp.CpSolver) -> Dict[str, Any]:
        """Extract solution from solved model."""
        solution = {
            'objective_value': solver.ObjectiveValue(),
            'active_large_rectangles': [],
            'small_rectangles': [],
            'point_assignments': {}
        }
        
        # Extract active large rectangles
        for i in range(self.max_rectangle_number):
            if solver.Value(self.variables['large_rect_active'][i]):
                solution['active_large_rectangles'].append(i)
        
        # Extract small rectangle details
        for i in range(self.max_rectangle_number):
            for j in range(self.small_rectangles_per_large):
                if solver.Value(self.variables['small_rect_active'][i][j]):
                    rect_info = {
                        'large_rect_id': i,
                        'small_rect_id': j,
                        'x_mother': solver.Value(self.variables['small_rect_x_mother'][i][j]),
                        'y_mother': solver.Value(self.variables['small_rect_y_mother'][i][j]),
                        'width_mother': solver.Value(self.variables['small_rect_width_mother'][i][j]),
                        'height_mother': solver.Value(self.variables['small_rect_height_mother'][i][j]),
                        'x_point': solver.Value(self.variables['small_rect_x_point'][i][j]),
                        'y_point': solver.Value(self.variables['small_rect_y_point'][i][j]),
                        'width_point': solver.Value(self.variables['small_rect_width_point'][i][j]),
                        'height_point': solver.Value(self.variables['small_rect_height_point'][i][j]),
                        'rotated': solver.Value(self.variables['small_rect_rotated'][i][j])
                    }
                    solution['small_rectangles'].append(rect_info)
        
        # Extract point assignments
        for p_idx in range(len(self.points)):
            covering_rects = []
            for i in range(self.max_rectangle_number):
                for j in range(self.small_rectangles_per_large):
                    if solver.Value(self.variables['point_covered_by'][p_idx][i][j]):
                        covering_rects.append((i, j))
            solution['point_assignments'][p_idx] = covering_rects
        
        return solution
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the model."""
        if self.model is None:
            return {"error": "Model not built"}
        return self.model.summary()


# Example usage
def example_usage():
    """Example of how to use the RectanglePackingOptimizer."""
    # Define some sample points
    points = [(100, 500), (200, 600), (150, 750), (800, 300), (900, 1200)]
    
    # Define forbidden y-intervals
    forbidden_intervals = [(400, 450), (800, 850)]
    
    # Create and solve the model
    optimizer = RectanglePackingOptimizer(
        points=points,
        forbidden_y_intervals=forbidden_intervals,
        max_rectangle_number=3
    )
    
    status, solution = optimizer.solve(max_time_in_seconds=60.0)
    
    print(f"Solver status: {status}")
    if status in [_cp.OPTIMAL, _cp.FEASIBLE]:
        print(f"Minimum rectangles needed: {solution['objective_value']}")
        print(f"Active large rectangles: {solution['active_large_rectangles']}")
        print(f"Number of small rectangles used: {len(solution['small_rectangles'])}")
    
    return optimizer, status, solution


if __name__ == "__main__":
    example_usage()