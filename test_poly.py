from ortools.sat.python import cp_model
from shapely.geometry import Polygon, Point
import random

class RectVar:
    """
    Represents a rectangle with continuous positions (discretized for solver)
    """
    def __init__(self, model, name, width, height, x_range, y_range):
        self.model = model
        self.name = name
        self.width = width
        self.height = height
        self.x = model.NewIntVar(x_range[0], x_range[1], f"{name}_x")
        self.y = model.NewIntVar(y_range[0], y_range[1], f"{name}_y")
        
    def corners(self):
        """Return expressions for rectangle corners"""
        return [(self.x, self.y),
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
                (self.x, self.y + self.height)]
    
    def contains_point_constraint(self, px, py):
        """Return a BooleanVar that is 1 if rectangle contains point (px, py)"""
        b = self.model.NewBoolVar(f"{self.name}_contains_{px}_{py}")
        # x <= px <= x+width AND y <= py <= y+height
        self.model.Add(self.x <= px).OnlyEnforceIf(b)
        self.model.Add(px <= self.x + self.width).OnlyEnforceIf(b)
        self.model.Add(self.y <= py).OnlyEnforceIf(b)
        self.model.Add(py <= self.y + self.height).OnlyEnforceIf(b)
        return b
    
    def non_overlap_constraint(self, other):
        """Ensure this rectangle does not overlap another rectangle"""
        # Boolean OR of 4 possible separating axes
        b1 = self.model.NewBoolVar(f"{self.name}_left_of_{other.name}")
        b2 = self.model.NewBoolVar(f"{self.name}_right_of_{other.name}")
        b3 = self.model.NewBoolVar(f"{self.name}_above_{other.name}")
        b4 = self.model.NewBoolVar(f"{self.name}_below_{other.name}")
        
        self.model.Add(self.x + self.width <= other.x).OnlyEnforceIf(b1)
        self.model.Add(other.x + other.width <= self.x).OnlyEnforceIf(b2)
        self.model.Add(self.y + self.height <= other.y).OnlyEnforceIf(b3)
        self.model.Add(other.y + other.height <= self.y).OnlyEnforceIf(b4)
        
        self.model.AddBoolOr([b1, b2, b3, b4])


def discretized_polygon_cover_example():
    # Example polygon (oblique sides)
    polygon_points = [(0,0), (6,2), (5,6), (1,5)]
    poly = Polygon(polygon_points)
    
    # Sample points inside polygon
    n_samples = 50
    minx, miny, maxx, maxy = poly.bounds
    sample_points = []
    while len(sample_points) < n_samples:
        px = random.uniform(minx, maxx)
        py = random.uniform(miny, maxy)
        if poly.contains(Point(px, py)):
            sample_points.append((px, py))

    # Model
    model = cp_model.CpModel()
    
    # Create 10 rectangles of size 1x1
    rects = []
    for i in range(10):
        r = RectVar(model, f"rect{i}", width=1, height=1,
                    x_range=(int(minx)-1, int(maxx)+1),
                    y_range=(int(miny)-1, int(maxy)+1))
        rects.append(r)

    # Coverage constraints: each sample point must be covered by at least one rectangle
    for px, py in sample_points:
        px_int = int(round(px))
        py_int = int(round(py))
        contains_vars = [r.contains_point_constraint(px_int, py_int) for r in rects]
        model.AddBoolOr(contains_vars)
    
    # Optional: prevent rectangles overlapping
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            rects[i].non_overlap_constraint(rects[j])
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    
    print("Status:", solver.StatusName(status))
    for r in rects:
        print(f"{r.name}: x={solver.Value(r.x)}, y={solver.Value(r.y)}")

if __name__ == "__main__":
    discretized_polygon_cover_example()
