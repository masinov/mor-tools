from ortools.sat.python import cp_model
import shapely.geometry as geom

# ------------------------
# Helper: point-in-rect (linearized)
# ------------------------
def inside_rect(model, px, py, x, y, w, h):
    in_x = model.NewBoolVar("in_x")
    in_y = model.NewBoolVar("in_y")
    model.Add(px >= x).OnlyEnforceIf(in_x)
    model.Add(px < x + w).OnlyEnforceIf(in_x)
    model.Add(py >= y).OnlyEnforceIf(in_y)
    model.Add(py < y + h).OnlyEnforceIf(in_y)
    return in_x, in_y

# ------------------------
# Build model
# ------------------------
def polygon_cover_example():
    model = cp_model.CpModel()

    # Example polygon: rectangle (0,0)-(3000,1500)
    poly = geom.Polygon([(0,0),(3000,0),(3000,1500),(0,1500)])

    # Forbidden y-intervals (example)
    forbidden = [(700, 900)]

    num_mothers = 2
    max_children_per_mother = 3

    # Vars
    mothers_active = []
    children = []

    for m in range(num_mothers):
        act = model.NewBoolVar(f"mother_{m}_active")
        mothers_active.append(act)

        for c in range(max_children_per_mother):
            # Placement in polygon
            x = model.NewIntVar(0, 3000, f"x_m{m}c{c}")
            y = model.NewIntVar(0, 1500, f"y_m{m}c{c}")
            # Size
            w = model.NewIntVar(300, 1000, f"w_m{m}c{c}")
            h = model.NewIntVar(300, 2000, f"h_m{m}c{c}")
            # Placement inside mother board (aux)
            u = model.NewIntVar(0, 1000, f"u_m{m}c{c}")
            v = model.NewIntVar(0, 2000, f"v_m{m}c{c}")

            # Ensure fits inside mother
            model.Add(u + w <= 1000)
            model.Add(v + h <= 2000)

            # Forbidden y intervals
            for (y0, y1) in forbidden:
                model.Add(y < y0).OnlyEnforceIf(act)
                model.Add(y + h > y1).OnlyEnforceIf(act.Not())

            children.append((x,y,w,h,u,v,act))

    # Coverage via sampled points
    samples = []
    for px in range(0, 3001, 500):
        for py in range(0, 1501, 500):
            if poly.contains(geom.Point(px,py)):
                samples.append((px,py))

    for (px,py) in samples:
        covered_bools = []
        for (x,y,w,h,u,v,act) in children:
            in_x = model.NewBoolVar(f"px{px}py{py}_in_x")
            in_y = model.NewBoolVar(f"px{px}py{py}_in_y")
            model.Add(px >= x).OnlyEnforceIf(in_x)
            model.Add(px < x + w).OnlyEnforceIf(in_x)
            model.Add(py >= y).OnlyEnforceIf(in_y)
            model.Add(py < y + h).OnlyEnforceIf(in_y)
            inside = model.NewBoolVar(f"inside_px{px}py{py}")
            model.AddBoolAnd([in_x,in_y,act]).OnlyEnforceIf(inside)
            model.AddBoolOr([inside.Not()]).OnlyEnforceIf(inside.Not())
            covered_bools.append(inside)
        model.AddBoolOr(covered_bools)

    # Objective: minimize active mothers
    model.Minimize(sum(mothers_active))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    res = solver.Solve(model)

    print("Status:", solver.StatusName(res))
    for (x,y,w,h,u,v,act) in children:
        if solver.Value(act):
            print(f"Rect at ({solver.Value(x)},{solver.Value(y)}) size {solver.Value(w)}x{solver.Value(h)}")

if __name__ == "__main__":
    polygon_cover_example()
