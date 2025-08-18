from ortools.sat.python import cp_model as _cp
from typing import Union, List, Tuple

from model import EnhancedCpModel

M = 9999999999999

def NewGreaterOrEqualBoolVar(
    model: EnhancedCpModel, 
    variable: _cp.IntVar, 
    threshold: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is greater or equal to a given threshold, and 0 when it is strictly less than the threshold.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the threshold.
    - threshold (Union[_cp.IntVar, int]): The threshold value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is greater than 
      or equal to `threshold`, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> threshold_value = 5
    >>> bool_variable = NewGreaterOrEqualBoolVar(model, x, threshold_value, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """

    tag = f"{name}_internals"
    constructor_tag = "GreaterOrEqualBoolVar"

    geq_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_geq_cond"
    c2_name = f"{name}_lt_cond"
    c1 = model.Add(variable >= threshold, name=c1_name).OnlyEnforceIf(geq_var)
    c2 = model.Add(variable < threshold, name=c2_name).OnlyEnforceIf(geq_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return geq_var

def NewLessOrEqualBoolVar(
    model: EnhancedCpModel, 
    variable: _cp.IntVar, 
    threshold: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is less than or equal to a given threshold, and 0 when it is strictly greater than the threshold.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the threshold.
    - threshold (Union[_cp.IntVar, int]): The threshold value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is less than 
      or equal to `threshold`, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> threshold_value = 5
    >>> bool_variable = NewLessOrEqualBoolVar(model, x, threshold_value, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "LessOrEqualBoolVar"

    leq_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_leq_cond"
    c2_name = f"{name}_gt_cond"
    c1 = model.Add(variable <= threshold, name=c1_name).OnlyEnforceIf(leq_var)
    c2 = model.Add(variable > threshold, name=c2_name).OnlyEnforceIf(leq_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return leq_var

def NewGreaterBoolVar(
    model: EnhancedCpModel, 
    variable: _cp.IntVar, 
    threshold: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is strictly greater than a given threshold, and 0 when it is less than or equal to the threshold.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the threshold.
    - threshold (Union[_cp.IntVar, int]): The threshold value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is strictly greater than 
      `threshold`, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> threshold_value = 5
    >>> bool_variable = NewGreaterBoolVar(model, x, threshold_value, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "GreaterBoolVar"

    g_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_gt_cond"
    c2_name = f"{name}_leq_cond"
    c1 = model.Add(variable > threshold, name=c1_name).OnlyEnforceIf(g_var)
    c2 = model.Add(variable <=  threshold, name=c2_name).OnlyEnforceIf(g_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return g_var

def NewLessBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar, 
    threshold: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is strictly less than a given threshold, and 0 when it is greater than or equal to the threshold.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the threshold.
    - threshold (Union[_cp.IntVar, int]): The threshold value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is strictly less than 
      `threshold`, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> threshold_value = 5
    >>> bool_variable = NewLessBoolVar(model, x, threshold_value, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "LessBoolVar"

    l_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_lt_cond"
    c2_name = f"{name}_geq_cond"
    c1 = model.Add(variable < threshold, name=c1_name).OnlyEnforceIf(l_var)
    c2 = model.Add(variable >=  threshold, name=c2_name).OnlyEnforceIf(l_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return l_var

def NewEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar, 
    value: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is equal to a given value, and 0 when it is not equal to the value.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the value.
    - value (Union[_cp.IntVar, int]): The value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is equal to `value`, 
      and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> value_to_compare = 5
    >>> bool_variable = NewEqualBoolVar(model, x, value_to_compare, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.

    Note:
    The created boolean variable ensures that the specified integer variable meets the specified condition.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "EqualBoolVar"

    eq_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_eq_cond"
    c2_name = f"{name}_neq_cond"
    c1 = model.Add(variable == value, name=c1_name).OnlyEnforceIf(eq_var)
    c2 = model.Add(variable != value, name=c2_name).OnlyEnforceIf(eq_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return eq_var

def NewNotEqualBoolVar(
    model: EnhancedCpModel,
    variable: _cp.IntVar, 
    value: Union[_cp.IntVar, int], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the specified integer variable 
    is not equal to a given value, and 0 when it is equal to the value.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (_cp.IntVar): The integer variable to compare against the value.
    - value (Union[_cp.IntVar, int]): The value for the comparison. 
      Can be an integer or an integer variable.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is not equal to `value`, 
      and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> value_to_compare = 5
    >>> bool_variable = NewNotEqualBoolVar(model, x, value_to_compare, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "NotEqualBoolVar"

    neq_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_neq_cond"
    c2_name = f"{name}_eq_cond"
    c1 = model.Add(variable != value, name=c1_name).OnlyEnforceIf(neq_var)
    c2 = model.Add(variable == value, name=c2_name).OnlyEnforceIf(neq_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return neq_var

def NewAndBoolVar(
    model: EnhancedCpModel, 
    variables: List[_cp.IntVar], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when all the specified boolean variables 
    in the given list are true, and 0 when at least one of them is false.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variables (List[_cp.IntVar]): The list of boolean variables to be combined with a logical AND operation.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when all the boolean variables in `variables` 
      are true, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> bool_var1 = model.NewBoolVar('bool_var1')
    >>> bool_var2 = model.NewBoolVar('bool_var2')
    >>> bool_variables = [bool_var1, bool_var2]
    >>> and_bool_variable = NewAndBoolVar(model, bool_variables, 'and_bool_variable')
    >>> # Use and_bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "AndBoolVar"

    and_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_and_cond"
    c2_name = f"{name}_or_not_cond"
    c1 = model.AddBoolAnd(variables, name=c1_name).OnlyEnforceIf(and_var)
    c2 = model.AddBoolOr([variable.Not() for variable in variables], name=c2_name).OnlyEnforceIf(and_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return and_var

def NewOrBoolVar(
    model: EnhancedCpModel, 
    variables: List[_cp.IntVar], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when at least one of the specified boolean variables 
    in the given list is true, and 0 when all of them are false.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variables (List[_cp.IntVar]): The list of boolean variables to be combined with a logical OR operation.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when at least one boolean variable in `variables` 
      is true, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> bool_var1 = model.NewBoolVar('bool_var1')
    >>> bool_var2 = model.NewBoolVar('bool_var2')
    >>> bool_variables = [bool_var1, bool_var2]
    >>> or_bool_variable = NewOrBoolVar(model, bool_variables, 'or_bool_variable')
    >>> # Use or_bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "OrBoolVar"

    or_var = model.NewBoolVar(name)
    
    c1_name = f"{name}_or_cond"
    c2_name = f"{name}_and_not_cond"
    c1 = model.AddBoolOr(variables, name=c1_name).OnlyEnforceIf(or_var)
    c2 = model.AddBoolAnd([variable.Not() for variable in variables], name=c2_name).OnlyEnforceIf(or_var.Not())
    
    model.add_constraint_tags(c1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c2_name, [tag, constructor_tag, "Internal"])
    
    return or_var

def NewPointInIntervalBoolVar(
    model: EnhancedCpModel, 
    variable: Union[_cp.IntVar, int],
    interval: Union[Tuple[int, int], _cp.IntervalVar],
    name: str
) -> _cp.IntVar:
  
    """
    Creates a boolean variable in a CP-SAT model that is 1 when a specified integer variable lies within a given interval, and 0 otherwise.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - variable (Union[_cp.IntVar, int]): The integer variable for which the interval check is performed.
    - interval (Union[Tuple[int, int], _cp.IntervalVar]): A tuple representing the inclusive interval bounds [lower_bound, upper_bound].
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `variable` is within the interval, and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> x = model.NewIntVar(0, 10, 'x')
    >>> threshold_value = 5
    >>> bool_variable = NewPointInIntervalBoolVar(model, x, threshold_value, 'bool_variable')
    >>> # Use bool_variable in constraints and objectives of the CP-SAT model.
    """

    tag = f"{name}_internals"
    constructor_tag = "PointInIntervalBoolVar"

    if isinstance(variable, int):
        value = variable
        variable = model.NewIntVar(value, value, f"{name}_const")
    
    if isinstance(interval, Tuple):
        lb, ub = interval
    else:
        lb = interval.StartExpr()
        ub = interval.EndExpr()

    geq_name = f"{name}_geq_{lb}"
    leq_name = f"{name}_leq_{ub}"
    
    geq_var = NewGreaterOrEqualBoolVar(model, variable, lb, geq_name)
    leq_var = NewLessOrEqualBoolVar(model, variable, ub, leq_name)
    and_var = NewAndBoolVar(model, [geq_var, leq_var], name)

    model.add_constraint_tags(geq_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(leq_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(name, [tag, constructor_tag, "Internal"])
    
    return and_var

def NewOverlapBoolVar(
    model: EnhancedCpModel, 
    interval1: Union[_cp.IntervalVar, Tuple[int, int]], 
    interval2: Union[_cp.IntervalVar, Tuple[int, int]], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when two intervals overlap, and 0 when they do not.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - interval1 (Union[_cp.IntervalVar, Tuple[int, int]]): The first interval variable or tuple for the overlap check.
    - interval2 (Union[_cp.IntervalVar, Tuple[int, int]]): The second interval variable or tuple for the overlap check.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `interval1` and `interval2` overlap, 
      and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> interval1 = model.NewIntervalVar(0, 5, 5, 'interval1')
    >>> interval2 = model.NewIntervalVar(4, 4, 8, 'interval2')
    >>> overlap_bool_variable = NewOverlapBoolVar(model, interval1, interval2, 'overlap_bool_variable')
    >>> # Use overlap_bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "OverlapBoolVar"

    assert isinstance(interval1, _cp.IntervalVar) or isinstance(interval2, _cp.IntervalVar), "At least one of the intervals must be a cp_model.IntervalVar."

    interval1_start = interval1.StartExpr() if isinstance(interval1, _cp.IntervalVar) else interval1[0]
    interval1_end = interval1.EndExpr() if isinstance(interval1, _cp.IntervalVar) else interval1[1]
    interval2_start = interval2.StartExpr() if isinstance(interval2, _cp.IntervalVar) else interval2[0]
    interval2_end = interval2.EndExpr() if isinstance(interval2, _cp.IntervalVar) else interval2[1]
    
    overlap_var = model.NewBoolVar(name)

    i1_start_in_i2_name = f"{name}_i1_start_in_i2"
    i1_start_in_i2 = NewPointInIntervalBoolVar(model, interval1_start, (interval2_start, interval2_end), i1_start_in_i2_name)

    i1_end_in_i2_name = f"{name}_i1_end_in_i2"
    i1_end_in_i2 = NewPointInIntervalBoolVar(model, interval1_end, (interval2_start, interval2_end), i1_end_in_i2_name)
    
    i2_start_in_i1_name = f"{name}_i2_start_in_i1"
    i2_start_in_i1 = NewPointInIntervalBoolVar(model, interval2_start, (interval1_start, interval1_end), i2_start_in_i1_name)

    i2_end_in_i1_name = f"{name}_i2_end_in_i1"
    i2_end_in_i1 = NewPointInIntervalBoolVar(model, interval2_end, (interval1_start, interval1_end), i2_end_in_i1_name)
    
    c_max_name = f"{name}_max_overlap"
    c_max = model.AddMaxEquality(overlap_var, [i1_start_in_i2, i1_end_in_i2, i2_start_in_i1, i2_end_in_i1], name=c_max_name)

    model.add_constraint_tags(i1_start_in_i2_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(i1_end_in_i2_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(i2_start_in_i1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(i2_end_in_i1_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c_max_name, [tag, constructor_tag, "Internal"])
    
    return overlap_var

def NewContainedInBoolVar(
    model: EnhancedCpModel, 
    interval1: Union[_cp.IntervalVar, Tuple[int, int]], 
    interval2: Union[_cp.IntervalVar, Tuple[int, int]], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable in a CP-SAT model that is 1 when the first interval is contained in the second interval, and 0 when it is not.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - interval1 (Union[_cp.IntervalVar, Tuple[int, int]]): The first interval variable or tuple for the containment check.
    - interval2 (Union[_cp.IntervalVar, Tuple[int, int]]): The second interval variable or tuple for the containment check.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable that takes the value 1 when `interval1` is contained in `interval2`, 
      and 0 otherwise.

    Examples:
    >>> model = EnhancedCpModel()
    >>> interval1 = model.NewIntervalVar(0, 5, 5, 'interval1')
    >>> interval2 = model.NewIntervalVar(4, 4, 8, 'interval2')
    >>> contained_bool_variable = NewContainedInBoolVar(model, interval1, interval2, 'contained_in_bool_variable')
    >>> # Use contained_bool_variable in constraints and objectives of the CP-SAT model.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "ContainedInBoolVar"

    assert isinstance(interval1, _cp.IntervalVar) or isinstance(interval2, _cp.IntervalVar), "At least one of the intervals must be a cp_model.IntervalVar."
    
    interval1_start = interval1.StartExpr() if isinstance(interval1, _cp.IntervalVar) else interval1[0]
    interval1_end = interval1.EndExpr() if isinstance(interval1, _cp.IntervalVar) else interval1[1]
    interval2_start = interval2.StartExpr() if isinstance(interval2, _cp.IntervalVar) else interval2[0]
    interval2_end = interval2.EndExpr() if isinstance(interval2, _cp.IntervalVar) else interval2[1]
    
    i1_start_in_i2_name = f"{name}_i1_start_in_i2"
    i1_start_in_i2 = NewPointInIntervalBoolVar(model, interval1_start, (interval2_start, interval2_end), i1_start_in_i2_name)

    i1_end_in_i2_name = f"{name}_i1_end_in_i2"
    i1_end_in_i2 = NewPointInIntervalBoolVar(model, interval1_end, (interval2_start, interval2_end), i1_end_in_i2_name)
    
    contained_in_var = NewAndBoolVar(model, [i1_start_in_i2, i1_end_in_i2], name)

    model.add_constraint_tags(i1_start_in_i2_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(i1_end_in_i2_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(name, [tag, constructor_tag, "Internal"])
    
    return contained_in_var

def NewMinSubjectToBools(
    model: EnhancedCpModel, 
    values: Union[List[_cp.IntVar], List[int]], 
    bools: List[_cp.IntVar], 
    name: str, 
    return_bool_markers: bool = False
) -> _cp.IntVar:
    
    """
    Creates a new integer variable representing the minimum value among a list of integer variables, subject to boolean conditions.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the integer variable is created.
    - values (Union[List[_cp.IntVar], List[int]]): List of integer variables or values whose minimum value needs to be determined.
    - bools (List[_cp.IntVar]): List of boolean variables indicating whether corresponding value variables should be considered in finding the minimum.
    - name (str): The name for the integer variable.
    - return_bool_markers (bool): If True, also return the boolean markers for each value.

    Returns:
    - _cp.IntVar: The integer variable representing the minimum value.
    - Optional[List[_cp.IntVar]]: If return_bool_markers is True, also returns the list of boolean markers.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "MinSubjectToBools"

    assert len(values) == len(bools), "Must provide as many value variables as boolean variables."

    min_var_minimum = -M
    min_var_maximum = M

    var_lower_domains = []
    var_upper_domains = []
    for var in values:
        if isinstance(var, int):
           var_lower_domains.append(var)
           var_upper_domains.append(var)
        else:
          var_domain = var.Proto().domain
          var_lower_domains.append(var_domain[0]) 
          var_upper_domains.append(var_domain[1])

    min_var_minimum = max(min(var_lower_domains), min_var_minimum)
    min_var_maximum = min(max(var_upper_domains), min_var_maximum)

    min_var = model.NewIntVar(min_var_minimum, min_var_maximum, name)

    equal_check_vars = []
    equal_and_check_vars = []
    for i, var in enumerate(values):
        c_le_name = f"{name}_le_cond_{i}"
        c_le = model.Add(min_var <= var, name=c_le_name).OnlyEnforceIf(bools[i])
        eq_check_name = f"{name}_equal_check_{i}"
        equal_check_vars.append(NewEqualBoolVar(model, min_var, var, eq_check_name))
        and_check_name = f"{name}_equal_and_check_{i}"
        equal_and_check_vars.append(NewAndBoolVar(model, [equal_check_vars[i], bools[i]], and_check_name))

        model.add_constraint_tags(c_le_name, [tag, constructor_tag, "Internal"])
        model.add_constraint_tags(eq_check_name, [tag, constructor_tag, "Internal"])
        model.add_constraint_tags(and_check_name, [tag, constructor_tag, "Internal"])

    some_bool_true_name = f"{name}_some_bool_true"
    some_bool_true = NewOrBoolVar(model, bools, some_bool_true_name)
    c_at_least_one_name = f"{name}_at_least_one_cond"
    c_at_least_one = model.Add(sum(equal_and_check_vars) >= 1, name=c_at_least_one_name).OnlyEnforceIf(some_bool_true)

    model.add_constraint_tags(some_bool_true_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c_at_least_one_name, [tag, constructor_tag, "Internal"])

    if return_bool_markers:
        return min_var, equal_and_check_vars

    else:
      return min_var

def NewMaxSubjectToBools(
    model: EnhancedCpModel, 
    values: Union[List[_cp.IntVar], List[int]], 
    bools: List[_cp.IntVar], 
    name: str, 
    return_bool_markers: bool = False
) -> _cp.IntVar:
    
    """
    Creates a new integer variable representing the maximum value among a list of integer variables, subject to boolean conditions.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the integer variable is created.
    - values (Union[List[_cp.IntVar], List[int]]): List of integer variables whose maximum value needs to be determined.
    - bools (List[_cp.IntVar]): List of boolean variables indicating whether corresponding value variables should be considered in finding the maximum.
    - name (str): The name for the integer variable.
    - return_bool_markers (bool): If True, also return the boolean markers for each value.

    Returns:
    - _cp.IntVar: The integer variable representing the maximum value.
    - Optional[List[_cp.IntVar]]: If return_bool_markers is True, also returns the list of boolean markers.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "MaxSubjectToBools"

    assert len(values) == len(bools), "Must provide as many value variables as boolean variables."

    max_var_minimum = -M
    max_var_maximum = M

    var_lower_domains = []
    var_upper_domains = []
    for var in values:
        if isinstance(var, int):
           var_lower_domains.append(var)
           var_upper_domains.append(var)
        else:
          var_domain = var.Proto().domain
          var_lower_domains.append(var_domain[0]) 
          var_upper_domains.append(var_domain[1])

    max_var_minimum = max(min(var_lower_domains), max_var_minimum)
    max_var_maximum = min(max(var_upper_domains), max_var_maximum)

    max_var = model.NewIntVar(max_var_minimum, max_var_maximum, name)

    equal_check_vars = []
    equal_and_check_vars = []
    for i, var in enumerate(values):
        c_ge_name = f"{name}_ge_cond_{i}"
        c_ge = model.Add(max_var >= var, name=c_ge_name).OnlyEnforceIf(bools[i])
        eq_check_name = f"{name}_equal_check_{i}"
        equal_check_vars.append(NewEqualBoolVar(model, max_var, var, eq_check_name))
        and_check_name = f"{name}_equal_and_check_{i}"
        equal_and_check_vars.append(NewAndBoolVar(model, [equal_check_vars[i], bools[i]], and_check_name))

        model.add_constraint_tags(c_ge_name, [tag, constructor_tag, "Internal"])
        model.add_constraint_tags(eq_check_name, [tag, constructor_tag, "Internal"])
        model.add_constraint_tags(and_check_name, [tag, constructor_tag, "Internal"])

    some_bool_true_name = f"{name}_some_bool_true"
    some_bool_true = NewOrBoolVar(model, bools, some_bool_true_name)
    c_at_least_one_name = f"{name}_at_least_one_cond"
    c_at_least_one = model.Add(sum(equal_and_check_vars) >= 1, name=c_at_least_one_name).OnlyEnforceIf(some_bool_true)

    model.add_constraint_tags(some_bool_true_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c_at_least_one_name, [tag, constructor_tag, "Internal"])

    if return_bool_markers:
        return max_var, equal_and_check_vars

    else:
      return max_var

def NewOrSubjectToBools(
    model: EnhancedCpModel, 
    check_bools: List[_cp.IntVar], 
    constraint_bools: List[_cp.IntVar], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable representing the logical OR operation applied to pairs of boolean variables subject to additional constraint boolean variables.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - check_bools (List[_cp.IntVar]): List of boolean variables for checking conditions.
    - constraint_bools (List[_cp.IntVar]): List of boolean variables indicating whether corresponding check_boool variables should be considered.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable representing the logical OR operation applied to pairs of boolean variables.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "OrSubjectToBools"

    assert len(check_bools) == len(constraint_bools), "Must provide as many value variables as boolean variables."

    product_vars = []
    for i in range(len(check_bools)):
        product_name = f"{name}_product_var_{i}"
        product_var = model.NewBoolVar(product_name)
        product_vars.append(product_var)
        c_mult_name = f"{name}_mult_{i}"
        c_mult = model.AddMultiplicationEquality(product_var, [check_bools[i], constraint_bools[i]], name=c_mult_name)
        
        model.add_constraint_tags(c_mult_name, [tag, constructor_tag, "Internal"])

    or_var = NewOrBoolVar(model, product_vars, name)

    return or_var

def NewAndSubjectToBools(
    model: EnhancedCpModel, 
    check_bools: List[_cp.IntVar], 
    constraint_bools: List[_cp.IntVar], 
    name: str
) -> _cp.IntVar:
    
    """
    Creates a boolean variable representing the logical AND operation applied to pairs of boolean variables subject to additional constraint boolean variables.

    Parameters:
    - model (EnhancedCpModel): The CP-SAT model in which the boolean variable is created.
    - check_bools (List[_cp.IntVar]): List of boolean variables for checking conditions.
    - constraint_bools (List[_cp.IntVar]): List of boolean variables indicating whether corresponding check_boool variables should be considered.
    - name (str): The name for the boolean variable.

    Returns:
    - _cp.IntVar: The boolean variable representing the logical AND operation applied to pairs of boolean variables.
    """
    
    tag = f"{name}_internals"
    constructor_tag = "AndSubjectToBools"

    assert len(check_bools) == len(constraint_bools), "Must provide as many value variables as boolean variables."

    product_vars = []
    for i in range(len(check_bools)):
        product_name = f"{name}_product_var_{i}"
        product_var = model.NewBoolVar(product_name)
        product_vars.append(product_var)
        c_mult_name = f"{name}_mult_{i}"
        c_mult = model.AddMultiplicationEquality(product_var, [check_bools[i], constraint_bools[i]], name=c_mult_name)
        
        model.add_constraint_tags(c_mult_name, [tag, constructor_tag, "Internal"])

    product_sum_var = model.NewIntVar(0, len(check_bools), f"{name}_product_sum")
    constraint_bools_sum_var = model.NewIntVar(0, len(check_bools), f"{name}_constraint_bools_sum")
    c_product_sum_name = f"{name}_product_sum_cond"
    c_constraint_sum_name = f"{name}_constraint_sum_cond"
    c_product_sum = model.Add(product_sum_var == sum(product_vars), name=c_product_sum_name)
    c_constraint_sum = model.Add(constraint_bools_sum_var == sum(constraint_bools), name=c_constraint_sum_name)
    
    model.add_constraint_tags(c_product_sum_name, [tag, constructor_tag, "Internal"])
    model.add_constraint_tags(c_constraint_sum_name, [tag, constructor_tag, "Internal"])

    and_var = NewEqualBoolVar(model, product_sum_var, constraint_bools_sum_var, name)

    return and_var

class EnhancedConstructorsMixin:
    """
    Mix-in that exposes helper constructors as *methods* on EnhancedCpModel
    while keeping the implementation external.
    """

    # ---------- comparison booleans ----------
    def NewGreaterOrEqualBoolVar(
        self,
        variable: _cp.IntVar,
        threshold: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewGreaterOrEqualBoolVar(self, variable, threshold, name)

    def NewLessOrEqualBoolVar(
        self,
        variable: _cp.IntVar,
        threshold: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewLessOrEqualBoolVar(self, variable, threshold, name)

    def NewGreaterBoolVar(
        self,
        variable: _cp.IntVar,
        threshold: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewGreaterBoolVar(self, variable, threshold, name)

    def NewLessBoolVar(
        self,
        variable: _cp.IntVar,
        threshold: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewLessBoolVar(self, variable, threshold, name)

    def NewEqualBoolVar(
        self,
        variable: _cp.IntVar,
        value: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewEqualBoolVar(self, variable, value, name)

    def NewNotEqualBoolVar(
        self,
        variable: _cp.IntVar,
        value: Union[_cp.IntVar, int],
        name: str,
    ) -> _cp.IntVar:
        return NewNotEqualBoolVar(self, variable, value, name)

    # ---------- logical helpers ----------
    def NewAndBoolVar(
        self,
        variables: List[_cp.IntVar],
        name: str,
    ) -> _cp.IntVar:
        return NewAndBoolVar(self, variables, name)

    def NewOrBoolVar(
        self,
        variables: List[_cp.IntVar],
        name: str,
    ) -> _cp.IntVar:
        return NewOrBoolVar(self, variables, name)

    # ---------- interval helpers ----------
    def NewPointInIntervalBoolVar(
        self,
        variable: Union[_cp.IntVar, int],
        interval: Union[Tuple[int, int], _cp.IntervalVar],
        name: str,
    ) -> _cp.IntVar:
        return NewPointInIntervalBoolVar(self, variable, interval, name)

    def NewOverlapBoolVar(
        self,
        interval1: Union[_cp.IntervalVar, Tuple[int, int]],
        interval2: Union[_cp.IntervalVar, Tuple[int, int]],
        name: str,
    ) -> _cp.IntVar:
        return NewOverlapBoolVar(self, interval1, interval2, name)

    def NewContainedInBoolVar(
        self,
        interval1: Union[_cp.IntervalVar, Tuple[int, int]],
        interval2: Union[_cp.IntervalVar, Tuple[int, int]],
        name: str,
    ) -> _cp.IntVar:
        return NewContainedInBoolVar(self, interval1, interval2, name)

    # ---------- min / max under boolean masks ----------
    def NewMinSubjectToBools(
        self,
        values: Union[List[_cp.IntVar], List[int]],
        bools: List[_cp.IntVar],
        name: str,
        return_bool_markers: bool = False,
    ):
        return NewMinSubjectToBools(self, values, bools, name, return_bool_markers)

    def NewMaxSubjectToBools(
        self,
        values: Union[List[_cp.IntVar], List[int]],
        bools: List[_cp.IntVar],
        name: str,
        return_bool_markers: bool = False,
    ):
        return NewMaxSubjectToBools(self, values, bools, name, return_bool_markers)

    # ---------- logical OR / AND under boolean masks ----------
    def NewOrSubjectToBools(
        self,
        check_bools: List[_cp.IntVar],
        constraint_bools: List[_cp.IntVar],
        name: str,
    ) -> _cp.IntVar:
        return NewOrSubjectToBools(self, check_bools, constraint_bools, name)

    def NewAndSubjectToBools(
        self,
        check_bools: List[_cp.IntVar],
        constraint_bools: List[_cp.IntVar],
        name: str,
    ) -> _cp.IntVar:
        return NewAndSubjectToBools(self, check_bools, constraint_bools, name)
