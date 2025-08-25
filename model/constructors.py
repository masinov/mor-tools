from ortools.sat.python import cp_model as _cp
from typing import List, Union, Tuple

from constructor_tools import (
    NewGreaterOrEqualBoolVar,
    NewLessOrEqualBoolVar,
    NewGreaterBoolVar,
    NewLessBoolVar,
    NewEqualBoolVar,
    NewNotEqualBoolVar,
    NewAndBoolVar,
    NewOrBoolVar,
    NewPointInIntervalBoolVar,
    NewOverlapBoolVar,
    NewContainedInBoolVar,
    NewMinSubjectToBools,
    NewMaxSubjectToBools,
    NewOrSubjectToBools,
    NewAndSubjectToBools,
)

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
