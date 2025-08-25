# model.py
from variables     import _VariablesMixin
from constraints   import _ConstraintsMixin
from objectives    import _ObjectivesMixin
from debug         import _DebugMixin
from import_export import _IEMixin
from regen         import _RegenMixin
import ortools.sat.python.cp_model as _cp

class EnhancedCpModel(
    _VariablesMixin,
    _ConstraintsMixin,
    _ObjectivesMixin,
    _RegenMixin,
    _DebugMixin,
    _IEMixin,
    _cp.CpModel,
):
    """Drop-in replacement for CpModel with rich debugging, cloning and I/O."""

    def __init__(self) -> None:
        super().__init__()
        self._constraint_counter = 0
        self._variable_counter = 0