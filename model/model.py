# model.py
from .variables   import _VariablesMixin
from .constraints import _ConstraintsMixin
from .objectives  import _ObjectivesMixin
from .debug       import _DebugMixin
from .io          import _IOMixin
import ortools.sat.python.cp_model as _cp

class EnhancedCpModel(
    _VariablesMixin,
    _ConstraintsMixin,
    _ObjectivesMixin,
    _DebugMixin,
    _IOMixin,
    _cp.CpModel,
):
    """Drop-in replacement for CpModel with rich debugging, cloning and I/O."""

    def __init__(self) -> None:
        super().__init__()
        self._constraint_counter = 0
        self._variable_counter = 0

    # ------------------------------------------------------------------
    # Global reset helper
    # ------------------------------------------------------------------
    def clear_model(self) -> None:
        """
        Reset *everything*: OR-Tools proto, Python registries, counters.
        Called internally by import / clone.
        """
        # 1. new proto
        self.__dict__.update(_cp.CpModel().__dict__)

        # 2. Python-side containers
        self._variables.clear()
        self._constraints.clear()
        self._objectives.clear()

        # 3. counters
        self._constraint_counter = 0
        self._variable_counter = 0