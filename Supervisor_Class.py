from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions, solve, statistics, matrix_functions

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Big_Class import Big_Class
    from Network_Structure import Network_Structure


# ===================================================
# Supervisor - training class
# ===================================================