"""Physical and material parameters for resistor-network evolution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import ExperimentConfig


class User_Variables:
    """Store physical parameters that remain fixed during a simulation."""

    def __init__(self, config: ExperimentConfig) -> None:
        variables = config.Variabs
        self.gamma: NDArray[np.float_] = np.asarray(variables.gamma, dtype=float).copy()
        self.R_update: str = variables.R_update
        self.normalize_step: bool = variables.normalize_step
        self.R_max: float = variables.R_max
        self.R_min: float = variables.R_min
        self.reset_thresh_b: float = 1e4
        self.reset_thresh_s: float = -1e4

        if self.R_update in {"deltaR_propto_dp_decay", "deltaR_propto_dp_nonlin_decay"}:
            self.decay: float = variables.decay_R
        if variables.hysteresis:
            self.hysteresis: bool = True
            self.hyst_thresh: float = variables.hysteresis
        else:
            self.hysteresis = False

        self.p_thresh: float = 0.1
        self.bc_noise: float = 0.0
