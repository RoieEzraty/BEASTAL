"""Physical and material parameters for resistor-network evolution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import ExperimentConfig


class User_Variables:
    """Store physical parameters that remain fixed during a simulation."""

    def __init__(self, config: ExperimentConfig) -> None:
        Variabs = config.Variabs
        self.gamma: NDArray[np.float_] = np.asarray(Variabs.gamma, dtype=float).copy()
        self.R_update: str = Variabs.R_update
        self.normalize_step: bool = Variabs.normalize_step
        self.R_max: float = Variabs.R_max
        self.R_min: float = Variabs.R_min
        self.reset_thresh_b: float = 1e4
        self.reset_thresh_s: float = -1e4

        if self.R_update in {"deltaR_propto_dp_decay", "deltaR_propto_dp_nonlin_decay"}:
            self.decay: float = Variabs.decay_R
        elif self.R_update == "deltaR_NTC":
            self.C_T: float = Variabs.C_T
            self.G_T: float = Variabs.G_T
            self.T_room: float = Variabs.T_room
            self.R_25: float = Variabs.R_25
            self.B: float = Variabs.B
            self.maximal_current: float = Variabs.maximal_current
            if Variabs.dt_upper <= 0 or Variabs.dt_lower <= 0:
                raise ValueError("dt_upper and dt_lower must be positive")
            if Variabs.dt_upper < Variabs.dt_lower:
                raise ValueError("dt_upper must be greater than or equal to dt_lower")
            if Variabs.euler_steps <= 0:
                raise ValueError("euler_steps must be positive")
            self.dt_upper: float = Variabs.dt_upper
            self.dt_lower: float = Variabs.dt_lower
            self.dt: float = self.dt_upper
            self.euler_steps: int = Variabs.euler_steps
        if Variabs.hysteresis:
            self.hysteresis: bool = True
            self.hyst_thresh: float = Variabs.hysteresis
        else:
            self.hysteresis = False

        self.p_thresh: float = 0.1
        self.bc_noise: float = 0.0
