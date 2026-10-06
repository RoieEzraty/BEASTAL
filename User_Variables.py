"""Physical and material parameters for resistor-network evolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from config import ExperimentConfig

if TYPE_CHECKING:
    from Network_Structure import Network_Structure


class User_Variables:
    """Store physical parameters that remain fixed during a simulation."""

    def __init__(self, config: ExperimentConfig, Strctr: "Network_Structure") -> None:
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
            self.R_25: NDArray[np.float_] = np.random.normal(Variabs.R_25, Variabs.noise_to_R_25, Strctr.NE)
            self.B: NDArray[np.float_] = np.random.normal(Variabs.B, Variabs.noise_to_B, Strctr.NE)
            self.maximal_current: float = Variabs.maximal_current
            if Variabs.dt_upper < Variabs.dt_lower:
                raise ValueError("dt_upper must be greater than or equal to dt_lower")
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
