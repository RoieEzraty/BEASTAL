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
        self.k_B: float = Variabs.k_B
        self.C: float = Variabs.C
        self.T_room: float = Variabs.T_room

        if self.R_update in {"deltaR_propto_dp_decay", "deltaR_propto_dp_nonlin_decay"}:
            self.decay: float = Variabs.decay_R
        elif self.R_update in {"deltaR_NTC", "grad_desc"}:
            self.C_T: float = Variabs.C_T
            self.G_T: float = Variabs.G_T
            self.NTC: NTC = NTC(Variabs.NTC_type)
            self.NTC_g: NTC = NTC(Variabs.NTC_g_type)
            Strctr.assign_NTCs(self.NTC, self.NTC_g, noise_to_R_25=Variabs.noise_to_R_25, noise_to_B=Variabs.noise_to_B)
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

        self.grad_desc_use: str = Variabs.grad_desc_use
        self.dK_step: float = Variabs.dK_step
        self.dR_step: float = Variabs.dR_step

        self.p_thresh: float = 0.1
        self.bc_noise: float = 0.0

class NTC:
    """Store parameters for NTC resistor model."""
    def __init__(self, type: str) -> None:
        if type == "104BH102HT1":
            self.B: float = 4100  # [K]
            self.R_25: float = 1000.0 # [Ohm] Resistance at 25°C
        elif type == "103EH101HT1":
            self.B: float = 3244  # [K]
            self.R_25: float = 100.0 # [Ohm] Resistance at 25°C
        elif type == "NTCG103JF103FT1A":
            self.B: float = 3380  # [K]
            self.R_25: float = 10000.0 # [Ohm] Resistance at 25°C
