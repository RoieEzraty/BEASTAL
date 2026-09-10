"""Configuration for a resistor-network training experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# -----------------------------
# Relevant to all
# -----------------------------

R_UPDATE = "deltaR_NTC"
# R_UPDATE = "deltaR_propto_dp_nonlin"
# R_UPDATE = "deltaR_propto_dp"

# -----------------------------
# Networ Structure
# -----------------------------

@dataclass(frozen=True)
class StructureConfig:
    """Network topology and boundary-node configuration."""

    net_type: str = "FC"
    # net_type: str = "PC"
    net_height: int = 16
    net_length: int = 16
    Nin: int = 1
    Nout: int = 2
    Ninter: int = 0
    in_nodes: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=np.int_))
    out_nodes: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=np.int_))
    add_ground: bool = True
    frozen_ground: bool = True
    rand_seed: int = 35

# -----------------------------
# User Variables
# -----------------------------

@dataclass(frozen=True)
class VariablesConfig:
    """Physical update-rule parameters."""

    R_update: str = R_UPDATE
    gamma: NDArray[np.float_] = field(default_factory=lambda: np.array([1.0]))
    R_max: float = 2.0
    R_min: float = 0.02
    hysteresis: float = 0.0
    decay_R: float = 2e-3
    normalize_step: bool = False

    # NTC variables
    C_T: float = 35 * 1e-3
    G_T: float = 3.5 * 1e-3   # physical one
    # G_T: float = 3.5 * 1e-4   # small
    B: float = 3500  # [K]
    R_25: float = 1000.0  # [Ohm] Resistance at 25°C
    T_room: float = 298.15

# -----------------------------
# Networkx python instance
# -----------------------------

@dataclass(frozen=True)
class NetworkxNetConfig:
    """Network plot geometry."""

    scale: float = 50.0
    squish: float = 0.01

# -----------------------------
# External Supervisor parameters
# -----------------------------

@dataclass(frozen=True)
class SupervisorConfig:
    """Dataset and training-loop configuration."""

    task_type: str = "Regression"
    dataset_type: str = "alternating ones"
    # dataset_type: str = "random uniform"
    if R_UPDATE == "deltaR_NTC":
        training_scheme: str = "BEASTAL_NTC"
    else:
        training_scheme: str = "Adaline"
    # training_scheme: str = "Adjoint_pressure_noIn"
    # training_scheme: str = "Adjoint_current_noIn"
    # training_scheme: str = "Adjoint_pressure"
    batch_size: int = 1
    iterations: int = 100 * batch_size
    
    if R_UPDATE == "deltaR_NTC":
        alpha = 10  # deltaR_NTC
        beta = 0  # added inside the update rule for constant shift
        initial_T = 1.00 * VariablesConfig.T_room
    else:
        if training_scheme == "Adaline":
            alpha: float = 0.028  # deltaR_propto_deltap
        else:
            alpha: float = 0.08   # Adjoint
    
    alpha_scale_nonlin: float = 25.0 * batch_size**(1/2.7)
    # alpha_scale_nonlin: float = 62.0
    # alpha_scale_nonlin: float = 7.15 * batch_size**(1/3)
    use_p_tag: bool = False
    stay_sample: int = 1
    # normalize_loss: bool = R_UPDATE in {
    #     "deltaR_propto_dp_nonlin",
    #     "deltaR_propto_dp_nonlin_decay",
    # }
    normalize_loss = True
    # normalize_loss = False
    supress_prints: bool = False
    measure_accuracy_every: int = 15
    anneal: bool = False
    T_annealing: float = 0.75
    include_Power: bool = False
    access_interNodes: bool = False
    noise_to_extra: bool = False
    loss_type: str = "MSE"
    print_every: int = 1
    calculate_cosine_sim: bool = False

    # M_values: NDArray[np.float_] = field(
    #     default_factory=lambda: np.array([2 / 4, 1 / 4, 0.1, 0.35, 0.75, 0.04])
    # )
    M_values: NDArray[np.float_] | None = None
    normalize_M: bool = True
    normalize: float = 0.75
    random_state_M: int = 45
    random_state: int = 53

# -----------------------------
# Chain State parameters
# -----------------------------

@dataclass(frozen=True)
class StateConfig:
    """Initial network-state configuration."""

    R_vec_i: NDArray[np.float_] = field(default_factory=lambda: np.ones(6))
    # R_noise: float = 0.1
    R_noise: float = 0.0

# -----------------------------
# One config class to rule them all
# -----------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level grouping of all experiment configuration sections."""

    Variabs: VariablesConfig = field(default_factory=VariablesConfig)
    Strctr: StructureConfig = field(default_factory=StructureConfig)
    NET: NetworkxNetConfig = field(default_factory=NetworkxNetConfig)
    Sprvsr: SupervisorConfig = field(default_factory=SupervisorConfig)
    State: StateConfig = field(default_factory=StateConfig)


CFG = ExperimentConfig()
