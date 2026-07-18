"""Configuration for a resistor-network training experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StructureConfig:
    """Network topology and boundary-node configuration."""

    net_type: str = "FC"
    net_height: int = 16
    net_length: int = 16
    Nin: int = 9
    Nout: int = 1
    Ninter: int = 0
    in_nodes: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=np.int_))
    out_nodes: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=np.int_))
    add_ground: bool = True
    rand_seed: int = 35


@dataclass(frozen=True)
class VariablesConfig:
    """Physical update-rule parameters."""

    R_update: str = "deltaR_propto_dp"
    # R_update: str = "deltaR_propto_dp_nonlin"
    gamma: NDArray[np.float_] = field(default_factory=lambda: np.array([1.0]))
    R_max: float = 2.0
    R_min: float = 0.02
    hysteresis: float = 0.0
    decay_R: float = 2e-3
    normalize_step: bool = False


@dataclass(frozen=True)
class NetworkxNetConfig:
    """Network plot geometry."""

    scale: float = 50.0
    squish: float = 0.01


@dataclass(frozen=True)
class SupervisorConfig:
    """Dataset and training-loop configuration."""

    task_type: str = "Regression"
    dataset_type: str = "uniform_random"
    training_scheme: str = "Adaline"
    iterations: int = 2000
    alpha: float = 0.028
    alpha_nonlin: float = alpha * 10
    use_p_tag: bool = False
    stay_sample: int = 1
    normalize_loss: bool = True
    supress_prints: bool = True
    measure_accuracy_every: int = 15
    anneal: bool = True
    T_annealing: float = 1.0
    include_Power: bool = False
    access_interNodes: bool = False
    noise_to_extra: bool = False
    loss_type: str = "MSE"
    print_every: int = 1
    calculate_cosine_sim: bool = False

    M_values: NDArray[np.float_] = field(
        default_factory=lambda: np.array([2 / 4, 1 / 4, 0.1, 0.35, 0.75, 0.04])
    )
    normalize_M: bool = True
    normalize: float = 0.75
    random_state_M: int = 35
    random_state: int = 52


@dataclass(frozen=True)
class StateConfig:
    """Initial network-state configuration."""

    R_vec_i: NDArray[np.float_] = field(default_factory=lambda: np.ones(6))


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level grouping of all experiment configuration sections."""

    Variabs: VariablesConfig = field(default_factory=VariablesConfig)
    Strctr: StructureConfig = field(default_factory=StructureConfig)
    NET: NetworkxNetConfig = field(default_factory=NetworkxNetConfig)
    Sprvsr: SupervisorConfig = field(default_factory=SupervisorConfig)
    State: StateConfig = field(default_factory=StateConfig)


CFG = ExperimentConfig()
