from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# -----------------------------
# Relating to all
# -----------------------------

# MATERIAL = "numerical"
# MATERIAL = "Leon_plastic"
# MATERIAL = "Leon_metal"
MATERIAL = "Roie_metal"

# HINGES: int = 5  # Hinges
HINGES: int = 6  # Hinges
# HINGES: int = 8  # Hinges
# HINGES: int = 4  # Hinges

L: float = 0.0472  # [m] length of each edge

# -----------------------------
# Structure and initial params
# -----------------------------
@dataclass(frozen=True)
class StructureConfig:
    H: int = HINGES
    S: int = 1  # Shims per hinge
    # Nin: int = 3  # tip position in (x, y) and its angle
    # Nout: int = 3  # Fx, Fy, torque, all on tip
    # Nin: int = 3  # tip position in (x, y) and its angle at left side
    # Nout: int = 3  # x, y, theta of tip
    # Nin: int = 3  # x, y, theta of tip
    # Nout: int = 2  # Fx, Fy
    Nin: int = 2  # total and tip angles
    Nout: int = 2  # Fx Fy transformed into total and tip angle forces

    L: float = L  # [m] length of each edge


# -----------------------------
# Material / variables
# -----------------------------
@dataclass(frozen=True)
class VariablesConfig:
    material: str = MATERIAL  # "Leon_plastic" | "Leon_metal" | "numerical" | "Roie_metal"

    # chosen per material
    k_type: str = field(init=False)
    tau_file: str | None = field(init=False)
    thetas_ss: float = field(init=False)
    thresh: float = field(init=False)
    k_soft: str | None = field(init=False)
    k_stiff: str | None = field(init=False)

    def __post_init__(self):
        if self.material == "Leon_plastic":
            object.__setattr__(self, "k_type", "Leon_plastic_txt")
            object.__setattr__(self, "tau_file", "single_hinge_files/Roee_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.03312)  # not used in experimental
            object.__setattr__(self, "thresh", 1.96257)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Leon_metal":
            object.__setattr__(self, "k_type", "Leon_metal_txt")
            object.__setattr__(self, "tau_file", "single_hinge_files/Roee_metal_offset3mm_dl75.txt")
            object.__setattr__(self, "thetas_ss", 1.227)  # not used in experimental
            object.__setattr__(self, "thresh", 1.693)
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "Roie_metal":
            object.__setattr__(self, "k_type", "Roie_metal_csv")
            # object.__setattr__(self, "tau_file", "Roie_metal_singleMylar_short.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Stress_Strain_steel_1myl1tp_short.csv")
            # object.__setattr__(self, "thresh", 1.53)  # Feb23 realistically from just before red south
            # object.__setattr__(self, "tau_file", "single_hinge_files/Stress_Strain_1myl1tp_otherEnd_short.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Mar9_filled_average.csv")
            # object.__setattr__(self, "tau_file", "single_hinge_files/Mar12_dl90.csv")  # up to May22
            # object.__setattr__(self, "thresh", 1.24)  # Mar12 dl90
            # object.__setattr__(self, "tau_file", "single_hinge_files/May22_old_dl90_toughend.csv")
            # object.__setattr__(self, "thresh", 1.1)  # May22_old_dl90_toughend
            object.__setattr__(self, "tau_file", "single_hinge_files/May24_dl90_2ndEnd.csv")  # May24 2nd end (notated on chain itself)
            object.__setattr__(self, "thresh", 1.15)  # May24 2nd end (notated on chain itself)
            # object.__setattr__(self, "tau_file", "single_hinge_files/May24_dl90_1stEnd.csv")   # May24 1st end (notated on chain itself)
            # object.__setattr__(self, "thresh", 0.96)  # May24 1st end (notated on chain itself)
            # object.__setattr__(self, "thresh", 1.99)  # Feb23
            # object.__setattr__(self, "thresh", 1.58)
            # object.__setattr__(self, "thresh", 1.9)  # Feb22 measurements from just before Red South
            object.__setattr__(self, "thetas_ss", 0.91)  # not used in experimental
            object.__setattr__(self, "k_soft", None)
            object.__setattr__(self, "k_stiff", None)
        elif self.material == "numerical":
            object.__setattr__(self, "k_type", "Numerical")
            object.__setattr__(self, "tau_file", None)
            object.__setattr__(self, "thetas_ss", 1/2)
            object.__setattr__(self, "thresh", 1)
            object.__setattr__(self, "k_soft", 1.0)
            object.__setattr__(self, "k_stiff", 1.5)
        else:
            raise ValueError(f"Unknown material: {self.material}")

    # ADMET stress-strain tests from 2025Oct by Roie
    exp_start: float = 280*1e-3  # tip position start, not accounting for 2 first edges [m]
    exp_start = exp_start*0.99  # make sure to not stretch too much in simulation
    distance: float = 140*1e-3  # how much the arms compressed, [m]

    # numerical stability
    contact_scale: float = 100  # max experimental torque and torque upon edge contact ratio, for numerical stability
    # contact_scale: float = 1  # max experimental torque and torque upon edge contact ratio, for numerical stability


# -----------------------------
# Networkx Network
# -----------------------------
@dataclass(frozen=True)
class NetworkxNetConfig:
    

# -----------------------------
# Supervisor
# -----------------------------
@dataclass(frozen=True)
class SupervisorConfig:
    T: int = 240  # total training set time (not time to reach equilibrium during every step)

    alpha: float = 0.35

    rand_key_dataset: int = 7  # for random sampling of dataset, if dataset_sampling is True


# -----------------------------
# State
# -----------------------------
@dataclass(frozen=True)
class StateConfig:


# -----------------------------
# Top-level config
# -----------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    Variabs: VariablesConfig = VariablesConfig()
    Strctr: StructureConfig = StructureConfig()
    NET: NetworkxNetConfig = NetworkxNetConfig()
    Sprvsr: SupervisorConfig = SupervisorConfig()
    State: StateConfig = StateConfig()
    
CFG = ExperimentConfig()
