"""Training supervision, datasets, task targets, and learning-rate scheduling."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import Bunch, shuffle

import functions
from config import ExperimentConfig

if TYPE_CHECKING:
    from Network_State import Network_State
    from Network_Structure import Network_Structure
    from User_Variables import User_Variables


class Supervisor:
    """Own the task, training data, and learning-rate schedule."""

    def __init__(self, config: ExperimentConfig, Strctr: "Network_Structure",
                 Variabs: "User_Variables") -> None:
        sprvsr = config.Sprvsr
        self.iterations: int = sprvsr.iterations
        self.task_type: str = sprvsr.task_type
        self.dataset_type: str = sprvsr.dataset_type
        self.training_scheme: str = sprvsr.training_scheme
        self.use_p_tag: bool = sprvsr.use_p_tag
        self.stay_sample: int = sprvsr.stay_sample
        self.normalize_loss: bool = sprvsr.normalize_loss
        self.supress_prints: bool = sprvsr.supress_prints
        self.measure_accuracy_every: int = sprvsr.measure_accuracy_every
        self.anneal: bool = sprvsr.anneal
        self.T: float = sprvsr.T_annealing
        self.include_Power: bool = sprvsr.include_Power
        self.access_interNodes: bool = sprvsr.access_interNodes
        self.noise_to_extra: bool = sprvsr.noise_to_extra
        self.loss_type: str = sprvsr.loss_type
        self.print_every: int = sprvsr.print_every
        self.calculate_cosine_sim: bool = sprvsr.calculate_cosine_sim
        self.alpha_scale_nonlin: float = sprvsr.alpha_scale_nonlin
        self.loss_fn = functions.loss_fn_2samples if self.use_p_tag else functions.loss_fn_1sample
        self.lam: float = -80.0**-1
        self.dataset: NDArray[np.float_]
        self.targets: NDArray[np.float_]
        self.X_train: NDArray[np.float_]
        self.X_test: NDArray[np.float_]
        self.y_train: NDArray[np.float_]
        self.y_test: NDArray[np.float_]
        self.means: NDArray[np.float_]

        self.assign_alpha(sprvsr.alpha, Variabs)
        self.assign_M(config, Strctr)
        self.create_dataset_and_targets(config, Strctr, Variabs)
        self.create_noise_for_extras(Strctr, Variabs)

    def assign_alpha(self, alpha: float, Variabs: "User_Variables") -> None:
        """Assign the learning rate, including nonlinear-rule scaling."""
        nonlinear_rules = {"deltaR_propto_dp_nonlin", "deltaR_propto_dp_nonlin_decay"}
        scale = self.alpha_scale_nonlin if Variabs.R_update in nonlinear_rules else 1.0
        self.alpha_initial: float = float(alpha) * scale
        self.alpha: float = self.alpha_initial
        self.alpha_in_t: NDArray[np.float_] = self.alpha_initial * np.ones(self.iterations)

    def assign_M(self, config: ExperimentConfig, Strctr: "Network_Structure") -> None:
        """Build and optionally normalize the regression task matrix."""
        M_values = config.Sprvsr.M_values.copy()
        required_size = Strctr.Nin * Strctr.Nout
        if np.size(M_values) < required_size:
            M_values = functions.random_gen_M(config.Sprvsr.random_state_M, required_size)
        else:
            M_values = M_values[:required_size]
        if config.Sprvsr.normalize_M:
            M_values = functions.normalize_M(
                M_values, config.Sprvsr.normalize, Strctr.Nin, Strctr.Nout
            )
        if np.size(M_values) != required_size:
            raise ValueError(
                f"M has {np.size(M_values)} values; expected {required_size} "
                f"for Nin={Strctr.Nin}, Nout={Strctr.Nout}."
            )
        self.M: NDArray[np.float_] = M_values.reshape(Strctr.Nout, Strctr.Nin)

    def create_dataset_and_targets(self, config: ExperimentConfig, Strctr: "Network_Structure",
                                   Variabs: "User_Variables",
                                   train_size: Optional[float] = 0.8) -> None:
        """Create training/test data and their desired targets."""
        random_state = config.Sprvsr.random_state
        if self.task_type == "Regression":
            np.random.seed(random_state)
            if Variabs.R_update == "beads":
                self.dataset = np.ones((self.iterations, Strctr.Nin))
            elif self.dataset_type == "alternating ones":
                self.dataset = np.tile(
                    np.eye(Strctr.Nin), (int(self.iterations / Strctr.Nin), 1)
                )
            else:
                self.dataset = np.random.uniform(0.0, 2.0, size=(self.iterations, Strctr.Nin))
            self.targets = self.dataset @ self.M.T
            self.X_train = copy.copy(self.dataset)
            self.y_train = copy.copy(self.targets)
            self.X_test = copy.copy(self.dataset)
            self.y_test = copy.copy(self.targets)
            return

        if self.task_type != "Iris_classification":
            raise ValueError(f"Unknown task type: {self.task_type}")

        iris = cast(Bunch, load_iris())
        iris_data = np.asarray(iris.data, dtype=float)
        iris_target = np.asarray(iris.target, dtype=int)
        scaler = MinMaxScaler(feature_range=(0, 5))
        self.dataset = scaler.fit_transform(iris_data)
        encoder = OneHotEncoder(sparse_output=False, categories="auto")
        self.targets = encoder.fit_transform(iris_target.reshape(-1, 1))
        if train_size:
            split_data = train_test_split(
                self.dataset, self.targets, train_size=train_size, random_state=random_state,
                stratify=iris_target
            )
            self.X_train = np.asarray(split_data[0], dtype=float)
            self.X_test = np.asarray(split_data[1], dtype=float)
            self.y_train = np.asarray(split_data[2], dtype=float)
            self.y_test = np.asarray(split_data[3], dtype=float)
        else:
            self.X_train = np.asarray(shuffle(copy.copy(self.dataset), random_state=random_state), dtype=float)
            self.X_test = np.asarray(shuffle(copy.copy(self.dataset), random_state=random_state), dtype=float)
            self.y_train = np.asarray(shuffle(copy.copy(self.targets), random_state=random_state), dtype=float)
            self.y_test = np.asarray(shuffle(copy.copy(self.targets), random_state=random_state), dtype=float)
        y_train_decoded = np.argmax(self.y_train, axis=1)
        self.means = np.array([
            np.mean(self.X_train[y_train_decoded == class_index], axis=0)
            for class_index in range(3)
        ])

    def create_noise_for_extras(self, Strctr: "Network_Structure",
                                Variabs: "User_Variables") -> None:
        """Create boundary-condition noise arrays for non-task nodes."""
        dataset_size = self.X_train.shape[0]

        def uniform_noise(size: tuple[int, int]) -> NDArray[np.float_]:
            return (np.random.uniform(0.0, 1.0, size=size) - 0.5) * Variabs.bc_noise

        self.noise_in = uniform_noise((dataset_size, Strctr.extraNin))
        self.noise_inter = uniform_noise((dataset_size, Strctr.Ninter))
        self.noise_out = uniform_noise((dataset_size, Strctr.extraNout))

    def anneal_alpha(self, State: "Network_State") -> None:
        """Update the exponentially annealed learning rate at the current training time."""
        self.alpha = self.alpha_initial * np.exp(-State.t / (self.T * self.iterations))
        if State.t < self.iterations:
            self.alpha_in_t[State.t] = self.alpha
