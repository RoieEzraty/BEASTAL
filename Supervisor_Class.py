"""Training supervision, datasets, task targets, and learning-rate scheduling."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import Bunch, shuffle

import functions
import matrix_functions
import solve
from config import ExperimentConfig

if TYPE_CHECKING:
    from Big_Class import Big_Class
    from Network_State import Network_State
    from Network_Structure import Network_Structure
    from User_Variables import User_Variables


class Supervisor:
    """Own the task, training data, losses, and update-modality values."""

    def __init__(self, config: ExperimentConfig, Strctr: "Network_Structure",
                 Variabs: "User_Variables") -> None:
        sprvsr = config.Sprvsr
        self.iterations: int = sprvsr.iterations
        batch_size = sprvsr.batch_size
        if (isinstance(batch_size, bool)
                or not isinstance(batch_size, (int, np.integer))
                or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self.batch_size: int = int(batch_size)
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

    def reset_training_history(self, Strctr: "Network_Structure") -> None:
        """Reset the task-related histories and update-modality values."""
        self.input_drawn_in_t: List[NDArray[np.float_]] = []
        self.extraInput_in_t: List[NDArray[np.float_]] = []
        self.desired_in_t: List[NDArray[np.float_]] = []
        self.loss_in_t: List[NDArray[np.float_]] = []
        self.loss_scalar_in_t: NDArray[np.float_] = np.array([], dtype=float)
        self.input_update_in_t: List[NDArray[np.float_]] = [np.ones(Strctr.Nin)]
        self.extraInput_update_in_t: List[NDArray[np.float_]] = [np.ones(Strctr.extraNin)]
        self.inter_update_in_t: List[NDArray[np.float_]] = [np.random.random(Strctr.Ninter)]
        self.output_update_in_t: List[NDArray[np.float_]] = [0.5 * np.ones(Strctr.Nout)]
        self.extraOutput_update_in_t: List[NDArray[np.float_]] = [0.5 * np.ones(Strctr.extraNout)]
        self.adjoint_output_pressure: NDArray[np.float_] = np.zeros(Strctr.Nout, dtype=float)
        self.update_vec: NDArray[np.float_] = np.zeros(Strctr.NN, dtype=float)
        self.update_vec_in_t: NDArray[np.float_] = np.zeros(
            (self.iterations, Strctr.NN), dtype=float
        )

    def assign_alpha(self, alpha: float, Variabs: "User_Variables") -> None:
        """Assign the learning rate, including nonlinear-rule scaling."""
        nonlinear_rules = {"deltaR_propto_dp_nonlin", "deltaR_propto_dp_nonlin_decay"}
        scale = self.alpha_scale_nonlin if Variabs.R_update in nonlinear_rules else 1.0
        self.alpha_initial: float = float(alpha) * scale
        self.alpha: float = self.alpha_initial
        self.alpha_in_t: NDArray[np.float_] = self.alpha_initial * np.ones(self.iterations)

    def assign_M(self, config: ExperimentConfig, Strctr: "Network_Structure") -> None:
        """Build and optionally normalize the regression task matrix."""
        required_size = Strctr.Nin * Strctr.Nout
        configured_values = config.Sprvsr.M_values
        if configured_values is None:
            M_values = functions.random_gen_M(config.Sprvsr.random_state_M, required_size)
        else:
            M_values = configured_values.copy()
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

    def calc_loss(self, State: "Network_State") -> None:
        """Calculate and record the task loss for the current measurement."""
        if self.use_p_tag:
            if self.include_Power:
                self.loss = functions.loss_fn_2samples(
                    State.output, State.output_in_t[-2], State.desired, self.desired_in_t[-2],
                    State.Power_norm, State.Power_norm_in_t[-2], self.lam
                )
            else:
                self.loss = functions.loss_fn_2samples(
                    State.output, State.output_in_t[-2], State.desired, self.desired_in_t[-2]
                )
        elif self.include_Power:
            print('Power_norm', State.Power_norm)
            print('lam', self.lam)
            self.loss = functions.loss_fn_1sample(
                State.output, State.desired, State.Power_norm, self.lam
            )
        else:
            self.loss = functions.loss_fn_1sample(State.output, State.desired)
        self.loss_in_t.append(self.loss)

    def calc_loss_scalar(self, State: "Network_State",
                         Strctr: "Network_Structure", sample_count: int = 16) -> None:
        """Calculate loss normalized by the initial network's first 16 samples."""
        sample_count = min(sample_count, len(self.X_train))
        if sample_count == 0:
            raise ValueError("Cannot normalize loss with an empty training dataset")
        initial_K = matrix_functions.K_from_R(State.R_in_t[0])
        nodes = (Strctr.input_nodes_arr, Strctr.extraInput_nodes_arr,
                 Strctr.ground_nodes_arr)
        initial_outputs = []
        for input_values in self.X_train[:sample_count]:
            constraints = functions.setup_constraints_given_pin(
                nodes, (input_values, self.extraInput_update_in_t[0]),
                Strctr.NN, Strctr.EI, Strctr.EJ
            )
            pressures, _ = solve.solve_flow(Strctr, constraints, initial_K)
            initial_outputs.append(pressures[Strctr.output_nodes_arr].ravel())
        desired = self.y_train[:sample_count]
        if self.task_type == 'Iris_classification':
            desired = desired @ State.targets_mat
        elif self.task_type != 'Regression':
            raise ValueError(f"Unknown task type: {self.task_type}")
        initial_losses = desired - np.asarray(initial_outputs)

        # Previous normalization denominators:
        # denominator = self.y_train
        # denominator = np.mean(np.square(np.sum(1/(Strctr.Nin+1)-self.M, axis=0)))
        if self.loss_type == 'MAE':
            denominator = np.mean(np.abs(initial_losses))
            self.loss_scalar_in_t = np.mean(np.mean(np.abs(self.loss_in_t), axis=1), axis=1)
        elif self.loss_type == 'MSE':
            denominator = np.mean(np.square(initial_losses))
            self.loss_scalar_in_t = np.mean(np.mean(np.square(self.loss_in_t), axis=1), axis=1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        self.loss_scalar_in_t /= denominator

    def update_input(self, BigClass: "Big_Class") -> None:
        """Calculate and record the next input pressure in update modality."""
        R_update = BigClass.Variabs.R_update
        loss = self.loss_in_t[-1]
        input_update = self.input_update_in_t[-1]
        input_drawn = self.input_drawn_in_t[-1]
        if self.training_scheme in ['GD_like', 'Adaline']:
            delta = self.update_vec[BigClass.Strctr.input_nodes_arr]
        else:
            if self.use_p_tag:
                input_drawn_prev = self.input_drawn_in_t[-2]
            else:
                input_drawn_prev = np.zeros([BigClass.Strctr.Nin])
                loss = np.array([copy.copy(loss[0]), np.zeros([BigClass.Strctr.Nout])])
            if self.normalize_loss:
                delta = -(input_drawn-input_drawn_prev) * self.alpha * \
                    (np.mean(loss[0]-loss[1])/np.linalg.norm(loss[0]-loss[1]))
            else:
                delta = -(input_drawn-input_drawn_prev) * self.alpha * np.mean(loss[0]-loss[1])
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp']:
            self.input_update_nxt = input_update + delta
        elif R_update == 'beads':
            self.input_update_nxt = input_update + self.alpha * np.mean(np.abs(loss[0]))
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_NTC',
                          'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.input_update_nxt = delta
        elif R_update == 'grad_desc':
            self.input_update_nxt = input_update
        if functions.reset_update(self.input_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.input_update_nxt = self.input_update_in_t[0]
        self.input_update_in_t.append(self.input_update_nxt)
        if not self.supress_prints:
            print('input_update_nxt=', self.input_update_nxt)

    def update_extraInput(self, BigClass: "Big_Class") -> None:
        """Calculate and record the next extra-input pressure in update modality."""
        R_update = BigClass.Variabs.R_update
        loss = self.loss_in_t[-1]
        extraInput_update = self.extraInput_update_in_t[-1]
        extraInput = self.extraInput_in_t[-1]
        if self.use_p_tag:
            extraInput_prev = self.extraInput_in_t[-2]
            delta = (extraInput-extraInput_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:
            delta = extraInput * self.alpha * np.mean(loss[0])
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            self.extraInput_update_nxt = extraInput_update - delta
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_NTC',
                          'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.extraInput_update_nxt = -delta
        elif R_update == 'grad_desc':
            self.extraInput_update_nxt = extraInput_update
        if functions.reset_update(self.extraInput_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.extraInput_update_nxt = self.extraInput_update_in_t[0]
        self.extraInput_update_in_t.append(self.extraInput_update_nxt)
        if not self.supress_prints:
            print('extraInput_update_nxt=', self.extraInput_update_nxt)

    def update_inter(self, BigClass: "Big_Class") -> None:
        """Calculate and record the next intermediate-node update pressure."""
        State = BigClass.State
        R_update = BigClass.Variabs.R_update
        loss = self.loss_in_t[-1]
        inter_update = self.inter_update_in_t[-1]
        inter = State.inter_in_t[-1]
        if self.use_p_tag:
            inter_prev = State.inter_in_t[-2]
            delta = (inter-inter_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:
            delta = inter * self.alpha * np.mean(loss[0])
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            self.inter_update_nxt = inter_update - delta
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_NTC',
                          'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.inter_update_nxt = -delta
        elif R_update == 'grad_desc':
            self.inter_update_nxt = inter_update
        if functions.reset_update(self.inter_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.inter_update_nxt = self.inter_update_in_t[0]
        self.inter_update_in_t.append(self.inter_update_nxt)
        if not self.supress_prints:
            print('inter_update_nxt=', self.inter_update_nxt)

    def update_output(self, BigClass: "Big_Class") -> None:
        """Calculate and record the next output pressure in update modality."""
        State = BigClass.State
        R_update = BigClass.Variabs.R_update
        loss = self.loss_in_t[-1]
        output_update = copy.copy(self.output_update_in_t[-1])
        if self.training_scheme in [
            'GD_like', 'Adaline', 'Adjoint_current_noIn', 'Adjoint_pressure_noIn'
        ]:
            delta = self.update_vec[BigClass.Strctr.output_nodes_arr]
        else:
            if self.use_p_tag:
                output_prev = State.output_in_t[-2]
                loss_multip = ((loss[0]-loss[1])/np.linalg.norm(loss[0]-loss[1])
                               if self.normalize_loss else loss[0]-loss[1])
                delta = self.alpha * (State.output-output_prev) * loss_multip
            else:
                loss_multip = loss[0]/np.linalg.norm(loss[0]) if self.normalize_loss else loss[0]
                delta = self.alpha * State.output * loss_multip
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp']:
            self.output_update_nxt = output_update + delta
        elif R_update == 'beads':
            self.output_update_nxt = output_update + self.alpha * np.mean(loss[0])
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_NTC',
                          'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.output_update_nxt = delta
        elif R_update == 'grad_desc':
            self.output_update_nxt = output_update
        if functions.reset_update(self.output_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.output_update_nxt = self.output_update_in_t[0]
        self.output_update_in_t.append(self.output_update_nxt)
        if not self.supress_prints:
            print('output_update_nxt', self.output_update_nxt)

    def update_extraOutput(self, BigClass: "Big_Class") -> None:
        """Calculate and record the next extra-output pressure in update modality."""
        State = BigClass.State
        R_update = BigClass.Variabs.R_update
        loss = self.loss_in_t[-1]
        extraOutput_update = copy.copy(self.extraOutput_update_in_t[-1])
        if self.use_p_tag:
            extraOutput_prev = State.extraOutput_in_t[-2]
            delta = (State.extraOutput-extraOutput_prev) * self.alpha * np.mean(loss[0]-loss[1])
        else:
            delta = State.extraOutput * self.alpha * np.mean(loss[0])
        if R_update in ['R_propto_dp', 'R_propto_Q', 'R_propto_sqrt_dp', 'R_propto_Power', 'R_propto_Q_exp', 'beads']:
            self.extraOutput_update_nxt = extraOutput_update + delta
        elif R_update in ['deltaR_propto_dp', 'deltaR_propto_Q', 'deltaR_propto_Power', 'deltaR_NTC',
                          'deltaR_propto_dp_nonlin',
                          'deltaR_propto_dp_decay', 'deltaR_propto_dp_nonlin_decay']:
            self.extraOutput_update_nxt = delta
        elif R_update == 'grad_desc':
            self.extraOutput_update_nxt = extraOutput_update
        if functions.reset_update(self.extraOutput_update_nxt, BigClass.Variabs.reset_thresh_b,
                                  BigClass.Variabs.reset_thresh_s):
            self.extraOutput_update_nxt = self.extraOutput_update_in_t[0]
        self.extraOutput_update_in_t.append(self.extraOutput_update_nxt)
        if not self.supress_prints:
            print('extraOutput_update_nxt', self.extraOutput_update_nxt)

    def calc_update_vals_vec(self, BigClass: "Big_Class") -> None:
        """Calculate update-modality node values or sources for gradient-based training."""
        State = BigClass.State
        in_nodes = copy.copy(BigClass.Strctr.input_nodes_arr)
        out_nodes = copy.copy(BigClass.Strctr.output_nodes_arr)
        ground_nodes = copy.copy(BigClass.Strctr.ground_nodes_arr)
        if self.training_scheme == 'GD_like':
            L_vec = np.zeros(BigClass.Strctr.NN)
            L_vec[out_nodes] = self.loss
            delta_p = np.matmul(BigClass.Strctr.DM, State.p[:BigClass.Strctr.NN]).T
            delta_p[delta_p == 0] = 10**(-9)
            one_over_delta_p_norm = 1 / delta_p / np.linalg.norm(1 / delta_p)
            C_vec = np.matmul(BigClass.Strctr.RM, L_vec) * one_over_delta_p_norm
            if BigClass.Variabs.R_update in ['deltaR_propto_dp_nonlin', 'deltaR_propto_dp_nonlin_decay']:
                C_vec_norm = C_vec[0] / np.linalg.norm(C_vec[0])
                update_vec = -self.alpha * np.matmul(BigClass.Strctr.DM_dagger, C_vec_norm)
            else:
                update_vec = -self.alpha * np.matmul(BigClass.Strctr.DM_dagger, C_vec[0])
        elif self.training_scheme == 'Adaline':
            Strctr = BigClass.Strctr_fict if BigClass.Strctr.Ninter > 0 else BigClass.Strctr
            p = np.concatenate([State.p[in_nodes], State.p[out_nodes], State.p[ground_nodes]])
            grad_loss_vec = matrix_functions.grad_loss_FC(Strctr.NE, p, Strctr.DM, Strctr.output_nodes_arr, Strctr.ground_nodes_arr, self.loss)
            self.grad_loss_vec = grad_loss_vec
            grad_loss_vec_norm = grad_loss_vec / np.linalg.norm(grad_loss_vec)
            self.grad_loss_vec_norm = grad_loss_vec_norm
            update_vec = - self.alpha * np.matmul(Strctr.DM_dagger, grad_loss_vec_norm if self.normalize_loss else grad_loss_vec)
            if BigClass.Strctr.Ninter > 0:
                for idx in BigClass.Strctr.inter_nodes_arr:
                    update_vec = np.insert(update_vec, idx, 0)
        elif self.training_scheme == 'Adjoint_pressure':
            Strctr = BigClass.Strctr
            if not State.p_in_t or State.p_adjoint.size < Strctr.NN:
                raise ValueError(
                    "Adjoint_pressure requires measurement and adjoint solves before calc_update_vals_vec"
                )
            forward_drop: NDArray[np.float_] = np.matmul(Strctr.DM, State.p_in_t[-1])
            adjoint_drop: NDArray[np.float_] = np.matmul(
                Strctr.DM, State.p_adjoint[:Strctr.NN]
            ).ravel()
            self.adjoint_edge_update_vec: NDArray[np.float_] = forward_drop * adjoint_drop
            update_vec = np.matmul(Strctr.DM_dagger, self.adjoint_edge_update_vec)
        elif self.training_scheme in {'Adjoint_current_noIn', 'Adjoint_pressure_noIn'}:
            # For L = 1/2 ||y-y_des||^2, -dL/dy = y_des-y, which is
            # exactly the sign convention used by self.loss. Alpha scales the
            # imposed current or pressure and therefore the update magnitude.
            L_vec = np.zeros(BigClass.Strctr.NN)
            L_vec[out_nodes] = self.loss.ravel()
            update_vec = self.alpha * L_vec
        else:
            raise ValueError(f"Unknown training scheme: {self.training_scheme}")
        self.update_vec = update_vec
        update_index = State.t - 1
        if not 0 <= update_index < self.iterations:
            raise IndexError(
                f"Cannot store update vector for t={State.t}; "
                f"expected 1 <= t <= {self.iterations}."
            )
        self.update_vec_in_t[update_index] = update_vec

    def calc_adjoint_output_pressure(self, Strctr: "Network_Structure") -> None:
        """Set the output-only pressure imposed during the adjoint intermediate state."""
        loss = np.asarray(self.loss, dtype=float)
        output_residual = loss[0] if loss.ndim > 1 else loss
        output_residual = output_residual.reshape(-1)
        if output_residual.size != Strctr.Nout:
            raise ValueError(
                f"Adjoint output residual has {output_residual.size} entries; expected {Strctr.Nout}"
            )
        self.adjoint_output_pressure = self.alpha * output_residual
