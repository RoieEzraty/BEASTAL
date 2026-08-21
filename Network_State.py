from __future__ import annotations
import numpy as np
import copy

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import functions, solve, matrix_functions
import statistical_analysis as statistics
from config import ExperimentConfig

if TYPE_CHECKING:
    from User_Variables import User_Variables
    from Supervisor_Class import Supervisor
    from Big_Class import Big_Class
    from Network_Structure import Network_Structure


# ===================================================
# Class - network state variables
# ===================================================


class Network_State:
    """
    Class with variables that hold information of state of network.
    what ends with _in_t holds all time instances of the variable, each list index is different t
    what ends w/out _in_t is at current time instance self.t
    """
    def __init__(self, config: ExperimentConfig, BigClass: "Big_Class") -> None:
        super().__init__()
        self.initial_R: NDArray[np.float_] = np.asarray(config.State.R_vec_i, dtype=float).copy()
        self.t: int = 0  # number of calculated training updates, including updates buffered in a batch
        self.p: NDArray[np.float_] = array([])  # pressure
        self.u: NDArray[np.float_] = array([])  # flow rate
        # "measurement" modality
        self.inter_in_t: List[NDArray[np.float_]] = []  # pressure at intermediate nodes (not input/output) in time
        self.output_in_t: List[NDArray[np.float_]] = []  # pressure at outputs in time
        self.p_in_t: List[NDArray[np.float_]] = []  # measured pressures on all physical nodes in time
        self.extraOutput_in_t: List[NDArray[np.float_]] = []  # pressure at additional outputs, loss not calculated
        Strctr = BigClass.Strctr
        BigClass.Sprvsr.reset_training_history(Strctr)
        self._update_value_snapshots: List[Tuple[NDArray[np.float_], ...]] = []
        self._last_update_snapshot_t: Optional[int] = None
        self.hysteresis: NDArray[np.float_] = np.zeros(Strctr.NE, dtype=float)
        self.Power_norm: float = 0.0
        self.Power_norm_in_t: List[float] = []  # Power dissipation in whole network, normalized by inputs
        # Other sizes that make problems sometimes
        self.extraInput: NDArray[np.float_] = copy.copy(BigClass.Sprvsr.extraInput_update_in_t[-1])

    def initiate_resistances(self, BigClass: "Big_Class", R_vec_i: Optional[NDArray[np.float_]] = None,
                             add_noise: Optional[float] = 0.0) -> None:
        """
        After using build_incidence, initiate resistances

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        R_vec_i  - optional initial resistances, array of size [NE,]
        """
        self._update_value_snapshots.clear()
        self._last_update_snapshot_t = None
        if R_vec_i is not None:  # user speficied initial resistances
            if np.size(R_vec_i) != BigClass.Strctr.NE:
                print('R_vec_i has wrong size, initializing all ones')
                self.R_in_t: List[NDArray[np.float_]] = [np.ones((BigClass.Strctr.NE), dtype=float)]
            else:
                self.R_in_t = [R_vec_i]
        else:
            if np.size(self.initial_R) == BigClass.Strctr.NE:
                self.R_in_t = [self.initial_R.copy()]
            else:
                self.R_in_t = [np.ones(BigClass.Strctr.NE, dtype=float)]

        if add_noise:
            self.R_in_t[0] += np.random.normal(loc=0.0, scale=add_noise, size=BigClass.Strctr.NE)

        # self.R_in_t[0] = 10*self.R_in_t[0]
        # resistances for bead net as if w/out beads
        self.R_backg: NDArray[np.float_] = BigClass.Variabs.R_min * np.ones(BigClass.Strctr.NE)

    def initiate_accuracy_vec(self, BigClass: "Big_Class") -> None:
        """
        For classification task, initiate array for accuracy with length=iteration/measure_accuracy_every

        inputs:
        BigClass               - class instance including User_Variables, Network_Structure instances, etc.
        measure_accuracy_every - measure accuracy every # steps, user input
        """
        measure_accuracy_every = BigClass.Sprvsr.measure_accuracy_every
        accuracy_size = int(np.floor(BigClass.Sprvsr.iterations/measure_accuracy_every))
        self.accuracy_in_t: NDArray[np.float_] = zeros(accuracy_size)
        self.t_for_accuracy: NDArray[np.int_] = zeros(accuracy_size, dtype=np.int_)

    def draw_p_in_and_desired(self, Sprvsr: "Supervisor", i: int, noise_to_extra: Optional[bool] = False,
                              modality: Optional[str] = "measure") -> None:
        """
        Every time step, draw random input pressures and calculate the desired output given input

        inputs:
        Variabs        - User_Variables class
        i              - int, iteration #
        noise_to_extra - bool, whether to add noise to p on extra nodes
        modality       - str, "measure" for measurement modality where outputs are measured
                              "update" for update modality where outputs are constained
                              and after which resistances change

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        desired: np.ndarray sized [Nout,], desired output defined by the task M*p_input
        """
        # draw  input from train or test sets
        if modality == 'measure_for_accuracy':
            self.input_drawn: NDArray[np.float_] = copy.copy(Sprvsr.X_test[i % np.shape(Sprvsr.X_test)[0]])
        else:
            self.input_drawn = copy.copy(Sprvsr.X_train[i % np.shape(Sprvsr.X_train)[0]])

        # draw noise if needed
        if noise_to_extra:
            self.extraInput += Sprvsr.noise_in[i % np.shape(Sprvsr.noise_in)[0]]
            self.extraOutput: NDArray[np.float_] = copy.copy(self.extraOutput_in_t[-1])
            self.extraOutput += Sprvsr.noise_out[i % np.shape(Sprvsr.noise_out)[0]]
            self.inter: NDArray[np.float_] = copy.copy(self.inter) + \
                Sprvsr.noise_inter[i % np.shape(Sprvsr.noise_inter)[0]]

        # calculate desired output from train or test sets
        if Sprvsr.task_type == 'Iris_classification':
            if modality == 'measure_for_accuracy':
                self.desired: NDArray[np.float_] = \
                    np.matmul(Sprvsr.y_test[i % np.shape(Sprvsr.X_test)[0]], self.targets_mat)
            else:
                self.desired = \
                    np.matmul(Sprvsr.y_train[i % np.shape(Sprvsr.X_train)[0]], self.targets_mat)
        else:
            self.desired = Sprvsr.y_train[i % np.shape(Sprvsr.X_train)[0]]

        # append to arrays in time
        if modality == 'measure_for_accuracy':  # don't add to time vector if this is accuracy calculation
            pass
        else:
            Sprvsr.input_drawn_in_t.append(self.input_drawn)
            Sprvsr.extraInput_in_t.append(self.extraInput)
            Sprvsr.desired_in_t.append(self.desired)

        # optionally print to user
        if not Sprvsr.supress_prints:
            print('input_drawn', self.input_drawn)
            # print('extraInput', self.extraInput)
            print('desired output=', self.desired)

    def draw_p_means_Iris(self, Sprvsr: "Supervisor", i: int) -> None:
        """
        Draw input pressure as mean value of every class of Iris dataset.

        inputs:
        Variabs - User_Variables class
        i       - int, iteration # rangin {0-2}

        outputs
        input_drawn: np.ndarray sized [Nin,], input pressures
        """
        self.input_drawn = Sprvsr.means[i]

    def assign_targets_Iris(self, BigClass: "Big_Class") -> None:
        """
        Compute and assign class-specific output targets for the Iris classification task.

        For each of the 3 Iris classes:
        - Computes the mean of the input data belonging to that class.
        - Simulates the network flow for that class-specific input mean.
        - Stores the resulting output as the target for that class.

        Parameters
        ----------
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        """
        targets_mat: NDArray[np.float_] = zeros([3, 3], dtype=np.float_)
        for j in range(3):  # go over all 3 Iris classes
            self.draw_p_means_Iris(BigClass.Sprvsr, j)  # compute class-mean input for class j
            self.solve_flow_given_modality(BigClass, "measure_for_mean")  # simulate without changing resistances
            targets_mat[j] = self.output  # The new target is the outputs of the mean input
        self.targets_mat: NDArray[np.float_] = targets_mat  # save into targets_mat array

        # optionally print to user
        if not BigClass.Sprvsr.supress_prints:
            print('targets_mat', self.targets_mat)

    def solve_flow_given_modality(self, BigClass: "Big_Class", modality: str,
                                  noise_to_extra: Optional[bool] = False,
                                  access_inters: Optional[bool] = False) -> None:
        """
        Calculates the constraint matrix Cstr, then solves the flow,
        using functions from functions.py and solve.py,
        given the modality variable.

        inputs:
        BigClass - class instance including User_Variables, Network_Structure instances, etc.
        modality - string stating the modality type: "measure" for no constraint on outputs
                                                     "measure_for_mean" for outputs of mean of Iris class
                                                     "measure_for_accuracy" for outputs of mean of Iris class
                                                     "update" for constrained outputs as well
        noise_to_extra - optional bool, whether to add noise to p on extra nodes
        access_inters  - optional bool, whether to change pressure in inter nodes

        outputs:
        p - pressure at every node under the specific BC, after convergence while allowing conductivities to change
        u - flow at every edge under the specific BC, after convergence while allowing conductivities to change
        """
        # Calculate pressure p and flow u
        # Select nodes and pressure data based on modality
        nodes_tuple: functions.NodeArrays
        nodeData_tuple: functions.NodeDataArrays
        if modality in {'measure', 'measure_for_mean', 'measure_for_accuracy'}:
            if noise_to_extra:
                nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                               BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.inter_nodes_arr)
                nodeData_tuple = (self.input_drawn, self.extraInput, self.inter)
            else:
                nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                               BigClass.Strctr.ground_nodes_arr)
                nodeData_tuple = (self.input_drawn, self.extraInput)
        elif modality == 'update':
            # Access inter nodes if needed
            inters = BigClass.Sprvsr.access_interNodes or access_inters

            latest_update_values = (
                BigClass.Sprvsr.input_update_in_t[-1],
                BigClass.Sprvsr.extraInput_update_in_t[-1],
                BigClass.Sprvsr.output_update_in_t[-1],
                BigClass.Sprvsr.extraOutput_update_in_t[-1],
                BigClass.Sprvsr.inter_update_in_t[-1],
            )
            if self._last_update_snapshot_t != self.t:
                self._update_value_snapshots.append(tuple(
                    np.asarray(values, dtype=float).copy()
                    for values in latest_update_values
                ))
                if len(self._update_value_snapshots) > BigClass.Sprvsr.batch_size:
                    del self._update_value_snapshots[:-BigClass.Sprvsr.batch_size]
                self._last_update_snapshot_t = self.t

            if self.t % BigClass.Sprvsr.batch_size == 0:
                recent_snapshots = self._update_value_snapshots[-BigClass.Sprvsr.batch_size:]
                update_values = tuple(
                    np.mean(np.stack([snapshot[index] for snapshot in recent_snapshots]), axis=0)
                    for index in range(len(latest_update_values))
                )
            else:
                update_values = latest_update_values

            nodes_tuple = (BigClass.Strctr.input_nodes_arr, BigClass.Strctr.extraInput_nodes_arr,
                           BigClass.Strctr.ground_nodes_arr, BigClass.Strctr.output_nodes_arr,
                           BigClass.Strctr.extraOutput_nodes_arr)
            nodeData_tuple = update_values[:4]

            if inters:  # add inter nodes if needed
                nodes_tuple += (BigClass.Strctr.inter_nodes_arr,)
                nodeData_tuple += (update_values[4],)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Constraint matrix given constrained nodes and values
        self.CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]
        self.CstrTuple = functions.setup_constraints_given_pin(nodes_tuple, nodeData_tuple, BigClass.Strctr.NN,
                                                               BigClass.Strctr.EI, BigClass.Strctr.EJ)

        # R to K
        self.K_vec: NDArray[np.float_]  # type hint conductivities
        self.K_vec = matrix_functions.K_from_R(self.R_in_t[-1])  # calculate conductivities

        self.p, self.u = solve.solve_flow(BigClass.Strctr, self.CstrTuple, self.K_vec)

        # add to State class variables
        if modality in {'measure', 'measure_for_mean', 'measure_for_accuracy'}:
            self.inter = copy.copy(self.p[BigClass.Strctr.inter_nodes_arr].ravel())
            self.output: NDArray[np.float_] = copy.copy(self.p[BigClass.Strctr.output_nodes_arr].ravel())
            self.extraOutput = copy.copy(self.p[BigClass.Strctr.extraOutput_nodes_arr].ravel())

            # print
            if not BigClass.Sprvsr.supress_prints:
                # print('inter measured=', self.inter)
                print('output measured=', self.output)
                # print('extraOutput measured=', self.extraOutput)

            if modality == 'measure':  # Only save in time if measuring during training
                self.output_in_t.append(self.output)
                self.p_in_t.append(copy.copy(
                    self.p[:BigClass.Strctr.NN].ravel()
                ))
                self.extraOutput_in_t.append(self.extraOutput)
                self.inter_in_t.append(self.inter)






    def update_Rs(self, BigClass: "Big_Class", delta_K: Optional[NDArray[np.float_]] = None) -> None:
        """
        Calculate and record the next resistances of all edges.

        The material remains unchanged within a batch. At the batch boundary,
        ``solve_flow_given_modality`` has solved the update modality using the
        mean update-node values for the batch, and this method applies the
        configured material rule once to that averaged update state. Thus
        ``R_in_t`` still receives one entry per training update, including the
        unchanged entries within a batch.

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        R_vec  - [NE] array of resistivities
        """
        if self.t % BigClass.Sprvsr.batch_size:
            self.R_in_t.append(self.R_in_t[-1].copy())
            return

        R_vec: NDArray[np.float_] = self.R_in_t[-1]
        history_length = len(self.R_in_t)
        delta_p: NDArray[np.float_] = self.u * R_vec
        # delta_p: NDArray[np.float_] = np.matmul(BigClass.Strctr.DM, BigClass.State.p[:BigClass.Strctr.NN])  # same as u * R_vec

        # if hysteretic material - update only if new history. if not hysteretic, update for sure
        if BigClass.Variabs.hysteresis:
            BigClass.State.hysteresis += delta_p
            update_cond = ((BigClass.State.hysteresis > BigClass.Variabs.hyst_thresh) |
                           (BigClass.State.hysteresis < -BigClass.Variabs.hyst_thresh))
        else:
            update_cond = np.ones(BigClass.Strctr.NE, dtype=bool)

        if BigClass.Variabs.R_update in {'deltaR_propto_dp', 'deltaR_propto_dp_decay'}:  # delta_R propto p_in-p_out
            delta_R = BigClass.Variabs.gamma*delta_p * update_cond
            if BigClass.Variabs.normalize_step:
                delta_R_norm = BigClass.Sprvsr.alpha * delta_R / np.linalg.norm(delta_R)
                R_nxt: NDArray[np.float_] = self.R_in_t[-1] + delta_R_norm
            else:
                if BigClass.Variabs.R_update == 'deltaR_propto_dp_decay':  # update and add decay of resistance to 1
                    R_nxt = self.R_in_t[-1] + delta_R - BigClass.Variabs.decay*(self.R_in_t[-1] - 1)
                else:  # update regular
                    R_nxt = self.R_in_t[-1] + delta_R
            self.R_in_t.append(np.clip(R_nxt, 1e-12, None))
        elif BigClass.Variabs.R_update == 'R_propto_dp':  # R propto p_in-p_out
            self.R_in_t.append(BigClass.Variabs.gamma * np.abs(delta_p))
            # self.R_in_t.append(BigClass.Variabs.gamma * delta_p)
        elif BigClass.Variabs.R_update == "R_propto_sqrt_dp":
            self.R_in_t.append(BigClass.Variabs.gamma * np.sqrt(np.abs(delta_p)))
        elif BigClass.Variabs.R_update == 'deltaR_propto_Q':  # delta_R propto flow Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'R_propto_Q':  # R propto flow Q
            self.R_in_t.append(BigClass.Variabs.gamma * self.u)
        elif BigClass.Variabs.R_update == 'R_propto_Q_exp':  # R propto flow Q
            R_max = copy.copy(BigClass.Variabs.R_max)
            R_min = copy.copy(BigClass.Variabs.R_min)
            R_bar: float = (R_max + R_min)/2.0
            u_0: float = 1 / (np.sqrt(BigClass.Strctr.NE) * R_bar)
            # R_nxt: float = R_max + (R_min - R_max) * np.exp(- self.u / u_0)
            R_nxt = R_max + (R_min - R_max) * np.exp(- np.abs(self.u) / u_0)
            self.R_in_t.append(BigClass.Variabs.gamma * R_nxt)
        elif BigClass.Variabs.R_update in {'deltaR_propto_dp_nonlin', 'deltaR_propto_dp_nonlin_decay'}:  # non linear rules 
            delta_R = BigClass.Variabs.gamma*(delta_p)**3 * update_cond
            if BigClass.Variabs.normalize_step:
                delta_R_norm = BigClass.Sprvsr.alpha * delta_R / np.linalg.norm(delta_R)
                R_nxt = self.R_in_t[-1] + delta_R_norm
            else:
                if BigClass.Variabs.R_update == 'deltaR_propto_dp_nonlin_decay':  # update and add decay of resistance to 1
                    R_nxt = self.R_in_t[-1] + delta_R - BigClass.Variabs.decay*(self.R_in_t[-1] - 1)
                else:
                    R_nxt = self.R_in_t[-1] + delta_R
            self.R_in_t.append(np.clip(R_nxt, 1e-12, None))
        elif BigClass.Variabs.R_update == 'grad_desc':
            if delta_K is None:
                raise ValueError("delta_K must be supplied for gradient-descent resistance updates")
            K_vec = matrix_functions.K_from_R(self.R_in_t[-1])
            K_vec_nxt = K_vec + BigClass.Sprvsr.alpha * delta_K
            R_nxt = 1/K_vec_nxt
            if BigClass.Variabs.normalize_step:
                delta_R = R_nxt - self.R_in_t[-1]
                delta_R_norm = BigClass.Sprvsr.alpha * delta_R / np.linalg.norm(delta_R)
                R_nxt = self.R_in_t[-1] + delta_R_norm
            self.R_in_t.append(R_nxt)
        elif BigClass.Variabs.R_update == 'deltaR_propto_Power':  # delta_R propto Power dissipation dp*Q
            self.R_in_t.append(R_vec + BigClass.Variabs.gamma * self.u * delta_p * np.sign(delta_p))
        elif BigClass.Variabs.R_update == 'R_propto_Power':  # delta_R propto Power dissipation dp*Q
            self.R_in_t.append(BigClass.Variabs.gamma * self.u * delta_p * np.sign(delta_p))
        elif BigClass.Strctr.net_type == 'beads':
            self.R_in_t.append(matrix_functions.ChangeRFromFlow(BigClass, BigClass.Variabs.R_max,
                                                                BigClass.Variabs.R_min,
                                                                R_change_scheme='beads_pressure', allowed_cells=[],
                                                                beta=0.0))

        if BigClass.Variabs.hysteresis:
            BigClass.State.hysteresis[BigClass.State.hysteresis > BigClass.Variabs.hyst_thresh] = BigClass.Variabs.hyst_thresh
            BigClass.State.hysteresis[BigClass.State.hysteresis < -BigClass.Variabs.hyst_thresh] = -BigClass.Variabs.hyst_thresh

        # print
        if not BigClass.Sprvsr.supress_prints:
            pass

        if len(self.R_in_t) != history_length + 1:
            raise ValueError(f"Unknown resistance update rule: {BigClass.Variabs.R_update}")
        self.R_in_t[-1][self.R_in_t[-1] < 10**-12] = 10**-12

    def dK_grad_desc(self, Strctr: "Network_Structure", dK_step: float,
                     p_desired: NDArray[np.float_], func: str) -> NDArray[np.float_]:
        """
        Add desc

        inputs:
        BigClass  - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
                    and network state (State) class instances
                    I will not go into everything used from there to save space here.
        CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                                   https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                       Cstr -      Cstr_full without last line
                                       f    -      constraint vector (Rocks and Katifori 2018) 1D np.arrays [NEdges]
                                                   such that EI[i] is node connected to EJ[i] at certain edge
        K_vec     - 1D np.array [NE] of conductivities (inverse of resistances)
        p_desired - 1D np.array [Nout] of desired outputs given the inputs
        func      - str

        outputs:
        cost: np.float, MSE or mean abs between maesured and desired outputs
        """
        GD_dcost_vec = np.zeros([np.size(self.K_vec)])
        for m in range(np.size(self.K_vec)):
            dK_vec = np.zeros([np.size(self.K_vec)])
            dK_vec[m] = dK_step
            GD_dcost_vec[m] = self.calc_GD_cost(Strctr, p_desired, K_vec_for_GD=self.K_vec+dK_vec,
                                                mod='for_grad_desc', func=func)
        dcost_dK = (GD_dcost_vec - self.GD_cost) / dK_step
        delta_K = -dcost_dK
        return delta_K

    def calc_GD_cost(self, Strctr: "Network_Structure", p_desired: NDArray[np.float_],
                     K_vec_for_GD: Optional[NDArray[np.float_]] = None,
                     mod: str = 'measure', func: str = 'MSE') -> float:
        """
        MSE or mean abs cost between desired and measured network output
        given pressure input, conductivities and constraint matrix

        inputs:
        BigClass  - class instance including the user variables (Variabs), network structure (Strctr) and networkx (NET)
                    and network state (State) class instances
                    I will not go into everything used from there to save space here.
        CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                                   https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                       Cstr -      Cstr_full without last line
                                       f    -      constraint vector (Rocks and Katifori 2018) 1D np.arrays sized NEdges
                                                   such that EI[i] is node connected to EJ[i] at certain edge
        K_vec     - 1D np.array [NE] of conductivities (inverse of resistances)
        p_desired - 1D np.array [Nout] of desired outputs given the inputs

        outputs:
        cost: np.float, MSE  or mean abs between maesured and desired outputs
        """
        K_vec = self.K_vec if K_vec_for_GD is None else K_vec_for_GD
        p, u = solve.solve_flow(Strctr, self.CstrTuple, K_vec)
        p_out: NDArray[np.float_] = p[Strctr.output_nodes_arr][:, 0]  # p at output nodes, indexed as 1D array

        if p_out.size != p_desired.size:
            print(f"Incompatible sizes, p_out shape = {p_out.shape}, p_desired shape = {p_desired.shape}")
            cost = float("nan")
        elif func == 'MSE':
            cost = float(np.mean((p_out - p_desired) ** 2))
        elif func == 'mean_abs':
            cost = float(np.mean(np.abs(p_out - p_desired)))
        else:
            raise ValueError(f"Unknown gradient-descent cost function: {func}")
        if mod == 'measure':  # save in State class only if not part of dK_grad_desc
            self.GD_cost: float = cost
        return cost



    def calc_Power_norm(self, BigClass: "Big_Class") -> None:
        self.Power_norm = statistics.power_dissip_norm(self.u, self.R_in_t[-1], self.input_drawn)
        self.Power_norm_in_t.append(self.Power_norm)

        # print
        if not BigClass.Sprvsr.supress_prints:
            print('Power dissipation normalized', self.Power_norm)

    def calculate_accuracy_fullDataset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec: NDArray[np.int_] = zeros(np.shape(BigClass.Sprvsr.dataset)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Sprvsr.dataset):
            self.draw_p_in_and_desired(BigClass.Sprvsr, i, modality='measure_for_accuracy')
            self.solve_flow_given_modality(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Sprvsr.targets[i])
        self.accuracy = np.mean(self.accuracy_vec)

    def calculate_accuracy_testset(self, BigClass: "Big_Class") -> None:
        self.accuracy_vec = zeros(np.shape(BigClass.Sprvsr.X_test)[0], dtype=np.int_)
        for i, datapoint in enumerate(BigClass.Sprvsr.X_test):
            self.draw_p_in_and_desired(BigClass.Sprvsr, i, modality='measure_for_accuracy')
            self.solve_flow_given_modality(BigClass, "measure_for_accuracy")  # measure and don't change resistances
            self.accuracy_vec[i] = statistics.calculate_accuracy_1sample(self.output, self.targets_mat,
                                                                         BigClass.Sprvsr.y_test[i])
        self.accuracy = np.mean(self.accuracy_vec)

    def measure_flow(self, BigClass: "Big_Class") -> None:
        """
        Measure flow u (or Q) from all output nodes of network; summing on flow over all edges connected to output nodes

        inputs:
        BigClass: Class instance containing User_Variables, Network_Structure, etc.

        outputs:
        u_out: flow from all output nodes np.ndarray sized [Nout,]
        """
        self.u_out = np.sum(self.u[BigClass.Strctr.output_edges]*BigClass.Strctr.output_edge_directions)
