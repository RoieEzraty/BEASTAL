from __future__ import annotations
import numpy as np

from numpy.typing import NDArray

import matrix_functions
from config import ExperimentConfig


# ===================================================
# Class - network structure variables
# ===================================================


class Network_Structure:
    """
    Net_structure class save the structure of the network
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        net_types:
        FC                   - each input connected to each output
        FC_connected_outputs - FC and all outputs connected
        partialInter         - each input connected to an inter node, that inter node to an output node
        square               - N*N array of nodes, each node has 4 neighbors, some are inputs and some outputs
        """
        self.Nin: int = config.Strctr.Nin
        self.Nout: int = config.Strctr.Nout
        self.Ninter: int = config.Strctr.Ninter
        if config.Sprvsr.task_type == "Iris_classification" and (self.Nin, self.Nout) != (4, 3):
            print("Iris classification requires Nin=4 and Nout=3; correcting the structure dimensions")
            self.Nin, self.Nout = 4, 3

        nodes = matrix_functions.build_input_output_and_ground(
            self.Nin, self.Nout, in_nodes=config.Strctr.in_nodes, Ninter=self.Ninter,
            out_nodes=config.Strctr.out_nodes, add_ground=config.Strctr.add_ground,
            net_type=config.Strctr.net_type, seed=config.Strctr.rand_seed,
            net_height=config.Strctr.net_height, net_len=config.Strctr.net_length
        )
        self.input_nodes_arr: NDArray[np.int_] = nodes[0]
        self.extraInput_nodes_arr: NDArray[np.int_] = nodes[1]
        self.inter_nodes_arr: NDArray[np.int_] = nodes[2]
        self.output_nodes_arr: NDArray[np.int_] = nodes[3]
        self.extraOutput_nodes_arr: NDArray[np.int_] = nodes[4]
        self.ground_nodes_arr: NDArray[np.int_] = nodes[5]
        self.extraNin: int = len(self.extraInput_nodes_arr)
        self.extraNout: int = len(self.extraOutput_nodes_arr)

        # for square network
        self.net_type = config.Strctr.net_type
        self.net_height = config.Strctr.net_height
        self.net_len = config.Strctr.net_length

    def build_incidence(self, type: str = 'FC') -> None:
        """
        build_incidence builds the incidence matrix DM

        inputs:
        None

        outputs:
        EI         - array, node number on 1st side of all edges
        EJ         - array, node number on 2nd side of all edges
        EIEJ_plots - array, combined EI and EJ, each line is two nodes of edge, for visual ease
        DM         - array, connectivity matrix NE X NN
        NE         - int, # edges in network
        NN         - int, # nodes in network
        """
        if type == 'FC' or type == 'FC_connected_outputs':
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence(self)
        elif type == 'partialInter':
            print('partialInter is true')
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN =\
                matrix_functions.build_incidence_partialInter(self)
        elif type == 'square':
            print('building square network')
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence_square(self)
        elif type == 'beads':
            print('building network for beads')
            self.EI, self.EJ, self.EIEJ_plots, self.DM, self.NE, self.NN = matrix_functions.build_incidence_beads(self)

    def build_inverse_incidence(self) -> None:
        self.DM_dagger: NDArray[np.float_] = matrix_functions.inverse_incidence(self.DM)

    def build_RM(self) -> None:
        """Build the repetition/selection matrix for this structure's dimensions."""
        self.RM: NDArray[np.int_] = matrix_functions.build_rep_sel(self.Nin, self.Nout, self.DM)
