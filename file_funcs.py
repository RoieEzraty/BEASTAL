from __future__ import annotations
import csv
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import copy
from os import PathLike
from pathlib import Path

from typing import Tuple, List, Dict, Any
from typing import TYPE_CHECKING
from numpy.typing import NDArray
from brokenaxes import brokenaxes
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

import statistical_analysis as statistics

if TYPE_CHECKING:
    from Big_Class import Big_Class
    from Color_Scheme import Color_Scheme


def export_importants_csv(BigClass: "Big_Class", file_path: str | PathLike[str]) -> None:
    """Export the principal network histories to one CSV row per update time.

    The array-valued columns are JSON arrays. Input, measured output, measured
    node values, and per-output loss have shape ``(Nvalues, batch_size)`` in
    each row, with one column per measurement in the batch. Combined
    input/output update values, the loss gradient averaged over the batch, and
    resistances are one-dimensional. MSE is the normalized scalar loss averaged
    over the batch. The initial update values and resistances are omitted; each
    row therefore contains the values resulting from that batch's averaged
    update.

    Parameters
    ----------
    BigClass
        Simulation classes containing the recorded time histories.
    file_path
        Destination CSV filename.
    """
    State = BigClass.State
    Sprvsr = BigClass.Sprvsr
    n_updates = int(State.t)
    batch_size = int(Sprvsr.batch_size)
    histories = {"input": Sprvsr.input_drawn_in_t, "measured_output": State.output_in_t, "node_values": State.p_in_t,
                  "input_update": Sprvsr.input_update_in_t[1:], "output_update": Sprvsr.output_update_in_t[1:],
                  "loss": Sprvsr.loss_in_t, "MSE": Sprvsr.loss_scalar_in_t_over_batch,
                  "grad_loss": Sprvsr.grad_loss_vec_in_t, "Rs": State.R_in_t[1:]}

    measurement_names = ("input", "measured_output", "node_values", "loss", "grad_loss")
    for name in measurement_names:
        history = histories[name]
        required_length = n_updates * batch_size
        if len(history) < required_length:
            raise ValueError(f"Cannot export {n_updates} batches of size {batch_size}: {name} contains only "
                             f"{len(history)} entries.")
    for name in ("input_update", "output_update", "MSE", "Rs"):
        history = histories[name]
        if len(history) < n_updates:
            raise ValueError(f"Cannot export {n_updates} updates: {name} contains only " f"{len(history)} entries.")

    destination = Path(file_path)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["t", "input", "measured_output", "node_values", "loss", "MSE", "update_values",
                         "grad_loss_over_batch", "Rs"])
        for t in range(1, n_updates + 1):
            batch_start = (t - 1) * batch_size
            batch_end = t * batch_size
            inputs = np.stack([np.asarray(value).ravel() for value in histories["input"][batch_start:batch_end]], axis=1)
            measured_outputs = np.stack([np.asarray(value).ravel() for value in histories["measured_output"][batch_start:batch_end]], axis=1)
            node_values = np.stack([np.asarray(value).ravel() for value in histories["node_values"][batch_start:batch_end]], axis=1)
            # Loss functions place the current sample first; it has one value
            # per measured output. Retain every current-sample loss in the batch.
            losses = np.stack([(value[0] if value.ndim > 1 else value).ravel()
                               for value in map(np.asarray, histories["loss"][batch_start:batch_end])], axis=1)
            grad_loss_over_batch = np.mean(np.asarray(histories["grad_loss"][batch_start:batch_end]), axis=0).ravel()
            update_values = np.concatenate((np.asarray(histories["input_update"][t - 1]).ravel(),
                                            np.asarray(histories["output_update"][t - 1]).ravel()))
            writer.writerow([t, json.dumps(inputs.tolist()), json.dumps(measured_outputs.tolist()),
                             json.dumps(node_values.tolist()), json.dumps(losses.tolist()), histories["MSE"][t - 1],
                             json.dumps(update_values.tolist()), json.dumps(grad_loss_over_batch.tolist()),
                             json.dumps(np.asarray(histories["Rs"][t - 1]).tolist())])
