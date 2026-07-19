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

import statistics

if TYPE_CHECKING:
    from Color_Scheme import Color_Scheme
    from Network_State import Network_State


def export_importants_csv(State: "Network_State", file_path: str | PathLike[str]) -> None:
    """Export the principal network histories to one CSV row per update time.

    The array-valued columns are JSON arrays, so each input, measured output,
    per-output loss, combined input/output update vector, and resistance vector
    occupies a single CSV cell. MSE is the mean squared value of the per-output
    loss. The initial update values and resistances are omitted; each row
    therefore contains the values resulting from that row's update.

    Parameters
    ----------
    State
        Network state containing the recorded time histories.
    file_path
        Destination CSV filename.
    """
    n_updates = int(State.t)
    histories = {
        "input": State.input_drawn_in_t,
        "measured_output": State.output_in_t,
        "input_update": State.input_update_in_t[1:],
        "output_update": State.output_update_in_t[1:],
        "loss": State.loss_in_t,
        "Rs": State.R_in_t[1:],
    }

    for name, history in histories.items():
        if len(history) < n_updates:
            raise ValueError(
                f"Cannot export {n_updates} updates: {name} contains only "
                f"{len(history)} entries."
            )

    # If multiple measurements precede each update, select the last one: it is
    # the measurement that drives the corresponding update.
    n_measurements = min(len(histories["input"]), len(histories["measured_output"]))
    measurement_indices = [
        ((t * n_measurements + n_updates - 1) // n_updates) - 1
        for t in range(1, n_updates + 1)
    ] if n_updates else []

    destination = Path(file_path)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "t", "input", "measured_output", "loss", "MSE", "update_values", "Rs"
        ])
        for t, measurement_idx in enumerate(measurement_indices, start=1):
            loss = np.asarray(histories["loss"][t - 1])
            # Loss functions store samples along the first axis. The current
            # sample is first and has the same shape as the measured output.
            if loss.ndim > 1:
                loss = loss[0]
            loss = loss.ravel()
            update_values = np.concatenate((
                np.asarray(histories["input_update"][t - 1]).ravel(),
                np.asarray(histories["output_update"][t - 1]).ravel(),
            ))
            writer.writerow([
                t,
                json.dumps(np.asarray(histories["input"][measurement_idx]).tolist()),
                json.dumps(np.asarray(histories["measured_output"][measurement_idx]).tolist()),
                json.dumps(loss.tolist()),
                float(np.mean(np.square(loss))),
                json.dumps(update_values.tolist()),
                json.dumps(np.asarray(histories["Rs"][t - 1]).tolist()),
            ])
