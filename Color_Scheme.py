from __future__ import annotations
from matplotlib.colors import Colormap
from typing import Tuple

import colors

# ===================================================
# Class - Color scheme
# ===================================================


class Color_Scheme:
    """
    Class with colors for network simulation.
    """
    def __init__(self, show=False) -> None:
        self.colors_lst, self.red, self.cmap = self.color_scheme(show)

    def color_scheme(self, show: bool = False) -> Tuple[list[str], str, Colormap]:
        """
        define color scheme and return main colors, main red color and a colormap

        inputs:
        show - boolean of whether to plot colormap and red color

        outputs:
        colors      - list of strings names of colors in hexadecimal (#RRGGBB)
        red         - str, hexadecimal for soft red
        custom_cmap - matplotlibcolors.Colormap of 256 colors on a scale using "colors"
        """

        return colors.color_scheme(show)
