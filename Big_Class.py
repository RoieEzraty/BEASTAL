from __future__ import annotations
from typing import TYPE_CHECKING

import colors

if TYPE_CHECKING:
    from Supervisor_Class import Supervisor
    from User_Variables import User_Variables
    from Network_State import Network_State
    from Network_Structure import Network_Structure
    from Color_Scheme import Color_Scheme
    from Networkx_Net import Networkx_Net

colors_lst, red, cmap = colors.color_scheme()

# ===================================================
# Class - Big class that contains all smaller classes
# ===================================================


class Big_Class:
    """
    Big_Class contains the main classes under Network Simulation
    """

    def __init__(self, Strctr: "Network_Structure") -> None:
        self.Strctr = Strctr
        self.Strctr_fict: "Network_Structure"
        self.Variabs: "User_Variables"
        self.Sprvsr: "Supervisor"
        self.State: "Network_State"
        self.NET: "Networkx_Net"
        self.Colorscheme: "Color_Scheme"

    def add_Variabs(self, Variabs: "User_Variables") -> None:
        self.Variabs = Variabs

    def add_Sprvsr(self, Sprvsr: "Supervisor") -> None:
        self.Sprvsr = Sprvsr

    def add_Strctr(self, Strctr: "Network_Structure") -> None:
        self.Strctr = Strctr

    def add_Strctr_fict(self, Strctr_fict: "Network_Structure") -> None:
        self.Strctr_fict = Strctr_fict

    def add_State(self, State: "Network_State") -> None:
        self.State = State

    def add_NET(self, NET: "Networkx_Net") -> None:
        self.NET = NET

    def add_Colors(self, Colorscheme: "Color_Scheme") -> None:
        self.Colorscheme = Colorscheme
