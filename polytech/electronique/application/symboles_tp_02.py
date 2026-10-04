"""Les symboles du TP 2 d'electronique, tels que l'enonce les nomme.

Le TP est litteral : ses resultats sont des expressions. `S` est la
variable de Laplace ; `OMEGA` la pulsation des exercices en phaseurs.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

S: Final = sp.Symbol("s")
OMEGA: Final = sp.Symbol("omega", positive=True)
R, L, C, R1, R2, R3, RS, RL, CL = sp.symbols(
    "R L C R_1 R_2 R_3 R_S R_L C_L", positive=True
)
Z = sp.Symbol("Z")
"""L'impedance quelconque du troisieme diviseur de l'exercice 4.1."""
VIN, V1, V2, E, IS, I1, I2 = sp.symbols("V_in V_1 V_2 E I_s I_1 I_2")
ITOT: Final = sp.Symbol("I_tot")
