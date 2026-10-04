"""Les symboles du TP 4 d'electronique, nommes comme dans l'enonce.

`S` est la variable de Laplace, `W` la pulsation. Les sources et leurs
courants inconnus sont des symboles libres : ils peuvent etre negatifs.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

S: Final = sp.Symbol("s")
W: Final = sp.Symbol("omega", positive=True)
R1, R2, R3, R4, R5 = sp.symbols("R_1 R_2 R_3 R_4 R_5", positive=True)
R, C, RL = sp.symbols("R C R_L", positive=True)
RS, GM, REQ = sp.symbols("R_S g_m R_eq", positive=True)
COL: Final = sp.Symbol("C_{ol}", positive=True)
GM1, GM2, RO1, RO2 = sp.symbols("g_m1 g_m2 r_o1 r_o2", positive=True)
RO: Final = sp.Symbol("r_o", positive=True)
RPI1: Final = sp.Symbol(r"r_{\pi 1}", positive=True)
RPI2: Final = sp.Symbol(r"r_{\pi 2}", positive=True)
VS, IS, VX, VY = sp.symbols("V_S I_S V_x V_y")
I1, IX, IY = sp.symbols("I_1 I_x I_y")
