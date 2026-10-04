"""Les symboles du TP 3 d'electronique.

`S` est la variable de Laplace, `T` le temps (positif : les reponses
valent pour `t > 0`), `W` la pulsation du cosinus d'entree.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

S: Final = sp.Symbol("s")
T: Final = sp.Symbol("t", positive=True)
W: Final = sp.Symbol("omega", nonnegative=True)
R, L, C, RL = sp.symbols("R L C R_L", positive=True)
