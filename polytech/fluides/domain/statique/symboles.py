"""Symboles litteraux de la statique (cours II).

Les formules du corrige sont d'abord etablies en litteral, puis evaluees :
c'est l'integrale qui fait foi, jamais une formule recopiee.
"""

from __future__ import annotations

import sympy as sp

RHO = sp.Symbol("rho", positive=True)
G = sp.Symbol("g", positive=True)
L = sp.Symbol("L", positive=True)
H = sp.Symbol("H", positive=True)
ALPHA = sp.Symbol("alpha", positive=True)
Z = sp.Symbol("z", nonnegative=True)
