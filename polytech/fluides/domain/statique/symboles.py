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
ALTITUDE = sp.Symbol("z", real=True)
"""Cote comptee vers le haut. Le cours ecrit ``Delta p = - rho g Delta z``."""

PROFONDEUR = sp.Symbol("h", nonnegative=True)
"""Enfoncement sous la surface libre, compte vers le bas.

Le cours lui reserve la lettre ``h`` et ecrit ``Delta p = rho g h`` : le
signe change parce que le sens de parcours change. Confondre les deux
lettres, c'est s'exposer a une erreur de signe -- d'ou deux symboles.
"""

Z = ALTITUDE
"""Alias historique : ``Z`` designait l'altitude."""

AIRE = sp.Symbol("A", positive=True)
"""Aire de la surface libre d'un bassin prismatique."""

ALLONGEMENT = sp.Symbol("lambda", positive=True)
"""Rapport longueur / hauteur d'une vallee auto-similaire."""

COTE = sp.Symbol("eta", nonnegative=True)
"""Variable d'integration : cote courante du remplissage, de 0 a H."""
