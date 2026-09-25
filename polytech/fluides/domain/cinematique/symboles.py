"""Symboles du cours III : coordonnees d'espace et temps.

Un seul jeu de symboles pour toute la bibliotheque, afin que deux expressions
construites separement restent comparables.
"""

from __future__ import annotations

import sympy as sp

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
T = sp.Symbol("t", real=True, nonnegative=True)

COORDONNEES = (X, Y)
