"""Mathematique, exercices 1, section 1.2 : les integrales 11 a 19."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.newton_exs_01 import (
    Definie,
    newton_leibniz,
    par_symetrie,
    racine,
)
from polytech.maths.application.resolution_exs_01 import A
from polytech.maths.domain.symboles import VARIABLE as X

CHANGEMENT = "changement de variable"
PARTIES = "integration par parties"


def integrale_11() -> Definie:
    """`2t / ((1 + t^2)(3 + t^2))` : `u = t^2`, puis des elements simples."""
    primitive = sp.log((1 + X**2) / (3 + X**2)) / 2
    integrande = 2 * X / ((1 + X**2) * (3 + X**2))
    return newton_leibniz(integrande, (0, 3), primitive, CHANGEMENT)


def integrale_12() -> Definie:
    """`t^2 arctan t / (1 + t^2)` : ecrire `t^2 = (1 + t^2) - 1`."""
    primitive = X * sp.atan(X) - sp.log(1 + X**2) / 2 - sp.atan(X) ** 2 / 2
    integrande = X**2 * sp.atan(X) / (1 + X**2)
    return newton_leibniz(integrande, (0, 1), primitive, "decomposition, puis parties")


def integrale_13() -> Definie:
    """`racine(e^t - 1)` : `u = racine(e^t - 1)`, et `2 u^2 / (1 + u^2)`."""
    u = racine(sp.exp(X) - 1)
    return newton_leibniz(u, (0, sp.log(2)), 2 * u - 2 * sp.atan(u), CHANGEMENT)


def integrale_14() -> Definie:
    """`1 / (racine(t) - 1)` : `u = racine(t)`, et `2u / (u - 1)`."""
    u = racine(X)
    primitive = 2 * u + 2 * sp.log(u - 1)
    return newton_leibniz(1 / (u - 1), (4, 9), primitive, CHANGEMENT)


def integrale_15() -> Definie:
    """`t e^t / racine(e^t + 1)` : parties, puis `w = racine(e^t + 1)`."""
    w = racine(sp.exp(X) + 1)
    primitive = 2 * X * w - 4 * w - 2 * sp.log((w - 1) / (w + 1))
    return newton_leibniz(X * sp.exp(X) / w, (0, 1), primitive, PARTIES)


def integrale_16() -> Definie:
    """`ln(1 - a^2 t^2) / t^2` : parties, `dv = dt / t^2` ; le crochet s'annule en 0."""
    logarithme = sp.log(1 - A**2 * X**2)
    primitive = -logarithme / X - A * sp.log((1 + A * X) / (1 - A * X))
    return newton_leibniz(logarithme / X**2, (0, 1), primitive, PARTIES)


def integrale_17() -> Definie:
    """`1 / (2 + racine(1 - t^2))` : `t = sin u`, puis `tan(u/2)`."""
    angle = sp.asin(X)
    primitive = angle - 4 / sp.sqrt(3) * sp.atan(sp.tan(angle / 2) / sp.sqrt(3))
    return newton_leibniz(1 / (2 + racine(1 - X**2)), (0, 1), primitive, CHANGEMENT)


def integrale_19() -> Definie:
    """`racine(1 + t^2)` : parties, ou `t = sinh u`."""
    primitive = (X * racine(1 + X**2) + sp.asinh(X)) / 2
    return newton_leibniz(racine(1 + X**2), (-1, 1), primitive, PARTIES)


def integrale_18() -> Definie:
    """`1 / (t + racine(t^2 + 1)) = racine(t^2 + 1) - t` : la part impaire s'annule."""
    integrande = 1 / (X + racine(X**2 + 1))
    return par_symetrie(integrande, integrale_19(), sp.Integer(1))
