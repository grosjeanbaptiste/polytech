"""Mathematique, exercices 1, section 1.2 : les integrales 1 a 10."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.newton_exs_01 import (
    DEMI_TOUR,
    Definie,
    impaire,
    newton_leibniz,
    par_symetrie,
)
from polytech.maths.application.resolution_exs_01 import A
from polytech.maths.domain.symboles import VARIABLE as X

SIN, COS = sp.sin(X), sp.cos(X)
CHANGEMENT = "changement de variable"
PARTIES = "integration par parties"


def integrale_1() -> Definie:
    """`cos^4` linearise : seul le terme `12 t / 32` survit aux bornes."""
    primitive = (sp.sin(4 * X) + 8 * sp.sin(2 * X) + 12 * X) / 32
    return newton_leibniz(COS**4, (0, DEMI_TOUR), primitive, "linearisation")


def integrale_2() -> Definie:
    """`sin^2 cos^3` : `u = sin t`, et il reste `u^2 (1 - u^2)`."""
    primitive = SIN**3 / 3 - SIN**5 / 5
    return newton_leibniz(
        SIN**2 * COS**3, (-DEMI_TOUR, DEMI_TOUR), primitive, CHANGEMENT
    )


def integrale_3() -> Definie:
    """`t^2 cos t` : deux parties, le polynome etant derive deux fois."""
    primitive = X**2 * SIN + 2 * X * COS - 2 * SIN
    return newton_leibniz(X**2 * COS, (0, DEMI_TOUR), primitive, PARTIES)


def integrale_4() -> Definie:
    """`t^2 sin t cos^2 t` est impaire : pair fois impair fois pair."""
    return impaire(X**2 * SIN * COS**2, DEMI_TOUR)


def integrale_5() -> Definie:
    """`sin t / (1 + cos^2 t)` : `u = cos t`, et `-du / (1 + u^2)`."""
    return newton_leibniz(SIN / (1 + COS**2), (0, DEMI_TOUR), -sp.atan(COS), CHANGEMENT)


def integrale_6() -> Definie:
    """`1 / (1 + sin t)` : `u = tan(t/2)`, et `2 du / (1 + u)^2`."""
    primitive = -2 / (1 + sp.tan(X / 2))
    return newton_leibniz(
        1 / (1 + SIN), (0, DEMI_TOUR), primitive, "tangente de l'arc moitie"
    )


def integrale_7() -> Definie:
    """`sin^2 / (sin + cos)` : echanger `t` et `pi/2 - t`, puis additionner.

    L'echange transforme l'integrale en celle de `cos^2 / (sin + cos)` ;
    leur somme est celle de `1 / (sin t + cos t)`, dont la primitive est
    `ln |tan(t/2 + pi/8)| / racine(2)`.
    """
    primitive = sp.log(sp.tan(X / 2 + sp.pi / 8)) / sp.sqrt(2)
    somme = newton_leibniz(
        1 / (SIN + COS), (0, DEMI_TOUR), primitive, "symetrie des bornes"
    )
    return par_symetrie(SIN**2 / (SIN + COS), somme, sp.Rational(1, 2))


def integrale_8() -> Definie:
    """`sin 2t / racine(1 - a sin t)` : `u = sin t`, et `2u / racine(1 - a u)`."""
    primitive = -4 * (A * SIN + 2) * sp.sqrt(1 - A * SIN) / (3 * A**2)
    integrande = sp.sin(2 * X) / sp.sqrt(1 - A * SIN)
    return newton_leibniz(integrande, (0, DEMI_TOUR), primitive, CHANGEMENT)


def integrale_9() -> Definie:
    """`t ln t` : parties ; en 0, `t^2 ln t` tend vers 0."""
    primitive = X**2 / 2 * sp.log(X) - X**2 / 4
    return newton_leibniz(X * sp.log(X), (0, 1), primitive, PARTIES)


def integrale_10() -> Definie:
    """`arcsin t` : la primitive de l'exercice 4 de la section 1.1."""
    primitive = X * sp.asin(X) + sp.sqrt(1 - X**2)
    return newton_leibniz(sp.asin(X), (0, 1), primitive, PARTIES)
