"""Mathematique, exercices 2, II.1 et II.2 : images et originaux.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 11
et 12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.laplace.definition import integrale_tronquee
from polytech.maths.domain.laplace.heaviside import (
    fractions_simples,
    par_fractions_simples,
)
from polytech.maths.domain.laplace.proprietes import (
    d_une_fonction_periodique,
    du_retard,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

if TYPE_CHECKING:
    from collections.abc import Callable

PULSATION: Final = sp.Symbol("omega", positive=True)
PHASE: Final = sp.Symbol("phi", real=True)

FONCTIONS: Final = {
    "a": 9 * T**7 - 3 * T**5 + T**2 - 3,
    "b": sp.exp(-2 * T) * (sp.cos(T) + 2 * T**2 - 1),
    "c": sp.sinh(2 * T),
    "d": sp.cos(PULSATION * T + PHASE),
    "e": sp.exp(-PULSATION * T) * sp.sin(PULSATION * T + PHASE),
}
"""II.1.1 : les cinq fonctions a transformer."""

IMAGES: Final = {
    1: 1 / (P**2 + 1),
    2: 1 / (P**2 + P + 1),
    3: 1 / (3 * P + 4),
    4: 2 * P / (3 * P**2 + 6 * P + 3),
    5: P / (P**2 + 2 * P - 3),
    6: (P**3 + P**2 - 2 * P - 2) / (P**2 - 1) ** 2,
    7: (3 * P + 4) / (P**2 + 9),
}
"""II.2 : les sept images a inverser."""


def developpee(fonction: sp.Expr) -> sp.Expr:
    """La fonction, sa phase developpee : `cos(wt + phi) = cos wt cos phi - ...`."""
    return sp.expand(sp.expand_trig(fonction)) if fonction.has(PHASE) else fonction


def images() -> dict[str, sp.Expr]:
    """II.1.1 : l'image de chacune des cinq fonctions."""
    return {
        lettre: Transformee.de(developpee(fonction)).image
        for lettre, fonction in FONCTIONS.items()
    }


def cube_tronque_et_cube_retarde() -> tuple[sp.Expr, sp.Expr]:
    """II.1.2 : l'image de `(t - 1)^3 H(t)`, puis celle de `tau_1(t^3 H)`."""
    tronquee = Transformee.de(sp.expand((T - 1) ** 3)).image
    return tronquee, du_retard(Transformee.de(T**3).image, 1)


def periodiques() -> tuple[sp.Expr, sp.Expr]:
    """II.1.3 : la dent de scie de periode 1, le creneau de periode 2."""
    scie = d_une_fonction_periodique(integrale_tronquee(T, sp.Integer(1)), 1)
    motif = integrale_tronquee(sp.Integer(1), sp.Integer(1))
    return sp.simplify(scie), sp.simplify(d_une_fonction_periodique(motif, 2))


def _original(numero: int) -> Callable[[], tuple[sp.Expr, sp.Expr]]:
    def inverser() -> tuple[sp.Expr, sp.Expr]:
        image = IMAGES[numero]
        return image, par_fractions_simples(image)

    return inverser


ORIGINAUX: Final = {numero: _original(numero) for numero in IMAGES}
"""II.2 : pour chaque numero, l'image et son original."""


def fractions_de_la_6() -> tuple[sp.Expr, ...]:
    """Les fractions simples de l'image 6, pole double compris."""
    return fractions_simples(IMAGES[6])
