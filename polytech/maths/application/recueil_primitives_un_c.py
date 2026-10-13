"""Recueil de primitives, exercice 1 : les primitives 41 a 56.

Deux d'entre elles, la 45 et la 49, dependent d'un entier `n` : une
integration par parties les ramene a l'ordre precedent, et c'est cette
relation de recurrence que l'on rend, puis que l'on deroule.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICE_UN as F,
)
from polytech.maths.application.recueil_primitives_enonces import N, X
from polytech.maths.application.recueil_primitives_gestes import (
    U,
    changement,
    immediate,
    terme_a_terme,
)
from polytech.maths.application.recueil_primitives_parties import (
    en_cascade,
    par_parties,
)
from polytech.maths.application.resolution_exs_01 import Resolution, par_equation

sin, cos, exp, sqrt, ln = sp.sin, sp.cos, sp.exp, sp.sqrt, sp.log
sinh, cosh = sp.sinh, sp.cosh
UN = sp.Integer(1)


def relation_45() -> tuple[sp.Expr, sp.Expr]:
    """`I_n = x ln^n x - n I_(n-1)` : le terme integre, puis le facteur de `I_(n-1)`."""
    return X * ln(X) ** N, -N


def relation_49() -> tuple[sp.Expr, sp.Expr]:
    """`I_n = -x^n e^-x + n I_(n-1)` : le terme integre, puis le facteur."""
    return -(X**N) * exp(-X), N


def _derouler(
    relation: tuple[sp.Expr, sp.Expr], depart: sp.Expr, ordre: int
) -> sp.Expr:
    crochet, facteur = relation
    primitive = depart
    for k in range(1, ordre + 1):
        primitive = crochet.subs(N, k) + facteur.subs(N, k) * primitive
    return sp.expand(primitive)


def puissance_du_logarithme(ordre: int) -> sp.Expr:
    """La primitive de `ln^n x`, la recurrence deroulee depuis `I_0 = x`."""
    return _derouler(relation_45(), X, ordre)


def puissance_fois_exponentielle(ordre: int) -> sp.Expr:
    """La primitive de `x^n e^-x`, deroulee depuis `I_0 = -e^-x`."""
    return _derouler(relation_49(), -exp(-X), ordre)


def resolutions() -> dict[int, Resolution]:
    """Les primitives 41 a 56, hors les deux recurrences."""
    produit = (cosh(4 * X) - cosh(2 * X)) / 2
    return {
        41: terme_a_terme(F[41], produit, "produit en somme"),
        42: parties((X + 1) ** 3, cosh(X), gestes=3),
        43: changement(F[43], 1 / (2 * (1 + U**2)), X / 2),
        44: changement(F[44], 1 / (sqrt(3) * sqrt(1 - U**2)), sqrt(3) * X),
        46: en_cascade((ln(X) ** 3, ln(X) ** 2, ln(X)), X**4),
        47: changement(F[47], 1 / (5 * cos(U) ** 2), 5 * X),
        48: immediate(F[48], sp.asinh(X)),
        50: par_equation((cos(X), sinh(3 * X)), (-sin(X) / 3, cosh(3 * X))),
        51: changement(F[51], U * exp(-U) / 2, X**2),
        52: par_equation((cos(ln(X)), UN), (-sin(ln(X)), UN)),
        53: changement(F[53], exp(U) / 3, sin(X**3)),
        54: par_parties(ln(X + sqrt(1 + X**2)), UN),
        55: par_equation((sin(3 * X), 2**X), (3 * cos(3 * X) / ln(2), 2**X)),
        56: en_cascade((ln(X) ** 2, ln(X)), UN),
    }
