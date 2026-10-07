"""Exercices resolus, 0.15 et 0.16 : deux transformees de Laplace lues sur un graphique.

Une exponentielle se lit en trois reperes : d'ou elle part, vers quoi elle
tend, et ou sa tangente au depart coupe l'asymptote -- sa constante de
temps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transformees import retard, transformee_de_laplace

if TYPE_CHECKING:
    from collections.abc import Sequence

DEPART_15: Final = sp.Integer(4)
ASYMPTOTE_15: Final = sp.Integer(-1)
CONSTANTE_15: Final = sp.Rational(4, 5)
"""Les trois reperes de la figure de 0.15 : 4, -1 et 0,8."""
BASCULE_16: Final = sp.Integer(8)
MONTEE_16: Final = 2 - sp.exp(-T)
"""0.16 avant 8 : de 1 vers 2, constante de temps 1."""
DESCENTE_16: Final = 2 * sp.exp(-T / 2)
"""0.16 apres 8, le temps compte depuis 8 : de 2 vers 0, constante de temps 2."""


def signal_15() -> sp.Expr:
    """`f(t) = asymptote + (depart - asymptote) e^(-t/tau)`."""
    return ASYMPTOTE_15 + (DEPART_15 - ASYMPTOTE_15) * sp.exp(-T / CONSTANTE_15)


def image_15() -> sp.Expr:
    """La transformee de Laplace du signal de 0.15."""
    return transformee_de_laplace(signal_15())


def limites(image: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """La valeur initiale et la valeur finale : `lim p F(p)` a l'infini, en zero."""
    return sp.limit(P * image, P, sp.oo), sp.limit(P * image, P, 0)


def instant_ou_la_tangente_coupe_l_asymptote() -> sp.Expr:
    """La tangente a l'origine rejoint l'asymptote apres une constante de temps."""
    signal = signal_15()
    pente = signal.diff(T).subs(T, 0)
    return sp.simplify((ASYMPTOTE_15 - signal.subs(T, 0)) / pente)


def image_16() -> sp.Expr:
    """La montee, puis, retardee de 8, la descente moins le palier qu'elle remplace."""
    palier = sp.limit(MONTEE_16, T, sp.oo)
    remplacement = transformee_de_laplace(DESCENTE_16 - palier)
    return transformee_de_laplace(MONTEE_16) + retard(remplacement, BASCULE_16)


def signal_16(instant: object) -> sp.Expr:
    """Le signal dont `image_16` est la transformee, en un instant."""
    point = sp.sympify(instant)
    valeur = MONTEE_16.subs(T, point)
    if point >= BASCULE_16:
        palier = sp.limit(MONTEE_16, T, sp.oo)
        valeur += (DESCENTE_16 - palier).subs(T, point - BASCULE_16)
    return sp.simplify(valeur)


def signaux_16(instants: Sequence[float]) -> list[float]:
    """Le meme signal, en plusieurs instants a la fois."""
    return [float(signal_16(sp.Rational(str(instant)))) for instant in instants]


def ecart_au_raccord() -> sp.Expr:
    """Ce qui manque a la montee pour valoir exactement son palier en `t = 8`."""
    return sp.limit(MONTEE_16, T, sp.oo) - MONTEE_16.subs(T, BASCULE_16)
