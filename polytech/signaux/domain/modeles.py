"""Deux transmittances que le TP 1 fait etablir : un circuit, une equation.

**Le circuit.** `L` et `C` en serie, puis `R` vers la masse, la sortie
prise sur `R` : un diviseur d'impedances, `H = R / (R + L p + 1/(C p))`.
Ses poles sont reels et distincts quand le discriminant de
`L C p^2 + R C p + 1` est positif, soit `R > 2 racine(L / C)`.

**L'equation.** A etat initial nul -- l'etat « quiescent » de l'enonce --
le theoreme de la derivation remplace chaque `d^k/dt^k` par `p^k`, et
l'equation devient un rapport de deux polynomes.

`coefficients` rend les deux listes que `scipy.signal` attend : c'est le
pont entre la forme symbolique du cours et les fonctions du TP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from collections.abc import Sequence


def transmittance_lc_serie_sur_r(
    resistance: sp.Expr, inductance: sp.Expr, capacite: sp.Expr
) -> Transmittance:
    """`R / (R + L p + 1/(C p))`, la sortie prise sur la resistance."""
    serie = resistance + inductance * IMAGE + 1 / (capacite * IMAGE)
    return Transmittance(expression=sp.cancel(resistance / serie))


def resistance_critique(inductance: sp.Expr, capacite: sp.Expr) -> sp.Expr:
    """`2 racine(L / C)` : au-dela, deux poles reels distincts."""
    return 2 * sp.sqrt(inductance / capacite)


def _polynome(coefficients_decroissants: Sequence[sp.Expr]) -> sp.Expr:
    """`a_n p^n + ... + a_0`, a partir des coefficients par degre decroissant."""
    degre = len(coefficients_decroissants) - 1
    return sum(
        (sp.sympify(c) * IMAGE ** (degre - rang))
        for rang, c in enumerate(coefficients_decroissants)
    )


def transmittance_d_une_equation(
    sortie: Sequence[sp.Expr], entree: Sequence[sp.Expr]
) -> Transmittance:
    """Les coefficients des derivees de la sortie et de l'entree, degre decroissant."""
    if not sortie or not entree:
        msg = "equation sans terme de sortie ou d'entree : aucune transmittance"
        raise ValueError(msg)
    return Transmittance(expression=_polynome(entree) / _polynome(sortie))


def coefficients(transmittance: Transmittance) -> tuple[list[float], list[float]]:
    """Numerateur et denominateur par degre decroissant, normalises a `scipy`."""
    numerateur = sp.Poly(transmittance.numerateur(), IMAGE).all_coeffs()
    denominateur = sp.Poly(transmittance.denominateur(), IMAGE).all_coeffs()
    return [float(c) for c in numerateur], [float(c) for c in denominateur]
