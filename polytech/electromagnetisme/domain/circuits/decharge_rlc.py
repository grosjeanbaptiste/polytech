"""La decharge d'un condensateur dans une resistance et une bobine.

Le syllabus d'electricite generale (15.12). Un condensateur charge `Q_0`
se ferme sur `R` et `L` : `L q'' + R q' + q / C = 0` (15.12.2), sans
phenomene de regime. Avec `T^2 = L C` et le coefficient d'amortissement
`xi = (R / 2) racine(C / L)`, la transformee de Laplace de la charge est
`(p T^2 + 2 xi T) Q_0 / (p^2 T^2 + 2 xi p T + 1)` (15.12.4), et celle du
courant `-Q_0 / (p^2 T^2 + 2 xi p T + 1)` (15.12.5).

Le signe de `xi^2 - 1` separe trois cas : deux racines reelles, les
phenomenes sont **aperiodiques** (15.12.8) ; une racine double, ils sont
aperiodiques **critiques** (15.12.12) ; deux racines conjuguees, ils sont
**pseudo-periodiques** (15.12.16), et periodiques a la limite `R = 0`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE


class Regime(Enum):
    """Les trois allures de la decharge."""

    APERIODIQUE = "aperiodique"
    CRITIQUE = "aperiodique critique"
    PSEUDO_PERIODIQUE = "pseudo-periodique"


def regime_de_la_decharge(amortissement: Any) -> Regime:  # noqa: ANN401 -- reel ou rationnel
    """Le cas que le discriminant `4 T^2 (xi^2 - 1)` designe."""
    if amortissement < 0:
        msg = f"amortissement negatif : xi = {amortissement} -- circuit non passif"
        raise ValueError(msg)
    if amortissement == 1:
        return Regime.CRITIQUE
    return Regime.APERIODIQUE if amortissement > 1 else Regime.PSEUDO_PERIODIQUE


def _denominateur(amortissement: Any, periode: Any) -> sp.Expr:  # noqa: ANN401 -- reels ou symboles
    lineaire = 2 * amortissement * VARIABLE * periode
    return VARIABLE**2 * periode**2 + lineaire + 1


def transformee_de_la_charge(amortissement: Any, periode: Any, initiale: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`Q(p) = (p T^2 + 2 xi T) Q_0 / (p^2 T^2 + 2 xi p T + 1)` (15.12.4)."""
    numerateur = VARIABLE * periode**2 + 2 * amortissement * periode
    return numerateur * initiale / _denominateur(amortissement, periode)


def transformee_du_courant(amortissement: Any, periode: Any, initiale: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`I(p) = -Q_0 / (p^2 T^2 + 2 xi p T + 1)` (15.12.5)."""
    return -initiale / _denominateur(amortissement, periode)


def _formes(amortissement: Any, reduit: sp.Expr) -> tuple[sp.Expr, sp.Expr]:  # noqa: ANN401 -- reel ou rationnel
    """`(sinus(r t / T) / r, cosinus(r t / T))`, hyperboliques ou circulaires."""
    regime = regime_de_la_decharge(amortissement)
    if regime is Regime.CRITIQUE:
        return reduit, sp.Integer(1)
    racine = sp.sqrt(sp.Abs(sp.sympify(amortissement) ** 2 - 1))
    if regime is Regime.APERIODIQUE:
        return sp.sinh(racine * reduit) / racine, sp.cosh(racine * reduit)
    return sp.sin(racine * reduit) / racine, sp.cos(racine * reduit)


def charge_de_decharge(amortissement: Any, periode: Any, initiale: Any) -> sp.Expr:  # noqa: ANN401 -- reels ou symboles
    """`q(t)` : (15.12.8), (15.12.12) ou (15.12.16) selon l'amortissement."""
    reduit = TEMPS / periode
    sinus, cosinus = _formes(amortissement, reduit)
    enveloppe = sp.exp(-amortissement * reduit)
    return initiale * (amortissement * sinus + cosinus) * enveloppe


def courant_de_decharge(amortissement: Any, periode: Any, initiale: Any) -> sp.Expr:  # noqa: ANN401 -- reels ou symboles
    """`i(t)` : (15.12.9), (15.12.13) ou (15.12.17) selon l'amortissement."""
    reduit = TEMPS / periode
    sinus, _ = _formes(amortissement, reduit)
    return -initiale / periode * sinus * sp.exp(-amortissement * reduit)
