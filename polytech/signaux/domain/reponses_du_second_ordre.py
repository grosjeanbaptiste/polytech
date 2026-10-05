"""Les reponses impulsionnelle et indicielle du second ordre.

Syllabus de signaux et systemes, sections 5.2.1 a 5.2.3. La forme de la
reponse ne depend que de la position des poles.

* Complexes conjugues : une sinusoide amortie, de pulsation
  `omega = racine(1 - xi^2) / T` et de constante de temps `T / xi`.
* Reels confondus : une exponentielle multipliee par une rampe.
* Reels distincts : une difference de deux exponentielles, de constantes
  de temps `T_1` et `T_2`.

La reponse indicielle y ajoute le terme de regime `K`, du au pole en
zero. Quand elle oscille, son premier maximum tombe a la demi-periode
propre, `t_max = pi T / racine(1 - xi^2)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import TEMPS as T

if TYPE_CHECKING:
    from polytech.signaux.domain.second_ordre import SecondOrdre

COMPLEXES = "poles complexes conjugues"
CONFONDUS = "poles reels confondus"


def _constantes_de_temps(systeme: SecondOrdre) -> tuple[sp.Expr, sp.Expr]:
    lent, rapide = sorted((-1 / pole for pole in systeme.poles()), reverse=True)
    return (lent, rapide)


def impulsionnelle(systeme: SecondOrdre) -> sp.Expr:
    """`h(t)`, selon le regime : (5.29), (5.34) ou (5.37)."""
    gain, constante = systeme.gain, systeme.constante
    if systeme.regime() == COMPLEXES:
        pulsation = systeme.pulsation_propre()
        amortie = sp.exp(-systeme.amortissement * T / constante)
        return gain / (constante**2 * pulsation) * amortie * sp.sin(pulsation * T)
    if systeme.regime() == CONFONDUS:
        return gain / constante**2 * T * sp.exp(-T / constante)
    lent, rapide = _constantes_de_temps(systeme)
    return gain / (lent - rapide) * (sp.exp(-T / lent) - sp.exp(-T / rapide))


def indicielle(systeme: SecondOrdre) -> sp.Expr:
    """`a(t)`, selon le regime : (5.39), (5.42) ou (5.45)."""
    gain, constante = systeme.gain, systeme.constante
    if systeme.regime() == COMPLEXES:
        pulsation = systeme.pulsation_propre()
        retard = sp.atan(systeme.amortissement / (pulsation * constante))
        amortie = sp.exp(-systeme.amortissement * T / constante)
        oscillation = sp.cos(pulsation * T - retard) / (pulsation * constante)
        return gain * (1 - amortie * oscillation)
    if systeme.regime() == CONFONDUS:
        return gain * (1 - sp.exp(-T / constante) * (1 + T / constante))
    lent, rapide = _constantes_de_temps(systeme)
    lente = lent / (lent - rapide) * sp.exp(-T / lent)
    return gain * (1 - lente - rapide / (rapide - lent) * sp.exp(-T / rapide))


def periode_propre(systeme: SecondOrdre) -> sp.Expr:
    """`2 pi T / racine(1 - xi^2)`, la periode des oscillations amorties."""
    return 2 * sp.pi / systeme.pulsation_propre()


def instant_du_maximum(systeme: SecondOrdre) -> sp.Expr:
    """`t_max = pi T / racine(1 - xi^2)`, la demi-periode propre."""
    return sp.pi / systeme.pulsation_propre()
