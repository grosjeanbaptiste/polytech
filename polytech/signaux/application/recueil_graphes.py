"""Recueil d'exercices, planches 9 et 14 : regler un graphe de fluence.

Le graphe canonique se lit par la regle de Mason : les gains de retour
donnent le denominateur, les gains directs le numerateur. Regler un gain,
c'est deplacer un pole ou fixer une limite de la reponse.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.application.systemes_exs_45 import (
    alpha_limite,
    poles_pour_alpha,
)
from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.symboles import IMAGE as P

__all__ = [
    "alpha_limite",
    "est_stable_une_fois_regle",
    "poles_pour_alpha",
    "reglages",
    "transmittance_a_quatre_integrateurs",
    "valeur_finale_de_l_indicielle",
    "valeur_initiale_de_l_impulsionnelle",
]

K, M = sp.symbols("k m", real=True)
VALEUR_INITIALE_VOULUE: Final = 1
VALEUR_FINALE_VOULUE: Final = 2


def transmittance_a_quatre_integrateurs() -> sp.Expr:
    """Planche 14 : retours `-10`, `-100`, `-200`, `m` ; gains directs `k`, 1, 1, 1."""
    return transmittance_du_graphe(retours=(-10, -100, -200, M), directs=(K, 1, 1, 1))


def valeur_initiale_de_l_impulsionnelle() -> sp.Expr:
    """`h(0+) = lim p H(p)` quand `p` tend vers l'infini."""
    return sp.limit(P * transmittance_a_quatre_integrateurs(), P, sp.oo)


def valeur_finale_de_l_indicielle() -> sp.Expr:
    """`a(infini) = H(0)`, si le systeme est stable."""
    return sp.simplify(transmittance_a_quatre_integrateurs().subs(P, 0))


def reglages() -> dict[sp.Symbol, sp.Expr]:
    """Les `k` et `m` qui donnent les deux limites voulues."""
    conditions = (
        sp.Eq(valeur_initiale_de_l_impulsionnelle(), VALEUR_INITIALE_VOULUE),
        sp.Eq(valeur_finale_de_l_indicielle(), VALEUR_FINALE_VOULUE),
    )
    solution: dict[sp.Symbol, sp.Expr] = sp.solve(conditions, (K, M))
    return solution


def est_stable_une_fois_regle() -> bool:
    """Tous les poles a gauche de l'axe imaginaire, pour les `k` et `m` trouves."""
    regle = transmittance_a_quatre_integrateurs().subs(reglages())
    poles = sp.Poly(sp.fraction(sp.cancel(regle))[1], P).nroots()
    return all(sp.re(pole) < 0 for pole in poles)
