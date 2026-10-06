"""Seance « Chap. 4-5 », exercices 3 et 7 : deux circuits et un graphe.

Un passe-bas `RC`, un passe-haut `RL`, et un graphe de fluence canonique
a deux integrateurs, que la regle de Mason transforme en transmittance.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

RESISTANCE: Final = sp.Integer(100)
CAPACITE: Final = sp.Rational(1, 10**9)
INDUCTANCE: Final = sp.Rational(1, 10**3)
ALPHA: Final = sp.Symbol("alpha", positive=True)


def graphe(retour: object = 20) -> sp.Expr:
    """Deux integrateurs : retours `-retour` et `-200`, sortie sur le premier."""
    return transmittance_du_graphe(retours=(-sp.sympify(retour), -200), directs=(1, 0))


def transmittances() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le passe-bas `RC`, le passe-haut `RL`, puis le graphe."""
    rc = 1 / (1 + RESISTANCE * CAPACITE * P)
    rl = INDUCTANCE * P / (RESISTANCE + INDUCTANCE * P)
    return sp.cancel(rc), sp.cancel(rl), graphe()


def constantes_de_temps() -> tuple[float, float]:
    """`RC` et `L/R`, en secondes."""
    return float(RESISTANCE * CAPACITE), float(INDUCTANCE / RESISTANCE)


def extremes_des_indicielles() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """`a(0+)` et `a(infini)` : les limites de `H(p)` a l'infini et en zero."""
    noms = ("rc", "rl", "graphe")
    return {
        nom: (sp.limit(h, P, sp.oo), sp.limit(h, P, 0))
        for nom, h in zip(noms, transmittances(), strict=True)
    }


def impulsionnelle_du_graphe() -> sp.Expr:
    """`h(t)` du graphe : l'original de sa transmittance."""
    brut = sp.inverse_laplace_transform(graphe(), P, T)
    return sp.simplify(brut.subs(sp.Heaviside(T), 1))


def premier_zero_du_graphe() -> sp.Expr:
    """Le premier instant ou `h(t)` s'annule."""
    return sp.solve(sp.cos(10 * T) - sp.sin(10 * T), T)[0]


def alpha_limite() -> sp.Expr:
    """Exercice 7 : le plus petit `alpha` pour lequel les poles sont reels."""
    denominateur = sp.fraction(sp.cancel(graphe(ALPHA)))[1]
    discriminant = sp.discriminant(denominateur, P)
    (limite,) = sp.solve(discriminant, ALPHA)
    return limite


def poles_pour_alpha(alpha: object) -> tuple[sp.Expr, ...]:
    """Exercice 7 : les deux poles du graphe pour cette valeur de `alpha`."""
    denominateur = sp.fraction(sp.cancel(graphe(alpha)))[1]
    return tuple(sp.Poly(denominateur, P).all_roots())
