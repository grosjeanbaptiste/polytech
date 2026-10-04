"""Une solution particuliere de `K(D) y = x(t)`, selon la forme de `x`.

* `x = P_m(t)`, un polynome. Si `K(0) != 0`, on divise 1 par `K(D)` selon
  les puissances croissantes jusqu'a `D^m` : `1 = K Q + D^(m+1) R`, et
  comme `D^(m+1) P_m = 0`, `y = Q(D) P_m`. Si `K(0) = 0`, `K = D^s phi`
  avec `phi(0) != 0` : `D^s y = P_m / phi(D)`, que l'on integre `s` fois.
* `x = e^(alpha t) P_m(t)` : poser `y = e^(alpha t) z` donne
  `K(D + alpha) z = P_m`, le cas precedent.
* `x = e^(alpha t)` : `y = e^(alpha t) / K(alpha)`, si `K(alpha) != 0`.
* `x = cos(omega t)` ou `sin(omega t)` : la partie reelle ou imaginaire de
  la reponse a `e^(j omega t)`, soit `e^(j omega t) / K(j omega)` hors de
  la resonance, et `e^(j omega t) z` avec `K(D + j omega) z = 1` sinon.

Par linearite, la reponse a une somme de sollicitations est la somme des
reponses : c'est le principe de superposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Callable

PARTIES: dict[object, Callable[[sp.Expr], sp.Expr]] = {sp.cos: sp.re, sp.sin: sp.im}


def pour_un_polynome(operateur: OperateurDifferentiel, polynome: sp.Expr) -> sp.Expr:
    """`Q(D) P_m`, integre autant de fois que 0 est racine de `K`."""
    zero = operateur.racines().get(sp.Integer(0), 0)
    regulier = OperateurDifferentiel(sp.cancel(operateur.polynome / D**zero))
    degre = int(sp.degree(sp.sympify(polynome), TEMPS))
    quotient = OperateurDifferentiel(regulier.inverse_tronque(degre))
    solution = quotient.appliquer(sp.sympify(polynome))
    for _ in range(zero):
        solution = sp.integrate(solution, TEMPS)
    return sp.expand(solution)


def pour_un_polynome_module(
    operateur: OperateurDifferentiel, polynome: sp.Expr, alpha: object
) -> sp.Expr:
    """`e^(alpha t) z`, ou `K(D + alpha) z = P_m`."""
    z = pour_un_polynome(operateur.decaler(alpha), polynome)
    return sp.exp(sp.sympify(alpha) * TEMPS) * z


def pour_une_exponentielle(operateur: OperateurDifferentiel, alpha: object) -> sp.Expr:
    """`e^(alpha t) / K(alpha)`."""
    exposant = sp.sympify(alpha)
    if operateur.en(exposant) == 0:
        msg = f"{exposant} est racine de l'equation caracteristique : K s'y annule"
        raise ValueError(msg)
    return sp.exp(exposant * TEMPS) / operateur.en(exposant)


def pour_une_sinusoide(
    operateur: OperateurDifferentiel, forme: object, pulsation: object
) -> sp.Expr:
    """La partie reelle (cosinus) ou imaginaire (sinus) de `e^(jwt) / K(jw)`."""
    if forme not in PARTIES:
        msg = f"sollicitation ni cosinus ni sinus : {forme}"
        raise ValueError(msg)
    exposant = sp.I * sp.sympify(pulsation)
    complexe = pour_un_polynome_module(operateur, sp.Integer(1), exposant)
    return sp.simplify(PARTIES[forme](sp.expand(complexe, complex=True)))
