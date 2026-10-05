"""Primitiver une fonction irrationnelle en la rendant rationnelle.

Le changement de variable « de type II » remplace `x` par une fonction
`x = phi(t)` choisie pour faire disparaitre les radicaux :

* une racine `n`-ieme d'une homographie -- poser `t^n = (ax + b)/(cx + d)` ;
* plusieurs radicaux de `x` -- poser `t^k = x`, `k` multiple commun des
  indices ;
* `racine(x^2 + 1)` -- poser `t = racine(x^2 + 1) - x`, soit
  `x = (1 - t^2) / (2t)` (changement d'Euler).

L'integrale devient celle de `f(phi(t)) phi'(t)`, rationnelle en `t` ; on
revient a `x` par `t = psi(x)`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def indice_commun(*indices: int) -> int:
    """Le plus petit multiple commun des indices des radicaux."""
    return math.lcm(*indices)


def par_changement_inverse(
    integrande: sp.Expr, nouvelle: sp.Symbol, x_de_t: sp.Expr, t_de_x: sp.Expr
) -> sp.Expr:
    """La primitive en `x`, par `x = phi(t)` puis retour par `t = psi(x)`."""
    brut = integrande.subs(VARIABLE, x_de_t) * sp.diff(x_de_t, nouvelle)
    en_t = sp.simplify(sp.factor(brut, deep=True))
    if not en_t.is_rational_function(nouvelle):
        msg = f"le changement ne rend pas l'integrande rationnel : {en_t}"
        raise ValueError(msg)
    return sp.integrate(en_t, nouvelle).subs(nouvelle, t_de_x)
