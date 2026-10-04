"""Les systemes normaux `y' = P(t) y + x(t)` et leur probleme de Cauchy.

Le theoreme d'existence et d'unicite du chapitre 5 passe aux vecteurs
sans changement : la transformation

    T(y)(t) = C + int_{t_0}^{t} (P(tau) y(tau) + x(tau)) dtau

est contractante pres de `t_0`, de constante de Lipschitz `||P||`, et ses
iterees convergent vers la solution. Pour une matrice constante `A` elles
construisent, terme apres terme, `e^(A (t - t_0)) C`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.series.uniforme import distance_uniforme
from polytech.maths.domain.symboles import TEMPS


def iterees_vectorielles(
    matrice: sp.Matrix,
    conditions: sp.Matrix,
    *,
    nombre: int,
    t0: object = 0,
    force: sp.Matrix | None = None,
) -> tuple[sp.Matrix, ...]:
    """`y_0 = 0`, puis `y_(k+1) = T(y_k)`, jusqu'a `y_nombre`."""
    muette = sp.Symbol("tau", real=True)
    libre = sp.zeros(*conditions.shape) if force is None else force
    iterees = [sp.zeros(*conditions.shape)]
    for _ in range(nombre):
        integrande = (matrice * iterees[-1] + libre).subs(TEMPS, muette)
        integrale = integrande.applyfunc(
            lambda e: sp.integrate(e, (muette, sp.sympify(t0), TEMPS))
        )
        iterees.append((conditions + integrale).applyfunc(sp.expand))
    return tuple(iterees)


def norme_de_la_matrice(
    matrice: sp.Matrix, intervalle: sp.Set | None = None
) -> sp.Expr:
    """La racine de la somme des carres des `sup |p_jk(t)|`."""
    carres = [
        distance_uniforme(element, 0, TEMPS, intervalle) ** 2
        if element.has(TEMPS)
        else element**2
        for element in matrice
    ]
    return sp.sqrt(sum(carres, sp.Integer(0)))
