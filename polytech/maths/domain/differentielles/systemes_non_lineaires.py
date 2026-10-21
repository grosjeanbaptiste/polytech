"""Les systemes normaux non lineaires, et la mise sous forme normale.

Le theoreme 6.1 vaut pour `y' = f(t, y)` quelconque, continue et
lipschitzienne : les iterees de `T(y)(t) = y0 + int_t0^t f(tau, y(tau))
dtau` convergent vers la solution. `systemes_normaux` ne les calcule que
pour `f = A y + x` ; ici `f` est un vecteur d'expressions en `t` et en
`y_1, ..., y_n`.

Une equation d'ordre `n` resolue, `y^(n) = phi(t, y, ..., y^(n-1))`, se met
sous forme normale par `y_1 = y, ..., y_n = y^(n-1)` (6.5) :
`y_k' = y_(k+1)` pour `k < n`, et `y_n' = phi`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import TEMPS

INCONNUES: Final = sp.symbols("y1:10", real=True)
"""`y_1, ..., y_9` : les composantes de l'inconnue vectorielle."""


def sous_forme_normale(phi: sp.Expr, ordre: int) -> sp.Matrix:
    """`(y_2, ..., y_n, phi)`, le second membre du systeme equivalent."""
    if not 1 <= ordre <= len(INCONNUES):
        msg = f"ordre {ordre} hors de 1..{len(INCONNUES)}"
        raise ValueError(msg)
    au_dela = sorted(set(INCONNUES[ordre:]) & phi.free_symbols, key=str)
    if au_dela:
        msg = f"phi cite {au_dela[0]}, au-dela de l'ordre {ordre}"
        raise ValueError(msg)
    return sp.Matrix([*INCONNUES[1:ordre], phi])


def iterees_de_picard(
    second_membre: sp.Matrix,
    conditions: sp.Matrix,
    *,
    nombre: int,
    t0: object = 0,
) -> tuple[sp.Matrix, ...]:
    """`y_0 = 0`, puis `y_(k+1) = T(y_k)`, jusqu'a `y_nombre`."""
    taille = second_membre.rows
    if conditions.rows != taille:
        msg = f"{conditions.rows} conditions pour {taille} equations"
        raise ValueError(msg)
    muette = sp.Symbol("tau", real=True)
    iterees = [sp.zeros(taille, 1)]
    for _ in range(nombre):
        valeurs = dict(zip(INCONNUES[:taille], iterees[-1], strict=True))
        integrande = second_membre.subs(valeurs).subs(TEMPS, muette)
        integrale = integrande.applyfunc(
            lambda e: sp.integrate(e, (muette, sp.sympify(t0), TEMPS))
        )
        iterees.append((conditions + integrale).applyfunc(sp.expand))
    return tuple(iterees)
