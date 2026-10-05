"""Les impedances de Laplace, avec l'etat qui precede l'instant zero.

Planches du chapitre 1, partie III (planches 11, 12 et 15 a 19). Un
circuit a une histoire : avant la fermeture d'un interrupteur, le
condensateur porte une tension `v(0-)` et la bobine un courant `i(0-)`.
La propriete de la derivee, `L{f'} = s F(s) - f(0-)`, les fait entrer
dans les equations :

    C dv/dt = i     devient   V(s) = I(s) / (s C) + v(0-) / s,
    v = L di/dt     devient   V(s) = s L I(s) - L i(0-).

Chaque composant est alors son impedance, en serie avec une source de
tension qui porte l'etat initial -- ou, de facon equivalente, son
admittance en parallele avec une source de courant. A etat initial nul,
il ne reste que `1 / (s C)` et `s L`.
"""

from __future__ import annotations

import sympy as sp


def tension_du_condensateur(
    capacite: object, courant: object, variable: object, *, initiale: object = 0
) -> sp.Expr:
    """`V(s) = I(s) / (s C) + v(0-) / s`."""
    s = sp.sympify(variable)
    return sp.sympify(courant) / (s * sp.sympify(capacite)) + sp.sympify(initiale) / s


def tension_de_la_bobine(
    inductance: object, courant: object, variable: object, *, initial: object = 0
) -> sp.Expr:
    """`V(s) = s L I(s) - L i(0-)`."""
    propre = sp.sympify(inductance)
    induite = sp.sympify(variable) * propre * sp.sympify(courant)
    return induite - propre * sp.sympify(initial)


def courant_de_la_bobine(
    inductance: object, tension: object, variable: object, *, initial: object = 0
) -> sp.Expr:
    """`I(s) = V(s) / (s L) + i(0-) / s` : la meme loi, lue en admittance."""
    s = sp.sympify(variable)
    return sp.sympify(tension) / (s * sp.sympify(inductance)) + sp.sympify(initial) / s
