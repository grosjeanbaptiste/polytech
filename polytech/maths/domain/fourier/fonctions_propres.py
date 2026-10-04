"""Les fonctions propres d'un systeme lineaire : pourquoi des exponentielles.

Pour `K(D) y = x(t)`, la sollicitation `phi_n(t) = e^(j n w t)` donne une
reponse de meme forme, `e^(j n w t) / K(j n w)` : c'est un vecteur propre
du systeme, de valeur propre `1 / K(j n w)`. Quand toutes les racines de
`K` sont a partie reelle negative, le transitoire s'eteint et il ne reste
que cela.

Decomposer `x(t)` en serie de Fourier, `somme c_n e^(j n w t)`, donne donc
la reponse terme a terme : `somme c_n e^(j n w t) / K(j n w)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polytech.maths.domain.differentielles.operateurs import OperateurDifferentiel


def harmonique(rang: int, pulsation: object) -> sp.Expr:
    """`phi_n(t) = e^(j n w t)`."""
    return sp.exp(sp.I * rang * sp.sympify(pulsation) * TEMPS)


def reponse_a_un_harmonique(
    operateur: OperateurDifferentiel, rang: int, pulsation: object
) -> sp.Expr:
    """`e^(j n w t) / K(j n w)` ; leve si `j n w` est racine de `K`."""
    valeur = operateur.en(sp.I * rang * sp.sympify(pulsation))
    if valeur == 0:
        msg = f"harmonique resonant : K(j n w) = 0 pour n = {rang}"
        raise ValueError(msg)
    return harmonique(rang, pulsation) / valeur


def reponse_periodique(
    operateur: OperateurDifferentiel,
    coefficients: Mapping[int, object],
    pulsation: object,
) -> sp.Expr:
    """`somme c_n e^(j n w t) / K(j n w)`."""
    termes = (
        sp.sympify(c) * reponse_a_un_harmonique(operateur, n, pulsation)
        for n, c in coefficients.items()
    )
    return sum(termes, sp.Integer(0))
