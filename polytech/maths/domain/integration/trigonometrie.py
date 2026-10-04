"""Deux gestes pour les integrales trigonometriques, nommes par les exercices.

**Lineariser.** Une puissance ou un produit de sinus et de cosinus se
reecrit en somme de sinus et de cosinus d'angles multiples, qui
s'integrent a vue : `cos^5 t = (cos 5t + 5 cos 3t + 10 cos t) / 16`.

**Poser `t = tan(x/2)`.** Toute fraction rationnelle en `sin x` et
`cos x` devient une fraction rationnelle en `t` :

    sin x = 2t / (1 + t^2),   cos x = (1 - t^2) / (1 + t^2),
    dx = 2 dt / (1 + t^2).

La primitive obtenue n'est definie que sur les intervalles ou `tan(x/2)`
l'est : la recoller sur la droite entiere demande un saut a chaque
multiple impair de `pi`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp
from sympy.simplify.fu import TR8

from polytech.maths.domain.symboles import VARIABLE

T_DEMI: Final = sp.Symbol("t", real=True)
"""`t = tan(x/2)`."""

PASSES_MAXIMALES: Final = 8
"""Chaque passe abaisse les degres ; huit suffisent largement aux exercices."""


def lineariser(expression: sp.Expr) -> sp.Expr:
    """La somme de sinus et cosinus d'angles multiples egale a l'expression."""
    courante = sp.expand(expression)
    for _ in range(PASSES_MAXIMALES):
        suivante = sp.expand(TR8(courante))
        if suivante == courante:
            break
        courante = suivante
    return courante


def par_la_tangente_de_l_arc_moitie(integrande: sp.Expr) -> sp.Expr:
    """L'integrande en `t`, le facteur `dx = 2 dt / (1 + t^2)` compris."""
    carre = 1 + T_DEMI**2
    en_t = integrande.subs(
        {
            sp.sin(VARIABLE): 2 * T_DEMI / carre,
            sp.cos(VARIABLE): (1 - T_DEMI**2) / carre,
            sp.tan(VARIABLE): 2 * T_DEMI / (1 - T_DEMI**2),
        }
    )
    if en_t.has(VARIABLE):
        msg = f"l'integrande n'est pas rationnel en sin x et cos x : {integrande}"
        raise ValueError(msg)
    return sp.simplify(en_t * 2 / carre)
