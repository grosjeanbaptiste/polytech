"""Exercices resolus, 0.9 et 0.10 : deux produits de convolution graphiques.

Convoluer, c'est retourner l'un des signaux, le faire glisser de `t`, et
mesurer l'aire du produit.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.frequence import INSTANT as T
from polytech.signaux.domain.frequence import rect
from polytech.signaux.domain.produits import convolution_en

RAMPE: Final = T * rect((T - 1) / 2)
"""`x(t)` de l'exercice 0.9 : il monte de 0 a 2 entre 0 et 2."""
CRENEAU: Final = 4 * rect((T - sp.Rational(3, 2)) / 3)
"""`h(t)` de l'exercice 0.9 : il vaut 4 entre 0 et 3."""
PORTE: Final = rect(T / 2 - 1)
"""`rect(t/2 - 1)` de l'exercice 0.10 : elle vaut 1 entre 1 et 3."""


def sortie_de_la_rampe(instant: object) -> sp.Expr:
    """`y(t) = (x * h)(t)` de l'exercice 0.9."""
    return convolution_en(RAMPE, CRENEAU, sp.sympify(instant))


def aire_du_trapeze_hachure() -> sp.Expr:
    """Le produit `x(tau) h(4 - tau)` entre 1 et 2 : un trapeze de largeur un."""
    petit, grand = (4 * RAMPE.subs(T, borne) for borne in (1, 2))
    return (petit + grand) / 2


def sortie_de_l_echelon(instant: object) -> sp.Expr:
    """`(echelon * porte)(t)` : l'aire de la porte deja passee a l'instant `t`."""
    return convolution_en(sp.Heaviside(T), PORTE, sp.sympify(instant))


def saut_si_le_milieu_valait_t() -> tuple[sp.Expr, sp.Expr]:
    """Les sauts en 1 et en 3 qu'aurait la sortie si son milieu valait `t`."""
    avant, apres = sortie_de_l_echelon(1), sortie_de_l_echelon(3)
    return 1 - avant, apres - 3
