"""Exercices resolus, 0.2, 0.3, 0.7 (a) et 0.8 : des signaux d'energie finie.

Un signal d'energie finie se mesure par son energie, le produit scalaire
de ce signal par lui-meme ; le produit scalaire compare deux signaux
comme on compare deux vecteurs.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.frequence import INSTANT as T
from polytech.signaux.domain.frequence import rect, tri
from polytech.signaux.domain.operations import decaler, dilater
from polytech.signaux.domain.produits import produit_scalaire_d_energie

SIGNAUX_D_ESSAI: Final = (rect(T), tri(T), sp.exp(-sp.Abs(T)))


def reperes_du_triangle() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Le debut, le sommet, la fin et la hauteur de `tri(3t - 2)/2`."""
    argument = 3 * T - 2
    debut, sommet, fin = (sp.solve(sp.Eq(argument, v), T)[0] for v in (-1, 0, 1))
    return debut, sommet, fin, (tri(argument) / 2).subs(T, sommet)


def energies() -> tuple[sp.Expr, sp.Expr]:
    """L'energie de `rect(t)`, puis celle de `tri(t)`."""
    return energie_amplifiee(1)


def energie_amplifiee(gain: int) -> tuple[sp.Expr, sp.Expr]:
    """Les memes energies, les signaux etant multiplies par `gain`."""
    porte, triangle = gain * rect(T), gain * tri(T)
    return (
        produit_scalaire_d_energie(porte, porte),
        produit_scalaire_d_energie(triangle, triangle),
    )


def produit_scalaire_des_formes() -> sp.Expr:
    """`<rect, tri>` : l'aire du triangle sous la porte."""
    return produit_scalaire_d_energie(rect(T), tri(T))


def rapports_d_energie(signal: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """L'energie de `x(2t)` puis de `x(t - 2)`, rapportees a celle de `x(t)`."""
    reference = produit_scalaire_d_energie(signal, signal)
    comprime, retarde = dilater(signal, sp.Integer(2)), decaler(signal, sp.Integer(2))
    return (
        sp.simplify(produit_scalaire_d_energie(comprime, comprime) / reference),
        sp.simplify(produit_scalaire_d_energie(retarde, retarde) / reference),
    )
