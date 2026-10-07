"""Exercice cote 2023 (Mons), question 2 : un cadre dans un champ qui croit avec `x`.

Le champ, de grandeur `B0 + k x`, entre dans la feuille ; le flux est
compte positivement selon `u_z`, qui en sort. Le cadre `a` par `b` glisse
vers la droite a la vitesse `v`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE, TEMPS

T: Final = TEMPS
A, B, D1, D2 = sp.symbols("a b d_1 d_2", positive=True)
B0, K, V = sp.symbols("B_0 k v", positive=True)


def composante_normale(abscisse: sp.Expr = ABSCISSE) -> sp.Expr:
    """`B . u_z = -(B0 + k x)` : le champ entre, la normale sort."""
    return -(B0 + K * abscisse)


def flux(depart: sp.Expr = D1) -> sp.Expr:
    """Le flux a travers le cadre dont le cote gauche est en `depart`."""
    return sp.expand(
        sp.integrate(
            composante_normale(),
            (ORDONNEE, D2, D2 + B),
            (ABSCISSE, depart, depart + A),
        )
    )


def fem() -> sp.Expr:
    """`-d(phi)/dt`, le cote gauche etant en `d1 + v t`."""
    return sp.simplify(force_electromotrice(flux(D1 + V * T)))


def fem_par_les_cotes(gauche: sp.Expr = D1) -> sp.Expr:
    """La circulation de `v x B` : seuls les deux cotes verticaux y contribuent.

    Avec `v` selon `x` et `B` selon `-z`, `v x B` est selon `+y`, de
    grandeur `v (B0 + k x)`. Dans le sens trigonometrique le cote droit se
    parcourt vers le haut, le cote gauche vers le bas. Le champ etant
    lineaire en `x`, l'abscisse `gauche` du cadre ne compte pas.
    """
    droit = -V * composante_normale(gauche + A) * B
    return sp.simplify(droit - -V * composante_normale(gauche) * B)
