"""La pente asymptotique que la carte des poles et zeros impose.

Chaque pole retire 20 dB par decade a haute frequence, chaque zero en
ajoute autant. La carte donne donc la pente sans qu'aucun calcul de module
soit necessaire -- mais elle ne donne pas le gain, qui depend d'un facteur
que la carte ne montre pas.
"""

from __future__ import annotations

import sympy as sp

PENTE_PAR_ORDRE = 20.0
"""Decibels par decade qu'un pole ou un zero unique apporte."""


def pente_asymptotique(zeros: int, poles: int) -> float:
    """Rend la pente en dB par decade, negative quand les poles dominent."""
    if zeros < 0 or poles < 0:
        msg = f"comptes negatifs de zeros ou de poles : {zeros}, {poles}"
        raise ValueError(msg)
    return (zeros - poles) * PENTE_PAR_ORDRE


def carte_des_poles_et_zeros(
    transfert: sp.Expr,
    variable: sp.Expr,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Les poles puis les zeros, tries : la carte que le fascicule analyse.

    Il y consacre cinq planches, parce que c'est elle qui explique
    **visuellement** ce que chaque transposition de frequence fait -- le
    passe-haut echange poles et zeros autour du cercle unite, le
    passe-bande dedouble chaque pole.
    """
    numerateur, denominateur = sp.fraction(sp.cancel(sp.together(transfert)))
    poles = sp.solve(sp.Eq(sp.expand(denominateur), 0), variable)
    zeros = sp.solve(sp.Eq(sp.expand(numerateur), 0), variable)
    return (
        tuple(sorted(poles, key=sp.default_sort_key)),
        tuple(sorted(zeros, key=sp.default_sort_key)),
    )


def harmonique_filtree(amplitude: float, module_du_filtre: float) -> float:
    """L'amplitude d'une harmonique apres le filtre.

    « Filtering digital signals — harmonics » : le fascicule montre ce que
    Butterworth et Bessel font a chaque harmonique d'un signal carre, et
    pourquoi le second deforme moins -- il ne les attenue pas mieux, il les
    retarde **egalement**.
    """
    if module_du_filtre < 0:
        msg = f"module du filtre negatif : {module_du_filtre}"
        raise ValueError(msg)
    return amplitude * module_du_filtre
