"""Exercices resolus, 0.4 a 0.6 : valeurs efficaces et rapport signal sur bruit.

Un signal periodique se mesure par sa puissance, la moyenne de son carre
sur une periode ; la valeur efficace en est la racine.
"""

from __future__ import annotations

import itertools
from typing import Final

import sympy as sp

from polytech.signaux.domain.elementaires import rapport_signal_sur_bruit_en_decibels
from polytech.signaux.domain.frequence import INSTANT as T
from polytech.signaux.domain.frequence import rect, tri
from polytech.signaux.domain.operations import decaler
from polytech.signaux.domain.produits import produit_scalaire_periodique

A: Final = sp.Symbol("a", positive=True)
DEMI: Final = sp.Rational(1, 2)
TRAINS: Final = (
    (rect(T), (-DEMI, DEMI), sp.Integer(2)),
    (rect(T), (-DEMI, DEMI), sp.Rational(3, 4)),
    (tri(T), (-1, 0, 1), sp.Integer(1)),
)
"""Les trains `x`, `y`, `z` de 0.4 : le motif, ses ruptures, la periode."""
_REPLIQUES: Final = 3


def _morceau(motif: sp.Expr, instant: sp.Expr) -> sp.Expr:
    """L'expression que ce motif, defini par morceaux, prend autour de l'instant."""
    for expression, condition in motif.args:
        if condition == sp.true or condition.subs(T, instant):
            return expression
    msg = f"aucun morceau de {motif} ne contient l'instant {instant}"
    raise ValueError(msg)


def train(motif: sp.Expr, periode: sp.Expr, instant: sp.Expr) -> sp.Expr:
    """La somme des repliques du motif, comme expression valable pres de l'instant."""
    rangs = range(-_REPLIQUES, _REPLIQUES + 1)
    repliques = (decaler(motif, rang * periode) for rang in rangs)
    return sum((_morceau(r, instant) for r in repliques), sp.S.Zero)


def puissance_du_train(
    motif: sp.Expr, ruptures: tuple[sp.Expr, ...], periode: sp.Expr
) -> sp.Expr:
    """`(1/T0) integrale de x^2` sur une periode, morceau par morceau."""
    rangs = range(-_REPLIQUES, _REPLIQUES + 1)
    toutes = {sp.sympify(r) + rang * periode for r in ruptures for rang in rangs}
    bornes = sorted({b for b in toutes if 0 < b < periode} | {sp.Integer(0), periode})
    total = sp.S.Zero
    for debut, fin in itertools.pairwise(bornes):
        somme = train(motif, periode, (debut + fin) / 2)
        total += sp.integrate(somme**2, (T, debut, fin))
    return sp.simplify(total / periode)


def valeurs_efficaces_des_trains() -> tuple[sp.Expr, ...]:
    """La racine de la puissance des trains `x`, `y` et `z`."""
    return tuple(sp.sqrt(puissance_du_train(*donnees)) for donnees in TRAINS)


def train_de_triangles(instant: object) -> sp.Expr:
    """`z(t)` en un instant : des triangles de base 2, un par seconde."""
    motif, _, periode = TRAINS[2]
    point = sp.sympify(instant)
    return train(motif, periode, point).subs(T, point)


def valeurs_efficaces_des_formes() -> dict[str, sp.Expr]:
    """Quatre formes d'amplitude `a` et de moyenne nulle, sur une periode unite."""
    formes = {
        "sinus": A * sp.sin(2 * sp.pi * T),
        "rectangle": A * sp.sign(T),
        "triangle": A * (1 - 4 * sp.Abs(T)),
        "rampe": 2 * A * T,
    }
    return {
        nom: sp.sqrt(produit_scalaire_periodique(forme, forme, sp.Integer(1)))
        for nom, forme in formes.items()
    }


def rapport_signal_sur_bruit(*, gain_du_bruit: float = 1.0) -> float:
    """Le sinus sur le rectangle de meme amplitude, le bruit etant amplifie."""
    efficaces = valeurs_efficaces_des_formes()
    signal = float((efficaces["sinus"] ** 2).subs(A, 1))
    bruit = float((efficaces["rectangle"] ** 2).subs(A, 1)) * gain_du_bruit**2
    return rapport_signal_sur_bruit_en_decibels(signal, bruit)


def facteur_d_amplitude(decibels: float) -> float:
    """Le rapport d'amplitudes que valent ces decibels : `10^(dB/20)`."""
    return float(10 ** (decibels / 20))
