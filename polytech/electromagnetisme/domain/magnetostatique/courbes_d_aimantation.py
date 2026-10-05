"""Les courbes d'aimantation et le cycle d'hysteresis.

Le syllabus d'electricite generale (10.4, 10.5 et 10.7). Amener un milieu
de `(H, B)` a `(H + dH, B + dB)` coute le travail volumique `dw = H . dB`
(10.4.1) ; le long d'un cycle ferme, `w = integrale de H . dB` (10.7.2)
est l'aire du cycle, « essentiellement positive », dissipee en chaleur.

Sur la courbe de premiere aimantation, `M` tend vers l'aimantation a
saturation : `B` admet une asymptote oblique de pente `mu_0`. La
permeabilite `B / H` part de sa valeur initiale `mu_i` ; la permeabilite
differentielle est la pente `dB / dH` de la tangente.

Sur le cycle, l'induction remanente `B_r` se lit a champ nul et le champ
coercitif `H_c` a induction nulle. Un cycle est ici la suite de ses
sommets `(H, B)`, dans le sens ou il est parcouru.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    Cycle = Sequence[tuple[float, float]]

SOMMETS_MINIMAUX = 3


def travail_volumique(
    induction: sp.Expr, champ: sp.Symbol, depart: object, arrivee: object
) -> sp.Expr:
    """`integrale de H dB` (10.7.1) le long de la courbe `B(H)`, en J/m^3."""
    return sp.integrate(champ * sp.diff(induction, champ), (champ, depart, arrivee))


def permeabilite_differentielle(induction: sp.Expr, champ: sp.Symbol) -> sp.Expr:
    """`mu_diff = dB / dH` : la pente de la tangente a la courbe."""
    return sp.diff(induction, champ)


def permeabilite_initiale(induction: sp.Expr, champ: sp.Symbol) -> sp.Expr:
    """`mu_i` : la limite de `B / H` quand le champ part de zero."""
    return sp.limit(induction / champ, champ, 0, "+")


def pente_asymptotique(induction: sp.Expr, champ: sp.Symbol) -> sp.Expr:
    """La limite de `dB / dH` a champ infini : `mu_0` une fois le milieu sature."""
    return sp.limit(sp.diff(induction, champ), champ, sp.oo)


def _cotes(cycle: Cycle) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(cycle) < SOMMETS_MINIMAUX:
        msg = f"un cycle a au moins trois points, pas {len(cycle)}"
        raise ValueError(msg)
    return list(pairwise([*cycle, cycle[0]]))


def aire_du_cycle(cycle: Cycle) -> float:
    """`integrale fermee de H dB` (10.7.2), par trapezes, en J/m^3.

    Positive dans le sens physique du parcours : l'induction retarde sur
    le champ.
    """
    return sum((h1 + h2) / 2 * (b2 - b1) for (h1, b1), (h2, b2) in _cotes(cycle))


def _croisements(cycle: Cycle, axe: int) -> list[float]:
    """L'autre coordonnee la ou celle de rang `axe` change de signe."""
    lus = []
    for debut, fin in _cotes(cycle):
        if debut[axe] == 0:
            lus.append(debut[1 - axe])
        elif debut[axe] * fin[axe] < 0:
            part = debut[axe] / (debut[axe] - fin[axe])
            lus.append(debut[1 - axe] + part * (fin[1 - axe] - debut[1 - axe]))
    if not lus:
        msg = f"le cycle ne coupe pas l'axe {'HB'[1 - axe]} : rien a y lire"
        raise ValueError(msg)
    return lus


def induction_remanente(cycle: Cycle) -> float:
    """`B_r`, en teslas : l'induction qui subsiste a champ nul."""
    return max(_croisements(cycle, 0))


def champ_coercitif(cycle: Cycle) -> float:
    """`H_c`, en A/m : le champ positif qui annule l'induction."""
    return max(_croisements(cycle, 1))
