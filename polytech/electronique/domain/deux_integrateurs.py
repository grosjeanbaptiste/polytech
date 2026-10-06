"""Les biquads a deux integrateurs : KHN, Tow-Thomas, Fleischer-Tow.

Chapitre 3, partie II, planches 44 a 60. Deux integrateurs inverseurs
dans une boucle, et un etage qui somme : la structure realise
`G a_0 / (s^2 + b_1 s + b_0)`, et ses noeuds intermediaires donnent
gratuitement un passe-bande et un passe-haut.

Les planches impriment les fonctions de transfert « apres quelques
manipulations algebriques ». Ce module les fait sortir de l'analyse
nodale du montage dessine, chaque AOP etant ideal (`aop_ideal`) : un
test peut alors les confronter a ce qui est imprime.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.domain.aop_ideal import tensions_a_aop_ideaux
from polytech.electronique.domain.netliste import Admittance

Six = tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]
_UN = {1: sp.Integer(1)}
"""La source unite au noeud 1 : chaque tension est alors un transfert."""


def _branche(a: int, b: int, resistance: sp.Expr) -> Admittance:
    """Une resistance, l'infini valant une branche retiree."""
    return Admittance(a, b, 1 / sp.sympify(resistance))


def sommateur_differentiel(
    r1: sp.Expr, r2: sp.Expr, r3: sp.Expr, rf: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Planche 48 : les poids de `V_1`, `V_2` (entree `+`) et `V_3` (entree `-`)."""
    entrees = sp.symbols("V_1 V_2 V_3")
    branches = [_branche(1, 4, r1), _branche(2, 4, r2), _branche(3, 5, r3)]
    branches.append(_branche(5, 6, rf))
    sources = dict(enumerate(entrees, start=1))
    sortie = tensions_a_aop_ideaux(branches, [(4, 5, 6)], sources, 6)[5]
    k1, k2, k3 = (sp.cancel(sp.diff(sortie, tension)) for tension in entrees)
    return k1, k2, k3


def khn(
    resistances: Six, capacites: tuple[sp.Expr, sp.Expr], s: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Planches 47 a 52 : `E_2 / E_1` (passe-bas), `E_3 / E_1`, `E_4 / E_1`.

    `E_3` est un passe-bande inverseur, `E_4` un passe-haut.
    """
    r1, r2, r3, r4, r5, r6 = resistances
    c1, c2 = capacites
    branches = [_branche(1, 2, r3), _branche(2, 6, r4)]
    branches += [_branche(3, 8, r5), _branche(3, 4, r6)]
    branches += [_branche(4, 5, r1), Admittance(5, 6, s * c1)]
    branches += [_branche(6, 7, r2), Admittance(7, 8, s * c2)]
    aops = [(2, 3, 4), (0, 5, 6), (0, 7, 8)]
    tensions = tensions_a_aop_ideaux(branches, aops, _UN, 8)
    return tensions[7], tensions[5], tensions[3]


def tow_thomas(
    resistances: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    capacites: tuple[sp.Expr, sp.Expr],
    s: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """Planche 55 : `E_2 / E_1` (passe-bas) et `E_3 / E_1` (passe-bande).

    L'inverseur de sortie porte deux resistances egales, dont la valeur
    ne compte pas : elles sont prises unite.
    """
    r1, r2, r3, r4 = resistances
    c1, c2 = capacites
    un = sp.Integer(1)
    branches = [_branche(1, 2, r4), _branche(7, 2, r3), _branche(2, 3, r1)]
    branches += [Admittance(2, 3, s * c1), _branche(3, 4, r2)]
    branches += [Admittance(4, 5, s * c2), Admittance(5, 6, un), Admittance(6, 7, un)]
    aops = [(0, 2, 3), (0, 4, 5), (0, 6, 7)]
    tensions = tensions_a_aop_ideaux(branches, aops, _UN, 7)
    return tensions[6], tensions[2]


def fleischer_tow(
    resistances: Six, r: sp.Expr, capacites: tuple[sp.Expr, sp.Expr], s: sp.Expr
) -> sp.Expr:
    """Planche 59 : `E_2 / E_1`, pris a la sortie de l'inverseur.

    L'entree attaque les trois etages, par `R_4`, `R_5` et `R_6` : d'ou
    un numerateur complet, sur le denominateur du Tow-Thomas.
    """
    r1, r2, r3, r4, r5, r6 = resistances
    c1, c2 = capacites
    branches = [_branche(1, 2, r4), _branche(7, 2, r3), _branche(2, 3, r1)]
    branches += [Admittance(2, 3, s * c1), _branche(3, 4, r), _branche(4, 5, r)]
    branches += [_branche(1, 4, r5), _branche(5, 6, r2), _branche(1, 6, r6)]
    branches.append(Admittance(6, 7, s * c2))
    aops = [(0, 2, 3), (0, 4, 5), (0, 6, 7)]
    return tensions_a_aop_ideaux(branches, aops, _UN, 7)[4]
