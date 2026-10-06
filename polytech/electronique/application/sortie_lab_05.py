"""Laboratoire 5 d'electronique, questions 8 a 12 : l'etage de sortie.

Un AOP alimente entre 0 et 5 V ne sort rien de negatif : on decale le
signal autour d'une tension continue `E`, et deux AOP montes en pont
appliquent au haut-parleur la difference de leurs sorties, ou `E`
disparait.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.application.dimensionnement_lab_05 import HAUT_PARLEUR
from polytech.electronique.domain.aop_ideal import tensions_a_aop_ideaux
from polytech.electronique.domain.netliste import Admittance

R4: Final = 22e3
R5: Final = 22e3
C2: Final = 330e-9
ALIMENTATION: Final = 5.0
BASSE_FREQUENCE: Final = 20.0
_E, _VI = sp.symbols("E V_i")


def _etage(p: sp.Expr, tension: sp.Expr, signal: sp.Expr) -> sp.Expr:
    """La sortie : `E` sur l'entree `+`, `V_i` par `C_2` et `R_4` sur l'entree `-`.

    Noeuds : 1 le signal, 2 entre `C_2` et `R_4`, 3 l'entree `-`, 4 la
    sortie, 5 l'entree `+`.
    """
    r4, r5, c2 = (sp.nsimplify(v) for v in (R4, R5, C2))
    branches = [Admittance(1, 2, p * c2), Admittance(2, 3, 1 / r4)]
    branches.append(Admittance(3, 4, 1 / r5))
    sources = {1: signal, 5: tension}
    return tensions_a_aop_ideaux(branches, [(5, 3, 4)], sources, 5)[3]


def transferts(p: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """L'effet de `E` seule, puis celui de `V_i` seul, sur la sortie."""
    sortie = _etage(p, _E, _VI)
    return sp.cancel(sp.diff(sortie, _E)), sp.cancel(sp.diff(sortie, _VI))


def pulsation_de_coupure() -> float:
    """`1 / (R_4 C_2)` : le passe-haut que `C_2` forme avec `R_4`."""
    return 1 / (R4 * C2)


def frequence_de_coupure() -> float:
    """La meme en hertz."""
    return pulsation_de_coupure() / (2 * math.pi)


def condensateur_de_liaison(resistance: float = HAUT_PARLEUR) -> float:
    """`1 / (2 pi R f)` : ce qu'il faudrait en serie avec la charge, a 20 Hz."""
    return 1 / (2 * math.pi * resistance * BASSE_FREQUENCE)


def _pont(p: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`V_o1` et `V_o2` du pont, a la frequence complexe `p`.

    Le second AOP recopie `V_o1` inverse autour de `E`, par deux
    resistances egales. Noeuds : 1 le signal, 2 entre `C_2` et `R_4`, 3 et
    4 l'entree `-` et la sortie du premier AOP, 5 et 6 celles du second,
    7 la tension `E`.
    """
    r4, r5, c2 = (sp.nsimplify(v) for v in (R4, R5, C2))
    un = sp.Integer(1)
    branches = [Admittance(1, 2, p * c2), Admittance(2, 3, 1 / r4)]
    branches += [Admittance(3, 4, 1 / r5), Admittance(4, 5, un), Admittance(5, 6, un)]
    aops = [(7, 3, 4), (7, 5, 6)]
    tensions = tensions_a_aop_ideaux(branches, aops, {1: _VI, 7: _E}, 7)
    return tensions[3], tensions[5]


def sorties_du_pont(tension: sp.Expr, signal: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`V_o1` et `V_o2` pour `E` continue et `V_i` dans la bande passante.

    Les deux sources n'ont pas la meme frequence : `E` voit `C_2` ouvert,
    `V_i` le voit en court-circuit. Chaque sortie est donc la somme de
    l'effet de `E` en continu et de celui de `V_i` a frequence infinie.
    """
    p = sp.Symbol("p")
    resultats = []
    for sortie in _pont(p):
        en_continu = sp.diff(sortie, _E).subs(p, 0)
        en_alternatif = sp.limit(sp.diff(sortie, _VI), p, sp.oo)
        resultats.append(en_continu * tension + en_alternatif * signal)
    return resultats[0], resultats[1]


def crete_admissible_a_l_entree() -> float:
    """Avec `E` a mi-alimentation, chaque sortie a `E` de marge de chaque cote."""
    return ALIMENTATION / 2


def crete_maximale_sur_la_charge() -> float:
    """Le pont double la tension : deux fois la crete admissible."""
    return 2 * crete_admissible_a_l_entree()


def puissance_maximale_sur_la_charge() -> float:
    """Ce qu'une sinusoide de cette crete dissipe dans le haut-parleur."""
    return crete_maximale_sur_la_charge() ** 2 / (2 * HAUT_PARLEUR)
