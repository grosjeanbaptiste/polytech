"""Laboratoire 5 d'electronique, questions 6 et 7 : les decalages en sortie.

Deux etages en cascade -- un non-inverseur, puis un inverseur -- chacun
avec sa tension de decalage et ses courants d'entree. Les defauts sont
des sources ajoutees au montage, que l'analyse nodale resout telles
quelles ; l'entree est eteinte, derriere les 600 ohm du generateur.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.application import dimensionnement_lab_05 as dm
from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeCourant,
    SourceDeTension,
)

DECALAGE: Final = 0.5e-3
I_PLUS: Final = 1.5e-12
I_MOINS: Final = 0.5e-12
"""Les valeurs typiques que le corrige officiel tire de la fiche du TLV274."""
GENERATEUR: Final = 600.0
_COURANTS: Final = sp.symbols("I_a I_b")


def _g(resistance: float) -> sp.Expr:
    return 1 / sp.nsimplify(resistance, rational=True)


def decalages(
    *,
    vers_la_masse: float = dm.R_MASSE,
    contre_reaction: float = dm.R_RETOUR,
    courants: bool = True,
) -> tuple[float, float]:
    """La tension continue apres le premier etage, puis apres le second.

    Noeuds : 1 derriere le generateur, 2 l'entree `+` du premier AOP, 3 son
    entree `-`, 4 sa sortie ; 5 l'entree `+` du second, 6 son entree `-`,
    7 sa sortie. Le potentiometre est a son maximum.
    """
    i_plus, i_moins = (sp.nsimplify(i) if courants else 0 for i in (I_PLUS, I_MOINS))
    e0 = sp.nsimplify(DECALAGE)
    circuit: list[Element] = [
        Admittance(1, 0, _g(GENERATEUR)),
        SourceDeTension(2, 1, e0, _COURANTS[0]),
        SourceDeCourant(2, 0, i_plus),
        Admittance(3, 0, _g(vers_la_masse)),
        Admittance(3, 4, _g(contre_reaction)),
        SourceDeCourant(3, 0, i_moins),
        *aop(2, 3, 4),
        SourceDeTension(5, 0, e0, _COURANTS[1]),
        Admittance(4, 6, _g(dm.R_ENTREE)),
        Admittance(6, 7, _g(dm.POTENTIOMETRE)),
        SourceDeCourant(6, 0, i_moins),
        *aop(5, 6, 7),
    ]
    return float(sortie_ideale(circuit, 7, 4)), float(sortie_ideale(circuit, 7, 7))


def decalages_par_les_formules(
    *,
    vers_la_masse: float = dm.R_MASSE,
    contre_reaction: float = dm.R_RETOUR,
    courants: bool = True,
) -> tuple[float, float]:
    """Le second chemin : les formules du cours, sans analyse nodale.

    Non-inverseur : `(1 + R_f / R_m)(e_0 - R_s I_+) + R_f I_-` ; inverseur :
    `-(R_p / R_e) V_1 + (1 + R_p / R_e) e_0 + R_p I_-`.
    """
    i_plus, i_moins = (I_PLUS, I_MOINS) if courants else (0.0, 0.0)
    gain = 1 + contre_reaction / vers_la_masse
    premier = gain * (DECALAGE - GENERATEUR * i_plus) + contre_reaction * i_moins
    rapport = dm.POTENTIOMETRE / dm.R_ENTREE
    second = -rapport * premier + (1 + rapport) * DECALAGE
    return premier, second + dm.POTENTIOMETRE * i_moins
