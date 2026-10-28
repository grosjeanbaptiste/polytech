"""La tension de decalage en sortie : la somme des defauts, un montage a la fois.

Chaque defaut est une source ajoutee au montage ; le principe de
superposition donne la sortie comme la somme des contributions que
`defauts_aop` et `amplification` calculent une a une.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.noyau import core


@dataclass(frozen=True, slots=True)
class Defauts:
    """Les trois sources de defaut d'un AOP : `e_0`, `I_+` et `I_-`."""

    tension_de_decalage: float
    courant_plus: float
    courant_moins: float


def decalage_du_non_inverseur(
    defauts: Defauts,
    *,
    vers_la_masse: float,
    contre_reaction: float,
    source: float,
) -> float:
    """`G (e_0 - R_s I_+) + R_f I_-`, `G = 1 + R_f / R_m`, en volts."""
    return core.decalage_du_non_inverseur(
        defauts.tension_de_decalage,
        defauts.courant_plus,
        defauts.courant_moins,
        vers_la_masse,
        contre_reaction,
        source,
    )


def decalage_de_l_inverseur(
    entree: float,
    defauts: Defauts,
    *,
    resistance_entree: float,
    contre_reaction: float,
) -> float:
    """`-(R_f / R_e) V_e + (1 + R_f / R_e) e_0 + R_f I_-`, en volts."""
    return core.decalage_de_l_inverseur(
        entree,
        defauts.tension_de_decalage,
        defauts.courant_plus,
        defauts.courant_moins,
        resistance_entree,
        contre_reaction,
    )
