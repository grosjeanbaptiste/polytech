"""Macros de l'exercice 7 : le champ de deux conducteurs, region par region.

Entierement litteral. Les trois expressions encadrees par le manuscrit
sont publiees, plus ce qu'il ne verifie pas : les deux recollements aux
frontieres, l'egalite de la formule generale avec la superposition
refaite au-dessous, le minimum de l'entrefer et la valeur au centre d'un
conducteur.

Le prefixe est `ChampSept`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    forme_signee,
)
from polytech.electromagnetisme.domain.magnetostatique.regions import (
    champ_a_l_exterieur,
    champ_au_dessous_par_superposition,
    champ_dans_le_conducteur_inferieur,
    champ_dans_le_conducteur_superieur,
    courant_enlace,
)

COURANT = sp.Symbol("I", positive=True)
ECART = sp.Symbol("a", positive=True)
RAYON = sp.Symbol("R", positive=True)
ORDONNEE = sp.Symbol("y", positive=True)
PROFONDEUR = sp.Symbol("h", positive=True)


def _macros_du_courant_enlace() -> dict[str, str]:
    """La regle du courant enlace, et ses deux valeurs remarquables."""
    distance = sp.Symbol("rho", positive=True)
    return {
        "ChampSeptCourantEnlace": expression_latex(
            courant_enlace(COURANT, distance, RAYON),
        ),
        "ChampSeptEnlaceALaFrontiere": expression_latex(
            sp.simplify(courant_enlace(COURANT, RAYON, RAYON)),
        ),
        "ChampSeptEnlaceAuCentre": expression_latex(
            sp.simplify(courant_enlace(COURANT, sp.Integer(0), RAYON)),
        ),
    }


def _macros_des_trois_regions() -> dict[str, str]:
    """Les trois expressions encadrees par le manuscrit."""
    return {
        "ChampSeptExterieur": forme_signee(
            champ_a_l_exterieur(COURANT, ECART, ORDONNEE),
        ),
        "ChampSeptSuperieur": forme_signee(
            champ_dans_le_conducteur_superieur(COURANT, ECART, RAYON, ORDONNEE),
        ),
        "ChampSeptInferieur": forme_signee(
            champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, ORDONNEE),
        ),
    }


def _macros_des_recollements() -> dict[str, str]:
    """Les deux frontieres, chacune vue des deux cotes."""
    haute = ECART - RAYON
    return {
        "ChampSeptFrontiereBasseDedans": forme_signee(
            champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, RAYON),
        ),
        "ChampSeptFrontiereBasseDehors": forme_signee(
            champ_a_l_exterieur(COURANT, ECART, RAYON),
        ),
        "ChampSeptFrontiereHauteDedans": forme_signee(
            champ_dans_le_conducteur_superieur(COURANT, ECART, RAYON, haute),
        ),
        "ChampSeptFrontiereHauteDehors": forme_signee(
            champ_a_l_exterieur(COURANT, ECART, haute),
        ),
    }


def _macros_des_ajouts() -> dict[str, str]:
    """La region au-dessous, le minimum de l'entrefer, le centre."""
    exterieur = champ_a_l_exterieur(COURANT, ECART, ORDONNEE)
    au_centre = sp.limit(
        champ_dans_le_conducteur_inferieur(COURANT, ECART, RAYON, ORDONNEE),
        ORDONNEE,
        0,
        "+",
    )
    return {
        "ChampSeptAuDessousParSuperposition": expression_latex(
            sp.simplify(
                champ_au_dessous_par_superposition(COURANT, ECART, PROFONDEUR),
            ),
        ),
        "ChampSeptAuDessousParLaFormule": expression_latex(
            sp.simplify(champ_a_l_exterieur(COURANT, ECART, -PROFONDEUR)),
        ),
        "ChampSeptMilieu": expression_latex(ECART / 2),
        "ChampSeptChampAuMilieu": forme_signee(
            exterieur.subs(ORDONNEE, ECART / 2),
        ),
        "ChampSeptChampAuCentre": forme_signee(au_centre),
    }


def macros_champ_sept() -> dict[str, str]:
    """Le courant enlace, les trois regions, les recollements et les ajouts."""
    return {
        **_macros_du_courant_enlace(),
        **_macros_des_trois_regions(),
        **_macros_des_recollements(),
        **_macros_des_ajouts(),
    }
