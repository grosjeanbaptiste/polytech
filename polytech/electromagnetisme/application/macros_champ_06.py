"""Macros de l'exercice 6 de la seance 2.1 : deux conducteurs, champ et force.

Entierement litteral. Deux macros publient le champ avec des courants de
sens **oppose** puis de **meme** sens : sans ce pendant, rien ne
distinguerait une somme d'une difference dans le document, et les deux
donneraient une expression non nulle.

Le prefixe est `ChampSix`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    forme_signee,
)
from polytech.electromagnetisme.domain.magnetostatique.mediatrice import (
    champ_si_les_courants_sont_de_meme_sens,
    champ_sur_la_mediatrice,
    force_lineique_en_un_point,
    produit_scalaire,
)

COURANT = sp.Symbol("I", positive=True)
COURANT_DU_TROISIEME = sp.Symbol("I_2", positive=True)
DEMI_ECART = sp.Symbol("a", positive=True)
ORDONNEE = sp.Symbol("b", positive=True)
PERMEABILITE = sp.Symbol("mu_0", positive=True)


def _champ() -> tuple[sp.Expr, sp.Expr]:
    """Le champ en `P`, calcule une fois pour les quatre macros."""
    return champ_sur_la_mediatrice(COURANT, DEMI_ECART, ORDONNEE)


def _force() -> tuple[sp.Expr, sp.Expr]:
    """La force lineique sur le troisieme conducteur place en `P`."""
    return force_lineique_en_un_point(
        COURANT,
        DEMI_ECART,
        ORDONNEE,
        COURANT_DU_TROISIEME,
        PERMEABILITE,
    )


def _macros_du_champ() -> dict[str, str]:
    """Les deux composantes, la distance et le cosinus de la projection."""
    champ_x, champ_y = _champ()
    distance = sp.sqrt(DEMI_ECART**2 + ORDONNEE**2)
    return {
        "ChampSixDistance": expression_latex(distance),
        "ChampSixCosinus": expression_latex(DEMI_ECART / distance),
        "ChampSixModuleDUnFil": expression_latex(
            COURANT / (2 * sp.pi * distance),
        ),
        "ChampSixChampX": expression_latex(sp.simplify(champ_x)),
        "ChampSixChampY": forme_signee(champ_y),
    }


def _macros_de_la_force() -> dict[str, str]:
    """La force, et le produit scalaire qui atteste l'orthogonalite."""
    force_x, force_y = _force()
    return {
        "ChampSixForceX": forme_signee(force_x),
        "ChampSixForceY": expression_latex(sp.simplify(force_y)),
        "ChampSixProduitScalaire": expression_latex(
            sp.simplify(produit_scalaire(_force(), _champ())),
        ),
    }


def _macros_des_ajouts() -> dict[str, str]:
    """L'optimum, l'asymptote dipolaire, et le pendant de meme sens."""
    champ_y = _champ()[1]
    meme_sens = champ_si_les_courants_sont_de_meme_sens(
        COURANT,
        DEMI_ECART,
        ORDONNEE,
    )
    return {
        "ChampSixEcartOptimal": expression_latex(ORDONNEE),
        "ChampSixChampOptimal": forme_signee(
            champ_y.subs(DEMI_ECART, ORDONNEE),
        ),
        "ChampSixCoefficientDipolaire": expression_latex(
            sp.limit(champ_y * ORDONNEE**2, ORDONNEE, sp.oo),
        ),
        "ChampSixMemeSensX": forme_signee(meme_sens[0]),
        "ChampSixMemeSensY": expression_latex(sp.simplify(meme_sens[1])),
    }


def macros_champ_six() -> dict[str, str]:
    """Le champ, la force, et les trois ajouts au corrige officiel."""
    return {
        **_macros_du_champ(),
        **_macros_de_la_force(),
        **_macros_des_ajouts(),
    }
