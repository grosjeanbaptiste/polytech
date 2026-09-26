"""Macros du domaine de validite des resolutions de la partie II.

Le corrige divisait par ``u`` et prenait des logarithmes sans dire ou ces
operations sont licites. Ces macros publient les lieux ou l'equation
degenere, et le residu obtenu en reportant la solution close dans son
equation -- la verification la plus directe qui soit.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.domain.cinematique.validite import (
    lieu_singulier,
    residu_de_la_fonction_de_courant,
    residu_des_lignes_de_courant,
    residu_du_potentiel,
)


def macros_validite() -> dict[str, str]:
    """Rend les domaines de validite et les residus de verification."""
    champs = {"B": enonce.CHAMP_B, "C": enonce.CHAMP_C}
    macros: dict[str, str] = {}
    for suffixe, champ in champs.items():
        macros[f"LieuSingulier{suffixe}"] = expression_latex(lieu_singulier(champ.u))
        macros[f"ResiduLignesCourant{suffixe}"] = expression_latex(
            residu_des_lignes_de_courant(champ),
        )
    champ = enonce.CHAMP_C
    macros["ResiduPotentielC"] = expression_latex(
        sp.Matrix(residu_du_potentiel(champ)).T,
    )
    macros["ResiduFonctionCourantC"] = expression_latex(
        sp.Matrix(residu_de_la_fonction_de_courant(champ)).T,
    )
    return macros
