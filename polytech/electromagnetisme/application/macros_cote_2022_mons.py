"""Macros de l'exercice cote du 23 mai 2022 (Mons). Prefixe `CoteMonsVingtDeux`."""

from __future__ import annotations

import math

from polytech.electromagnetisme.application import cote_2022_mons_couplage as cg
from polytech.electromagnetisme.application import cote_2022_mons_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    trois, quatre = th.courants_de_maille()
    courant, puissance = th.bobine()
    return {
        "CoteMonsVingtDeuxResistance": expression_latex(resistance),
        "CoteMonsVingtDeuxTension": nombre(float(tension), 1),
        "CoteMonsVingtDeuxCourantTrois": nombre(float(trois), 1),
        "CoteMonsVingtDeuxCourantQuatre": nombre(float(quatre), 1),
        "CoteMonsVingtDeuxCourant": expression_latex(courant),
        "CoteMonsVingtDeuxCourantDecimal": nombre(float(courant), 2),
        "CoteMonsVingtDeuxPuissance": nombre(float(puissance), 2),
    }


def _couplage() -> dict[str, str]:
    rapport = cg.rapport_chiffre()
    retard = math.degrees(math.atan2(rapport.real, rapport.imag))
    return {
        "CoteMonsVingtDeuxModule": nombre(abs(rapport) * 100, 2),
        "CoteMonsVingtDeuxDephasage": nombre(cg.dephasage_en_degres(), 2),
        "CoteMonsVingtDeuxRetard": nombre(retard, 2),
    }


def macros_cote_mons_vingt_deux() -> dict[str, str]:
    """Les macros des questions 2 et 3 ; la question 1 est litterale."""
    return {**_thevenin(), **_couplage()}
