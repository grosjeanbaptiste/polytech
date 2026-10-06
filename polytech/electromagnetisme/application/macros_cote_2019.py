"""Macros de l'exercice cote du 16 mai 2019. Prefixe `CoteDixNeuf`."""

from __future__ import annotations

import math

from polytech.electromagnetisme.application import cote_2019_fermeture as fe
from polytech.electromagnetisme.application import cote_2019_fils as fi
from polytech.electromagnetisme.application import cote_2019_noyau as ny
from polytech.electromagnetisme.application import cote_2019_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)

HAUTEUR_ARRONDIE = 4.5
"""`h2`, en metres, tel que le manuscrit le reporte a la seconde question."""


def _fils() -> dict[str, str]:
    exposant = 2 * math.pi * fi.PERMITTIVITE_DU_MANUSCRIT * fi.COEFFICIENT_VOULU
    arrondi = fi.champ_en_p(HAUTEUR_ARRONDIE)[1]
    exact = fi.champ_en_p(fi.hauteur())[1]
    return {
        "CoteDixNeufExposant": nombre(exposant, 3),
        "CoteDixNeufRapport": nombre(math.exp(exposant), 2),
        "CoteDixNeufHauteur": nombre(fi.hauteur(), 4),
        "CoteDixNeufChamp": nombre(arrondi, 2),
        "CoteDixNeufChampExact": nombre(exact, 0),
        "CoteDixNeufEcartDuChamp": nombre((arrondi - exact) / arrondi * 100, 1),
    }


def _noyau() -> dict[str, str]:
    etat = ny.etat()
    centre, cote, air = ny.amperes_tours()
    centrale, laterale = ny.longueurs()
    spires = ny.spires()
    return {
        "CoteDixNeufInduction": nombre(etat.induction, 4),
        "CoteDixNeufFluxLateral": nombre(etat.flux_lateral * 1e4, 2),
        "CoteDixNeufFluxCentral": nombre(etat.flux_central * 1e4, 2),
        "CoteDixNeufLongueurCentrale": nombre(centrale * 1e2, 1),
        "CoteDixNeufLongueurLaterale": nombre(laterale * 1e2, 1),
        "CoteDixNeufAmperesToursDuFer": nombre(centre + cote, 2),
        "CoteDixNeufAmperesToursDeLAir": nombre(air, 0),
        "CoteDixNeufAmperesTours": nombre(centre + cote + air, 2),
        "CoteDixNeufSpiresDuManuscrit": nombre(ny.spires(flux_central=7.04e-4), 1),
        "CoteDixNeufSpires": nombre(spires, 1),
        "CoteDixNeufCourant": nombre((centre + cote + air) / spires, 3),
        "CoteDixNeufEcartDeLecture": nombre(
            ny.spires(champ=170.0) - ny.spires(champ=160.0), 1
        ),
    }


def _circuits() -> dict[str, str]:
    tension, resistance = th.thevenin()
    _, second = th.courants_de_mailles()
    _, en_regime = fe.regime()
    _, pente = fe.depart_du_courant()
    return {
        "CoteDixNeufTension": expression_latex(tension),
        "CoteDixNeufResistance": expression_latex(resistance),
        "CoteDixNeufCourantDeMaille": expression_latex(second),
        "CoteDixNeufCourtCircuit": expression_latex(th.courant_de_court_circuit()),
        "CoteDixNeufTensionEnRegime": expression_latex(en_regime),
        "CoteDixNeufPente": expression_latex(pente),
    }


def macros_cote_dix_neuf() -> dict[str, str]:
    """Les macros des cinq questions."""
    return {**_fils(), **_noyau(), **_circuits()}
