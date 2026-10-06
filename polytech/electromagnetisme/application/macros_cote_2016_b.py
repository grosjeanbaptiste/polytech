"""Macros de l'exercice cote 2016, questions 4 a 6. Prefixe `CoteSeize`."""

from __future__ import annotations

import cmath
import math

from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.application import cote_2016_magnetique as ma
from polytech.electromagnetisme.application import cote_2016_sinusoidal as si
from polytech.electromagnetisme.application.formatage import nombre, scientifique


def _degres(phaseur: complex) -> float:
    return math.degrees(cmath.phase(phaseur))


def _sinusoidal() -> dict[str, str]:
    i1, i2 = si.courants_de_mailles()
    tension, courant = si.tension_de_la_branche(), si.courant_de_la_branche()
    puissance = si.puissance_de_la_branche()
    return {
        "CoteSeizeReactanceDeCDeux": nombre(1 / (en.PULSATION * en.C2), 1),
        "CoteSeizeReactanceDeLTrois": nombre(en.PULSATION * en.L3, 4),
        "CoteSeizeCourantUnReel": nombre(i1.real, 4),
        "CoteSeizeCourantUnImaginaire": nombre(-i1.imag, 4),
        "CoteSeizeCourantDeuxReel": nombre(-i2.real, 4),
        "CoteSeizeCourantDeuxImaginaire": nombre(i2.imag, 4),
        "CoteSeizeTensionReelle": nombre(tension.real, 2),
        "CoteSeizeTensionImaginaire": nombre(tension.imag, 2),
        "CoteSeizeTensionModule": nombre(abs(tension), 2),
        "CoteSeizeTensionArgument": nombre(_degres(tension), 3),
        "CoteSeizeCourantModule": nombre(abs(courant), 3),
        "CoteSeizeCourantArgument": nombre(_degres(courant), 3),
        "CoteSeizePuissanceActive": nombre(puissance.real, 1),
        "CoteSeizePuissanceReactive": nombre(puissance.imag, 1),
        "CoteSeizeDephasage": nombre(si.dephasage_du_courant(), 3),
        "CoteSeizeFacteurDePuissance": nombre(
            math.cos(math.radians(si.dephasage_du_courant())), 4
        ),
    }


def _ouverture() -> dict[str, str]:
    return {
        "CoteSeizeConstanteDeTemps": nombre(en.R * en.C * 1e6, 2),
        "CoteSeizeCourantAPhaseNulle": nombre(
            si.courant_apres_ouverture(0.0, phase=0.0), 2
        ),
    }


def _demi_tores() -> dict[str, str]:
    circuit = ma.circuit_ecarte()
    champ, inductance = ma.demi_tores_jointifs()
    return {
        "CoteSeizeInductionDuTore": nombre(en.INDUCTION_LUE_A_50, 2),
        "CoteSeizeFluxDuTore": nombre(ma.flux_du_tore() * 1e4, 3),
        "CoteSeizeSpiresDuTore": nombre(ma.spires_du_tore(), 0),
        "CoteSeizeChampDEntrefer": scientifique(ma.champ_d_entrefer(), 4),
        "CoteSeizeLongueurDuFer": nombre(circuit.longueur_du_fer * 1e2, 2),
        "CoteSeizeAmperesToursDuFer": nombre(circuit.amperes_tours_du_fer(), 2),
        "CoteSeizeAmperesToursDeLAir": nombre(circuit.amperes_tours_des_entrefers(), 1),
        "CoteSeizeEcartement": nombre(ma.ecartement() * 1e3, 4),
        "CoteSeizePartDesEntrefers": nombre(circuit.part_des_entrefers() * 100, 1),
        "CoteSeizeChampJointif": nombre(champ, 1),
        "CoteSeizeInductanceJointive": nombre(inductance, 3),
        "CoteSeizeInductanceVoulue": nombre(en.INDUCTANCE_VOULUE, 4),
    }


def macros_cote_seize_fin() -> dict[str, str]:
    """Les macros des questions 4 a 6."""
    return {**_sinusoidal(), **_ouverture(), **_demi_tores()}
