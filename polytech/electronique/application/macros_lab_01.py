"""Macros du laboratoire 1 d'electronique : filtrage.

Le prefixe est `LabUn`. Les courbes portent un nom : un nom de macro
n'a pas de chiffre.
"""

from __future__ import annotations

import math

import numpy as np

from polytech.electronique.application import bode_lab_01 as bo
from polytech.electronique.application import courbes_lab_01 as co
from polytech.electronique.application import rc_lab_01 as rc
from polytech.electronique.application import rlc_lab_01 as rl
from polytech.electronique.application.formatage import coordonnees, nombre

BOBINE = float(round(rl.resistance_parasite(28.0, 0.0)))
"""La resistance serie que le releve sans resistance revele, arrondie a l'ohm."""


def _ligne(points: list[tuple[float, float]]) -> str:
    abscisses, ordonnees = zip(*points, strict=True)
    return coordonnees(np.array(abscisses), np.array(ordonnees), 2)


def _passe_haut() -> dict[str, str]:
    plateau, coupure = rc.effet_du_generateur()
    return {
        "LabUnCoupure": nombre(rc.frequence_de_coupure(), 1),
        "LabUnPlateau": nombre(20 * math.log10(plateau), 2),
        "LabUnCoupureGenerateur": nombre(coupure, 1),
        "LabUnCourbeHaut": coordonnees(*co.passe_haut(), 2),
        "LabUnAsymptoteHaut": _ligne(co.asymptote_du_passe_haut()),
    }


def _passe_bas() -> dict[str, str]:
    lent, rapide = rl.poles(10e3)
    sans = rl.module_a_la_resonance(1e3, generateur=False)
    avec = rl.module_a_la_resonance(10e3, generateur=False)
    return {
        "LabUnPulsationResonance": nombre(rl.pulsation_de_resonance() / 1e5, 3),
        "LabUnResonance": nombre(rl.frequence_de_resonance() / 1e3, 2),
        "LabUnCaracteristique": nombre(rl.impedance_caracteristique(), 0),
        "LabUnModuleUnAvec": nombre(rl.module_a_la_resonance(1e3, generateur=True), 2),
        "LabUnModuleDixAvec": nombre(
            rl.module_a_la_resonance(10e3, generateur=True), 2
        ),
        "LabUnModuleUnSans": nombre(sans, 2),
        "LabUnModuleDixSans": nombre(avec, 3),
        "LabUnDecibelsUn": nombre(20 * math.log10(sans), 2),
        "LabUnDecibelsDix": nombre(20 * math.log10(avec), 2),
        "LabUnCritique": nombre(rl.resistance_critique(), 0),
        "LabUnPoleLent": nombre(-lent / (2 * math.pi) / 1e3, 1),
        "LabUnPoleRapide": nombre(-rapide / (2 * math.pi) / 1e3, 0),
        "LabUnParasiteSans": nombre(rl.resistance_parasite(28.0, 0.0), 0),
        "LabUnParasiteAvec": nombre(rl.resistance_parasite(8.0, 1e3), 0),
        "LabUnMesureeUn": nombre(100 * rl.tension_mesuree_a_la_resonance(1e3), 1),
        "LabUnMesureeBobine": nombre(
            100 * rl.tension_mesuree_a_la_resonance(0.0, parasite=BOBINE), 0
        ),
        "LabUnCourbeBobine": coordonnees(*co.passe_bas(BOBINE), 2),
        "LabUnCourbeKilo": coordonnees(*co.passe_bas(1e3), 2),
        "LabUnCourbeDix": coordonnees(*co.passe_bas(10e3), 2),
    }


def _bode() -> dict[str, str]:
    premiere, seconde = co.asymptotes_de_la_question_7()
    _, _, troisieme = bo.sorties()
    return {
        "LabUnCreux": nombre(bo.gain_exact(bo.SECONDE, 1e3), 2),
        "LabUnCoin": nombre(bo.gain_exact(bo.PREMIERE, 1e3), 2),
        "LabUnPhaseMilieu": nombre(bo.phase_exacte(bo.PREMIERE, math.sqrt(1e9)), 1),
        "LabUnCourbePremiere": coordonnees(*co.bode(bo.PREMIERE), 2),
        "LabUnCourbeSeconde": coordonnees(*co.bode(bo.SECONDE), 2),
        "LabUnAsymptotePremiere": _ligne(premiere),
        "LabUnAsymptoteSeconde": _ligne(seconde),
        "LabUnPhaseTroisieme": nombre(troisieme[1], 0),
        "LabUnOfficielTroisieme": nombre(bo.sortie_de_l_officiel()[0] * 1e3, 0),
    }


def macros_du_labo_un() -> dict[str, str]:
    """Toutes les valeurs que le corrige du laboratoire 1 compose."""
    return {**_passe_haut(), **_passe_bas(), **_bode()}
