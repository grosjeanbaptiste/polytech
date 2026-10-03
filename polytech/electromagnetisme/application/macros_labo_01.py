"""Macros du laboratoire I : precisions, valeurs efficaces, sonde, filtres.

Comme pour les deux autres labos, chaque mesure est publiee avec sa
valeur attendue. La table volt-amperemetrique porte ses quatre lignes,
pour montrer d'ou vient l'incertitude ; la sonde, sa coupure avant et sa
capacite de compensation apres ; le passe-bande, ses deux coupures.

Le prefixe est `LaboUn`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_01 import (
    C_RLC,
    L_RLC,
    R_RLC,
    capacite_de_compensation,
    coupure_de_la_sonde,
    coupures_du_passe_bande,
    mesure_volt_amperemetrique,
    precision_de_l_ohmmetre,
    resonance_du_passe_bande,
)
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    facteur_de_qualite,
)
from polytech.electromagnetisme.domain.circuits.valeurs_efficaces import (
    carre,
    sinus,
    triangle,
    valeur_efficace,
)
from polytech.electromagnetisme.domain.instruments import precision_d_une_lecture

TENSIONS = (4, 6, 8, 10)
NOMS = ("Quatre", "Six", "Huit", "Dix")
R_FILTRE, C_FILTRE = 2200, 1e-8


def _ligne(tension: int, nom: str) -> dict[str, str]:
    """Une ligne de la table : U, I, leurs precisions, R et la sienne."""
    courant = tension / 2.2
    resistance, incertitude = mesure_volt_amperemetrique(float(tension))
    return {
        f"LaboUnCourant{nom}": nombre(courant, 3),
        f"LaboUnPrecisionU{nom}": nombre(
            precision_d_une_lecture(tension, 10.0, 0.0015, 0.0004) * 1e6, 0
        ),
        f"LaboUnPrecisionI{nom}": nombre(
            precision_d_une_lecture(courant, 10.0, 0.005, 0.010) * 1e3, 2
        ),
        f"LaboUnResistance{nom}": nombre(resistance, 0),
        f"LaboUnPrecisionR{nom}": nombre(incertitude, 2),
    }


def _macros_des_mesures() -> dict[str, str]:
    """L'ohmmetre, la table, les trois valeurs efficaces."""
    amplitude, periode = sp.Integer(1), sp.Rational(1, 1000)
    lignes: dict[str, str] = {}
    for tension, nom in zip(TENSIONS, NOMS, strict=True):
        lignes |= _ligne(tension, nom)
    return {
        "LaboUnOhmmetre": nombre(precision_de_l_ohmmetre(), 3),
        "LaboUnPartLecture": nombre(2200 * 0.0020 / 100, 3),
        "LaboUnPartGamme": nombre(10_000 * 0.0005 / 100, 3),
        **lignes,
        "LaboUnEfficaceSinus": nombre(
            float(valeur_efficace(sinus(amplitude, periode), periode)), 4
        ),
        "LaboUnEfficaceCarre": nombre(
            float(valeur_efficace(carre(amplitude, periode), periode)), 4
        ),
        "LaboUnEfficaceTriangle": nombre(
            float(valeur_efficace(triangle(amplitude, periode), periode)), 4
        ),
    }


def _macros_des_filtres() -> dict[str, str]:
    """La sonde, le R-C et le R-L-C."""
    basse, haute = coupures_du_passe_bande()
    return {
        "LaboUnCoupureSonde": nombre(coupure_de_la_sonde(), 0),
        "LaboUnCoupureSondeKilo": nombre(coupure_de_la_sonde() / 1e3, 2),
        "LaboUnCompensation": nombre(float(capacite_de_compensation()) * 1e12, 3),
        "LaboUnCoupureRC": nombre(1 / (2 * math.pi * R_FILTRE * C_FILTRE), 0),
        "LaboUnResonance": nombre(resonance_du_passe_bande(), 0),
        "LaboUnQualite": nombre(float(facteur_de_qualite(R_RLC, L_RLC, C_RLC)), 3),
        "LaboUnCoupureBasse": nombre(basse, 0),
        "LaboUnCoupureHaute": nombre(haute, 0),
        "LaboUnBande": nombre(haute - basse, 0),
    }


def macros_du_labo_un() -> dict[str, str]:
    """Les cinq parties du laboratoire."""
    return {**_macros_des_mesures(), **_macros_des_filtres()}
