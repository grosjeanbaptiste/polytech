"""Macros du laboratoire II : ce que l'oscilloscope doit montrer.

Le labo est un labo de mesures ; chaque mesure a pourtant une valeur
attendue, et c'est elle qu'on publie : la constante de temps, la periode
et la duree du signal carre, le temps de montee, deux points de la
decharge, et en sinusoidal les amplitudes, les phases et les decalages
temporels qui se lisent a l'ecran.

Le prefixe est `LaboDeux`.
"""

from __future__ import annotations

import cmath
import math

import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_02 import (
    CAPACITE,
    FREQUENCE,
    LECTURES_OFFICIELLES,
    RESISTANCE_SINUS,
    charge_du_condensateur,
    constante_de_temps,
    phaseurs_du_sinus,
    reponse_a_l_impulsion,
)
from polytech.electromagnetisme.application.formatage import expression_latex, nombre
from polytech.electromagnetisme.domain.circuits.premier_ordre import (
    constante_par_deux_points,
    temps_de_montee,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

PERIODES_PAR_TAU = 20
"""La periode du carre que prescrit l'enonce : `T = 20 tau`."""
CRETE = 5.0
"""Le carre va de zero a cinq volts."""


def _macros_de_la_charge() -> dict[str, str]:
    """La reponse indicielle, l'impulsion, le carre, la montee."""
    v0, r, c = sp.symbols("v_0 R C", positive=True)
    tau = sp.Symbol("tau", positive=True)
    seconde = constante_de_temps()
    periode = PERIODES_PAR_TAU * seconde
    return {
        "LaboDeuxCharge": expression_latex(charge_du_condensateur(v0, r, c)),
        "LaboDeuxFinImpulsion": nombre(
            float(reponse_a_l_impulsion(tau, 5 * tau).subs(TEMPS, 5 * tau)), 4
        ),
        "LaboDeuxTau": nombre(seconde * 1e6, 0),
        "LaboDeuxPeriode": nombre(periode * 1e3, 0),
        "LaboDeuxDuree": nombre(periode / 2 * 1e3, 0),
        "LaboDeuxFrequenceCarre": nombre(1 / periode, 0),
        "LaboDeuxLnNeuf": nombre(math.log(9), 3),
        "LaboDeuxAUnTau": nombre(1 - math.exp(-1), 3),
        "LaboDeuxMontee": nombre(float(temps_de_montee(seconde)) * 1e6, 1),
    }


def _macros_des_deux_points() -> dict[str, str]:
    """Deux lectures attendues sur la decharge, une et deux constantes de temps."""
    seconde = constante_de_temps()
    premiere, seconde_tension = CRETE * math.exp(-1), CRETE * math.exp(-2)
    retrouvee = constante_par_deux_points(
        seconde, premiere, 2 * seconde, seconde_tension
    )
    return {
        "LaboDeuxCrete": nombre(CRETE, 0),
        "LaboDeuxPointUn": nombre(premiere, 3),
        "LaboDeuxPointDeux": nombre(seconde_tension, 3),
        "LaboDeuxTauRetrouve": nombre(retrouvee * 1e6, 0),
    }


def _macros_du_sinus() -> dict[str, str]:
    """Amplitudes, phases et decalages temporels des deux tensions."""
    resistance, condensateur = phaseurs_du_sinus()
    phase_r = math.degrees(cmath.phase(resistance))
    phase_c = math.degrees(cmath.phase(condensateur))
    return {
        "LaboDeuxRSinus": nombre(RESISTANCE_SINUS / 1e3, 1),
        "LaboDeuxCapacite": nombre(CAPACITE * 1e6, 2),
        "LaboDeuxFrequence": nombre(FREQUENCE / 1e3, 0),
        "LaboDeuxProduit": nombre(
            2 * math.pi * FREQUENCE * RESISTANCE_SINUS * CAPACITE, 4
        ),
        "LaboDeuxAmplitudeC": nombre(abs(condensateur), 4),
        "LaboDeuxPhaseC": nombre(phase_c, 2),
        "LaboDeuxAmplitudeR": nombre(abs(resistance), 4),
        "LaboDeuxPhaseR": nombre(phase_r, 2),
        "LaboDeuxDecalageC": nombre(phase_c / 360 / FREQUENCE * 1e6, 1),
        "LaboDeuxDecalageR": nombre(phase_r / 360 / FREQUENCE * 1e6, 1),
    }


def _macros_des_lectures_officielles() -> dict[str, str]:
    """Ce que les mesures du corrige officiel donnent, recalcule."""
    lu = LECTURES_OFFICIELLES
    par_points = constante_par_deux_points(
        lu["premier_instant"],
        lu["premiere_tension"],
        lu["second_instant"],
        lu["seconde_tension"],
    )
    return {
        "LaboDeuxOfficielPoints": nombre(par_points * 1e6, 1),
        "LaboDeuxOfficielMontee": nombre(lu["montee"] * 1e6, 0),
        "LaboDeuxOfficielDescente": nombre(lu["descente"] * 1e6, 0),
        "LaboDeuxOfficielParMontee": nombre(lu["montee"] / math.log(9) * 1e6, 1),
        "LaboDeuxOfficielParDescente": nombre(lu["descente"] / math.log(9) * 1e6, 1),
        "LaboDeuxOfficielPhaseC": nombre(-360 * FREQUENCE * lu["decalage_c"], 2),
        "LaboDeuxOfficielPhaseR": nombre(360 * FREQUENCE * lu["decalage_r"], 2),
    }


def macros_du_labo_deux() -> dict[str, str]:
    """Les trois parties du laboratoire."""
    return {
        **_macros_de_la_charge(),
        **_macros_des_deux_points(),
        **_macros_du_sinus(),
        **_macros_des_lectures_officielles(),
    }
