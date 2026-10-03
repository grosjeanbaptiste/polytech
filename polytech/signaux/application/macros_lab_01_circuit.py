"""Macros du TP 1, question 4 : le circuit L-C serie ferme sur R.

La transmittance est etablie en litteral -- c'est ce que l'enonce
demande -- puis evaluee. Les deux bouts de la reponse impulsionnelle se
lisent sans l'inverser : `h(0+) = lim p H(p) = R/L`, et `h(oo) = 0`.

La resistance de la question 4f est un choix, nomme dans l'enonce du
module de donnees ; ses deux constantes de temps sont publiees pour
qu'on les lise sur la seconde figure.

Le prefixe est `TpQuatre`.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from polytech.signaux.application import enonce_lab_01 as e
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.modeles import (
    coefficients,
    resistance_critique,
    transmittance_lc_serie_sur_r,
)
from polytech.signaux.domain.poles_numeriques import (
    constante_de_temps,
    frequence_d_oscillation,
    poles_et_zeros,
    reponse_impulsionnelle,
    reponse_indicielle,
)

INSTANTS = np.linspace(0.0, 0.01, 201)
"""Dix millisecondes : cinq constantes de temps du circuit de l'enonce."""


def _courbes(resistance: float, suffixe: str) -> dict[str, str]:
    """Les reponses impulsionnelle et indicielle, temps en millisecondes."""
    num, den = coefficients(
        transmittance_lc_serie_sur_r(resistance, e.QUATRE_L, e.QUATRE_C)
    )
    millisecondes = INSTANTS * 1e3
    return {
        f"TpQuatreImpulsion{suffixe}": coordonnees(
            millisecondes, reponse_impulsionnelle(num, den, INSTANTS), 3
        ),
        f"TpQuatreIndice{suffixe}": coordonnees(
            millisecondes, reponse_indicielle(num, den, INSTANTS), 4
        ),
    }


def _macros_de_l_enonce() -> dict[str, str]:
    """La transmittance litterale, ses poles, sa constante de temps."""
    r, inductance, c = sp.symbols("R L C", positive=True)
    num, den = coefficients(
        transmittance_lc_serie_sur_r(e.QUATRE_R, e.QUATRE_L, e.QUATRE_C)
    )
    poles, _, gain = poles_et_zeros(num, den)
    pole = poles[np.argmax(poles.imag)]
    return {
        "TpQuatreTransmittance": sp.latex(
            transmittance_lc_serie_sur_r(r, inductance, c).expression
        ),
        "TpQuatreL": nombre(e.QUATRE_L, 0),
        "TpQuatreC": f"{e.QUATRE_C:.0e}".replace("e-0", "e-"),
        "TpQuatreR": nombre(e.QUATRE_R, 0),
        "TpQuatreGain": nombre(gain, 0),
        "TpQuatrePoleReel": nombre(pole.real, 1),
        "TpQuatrePoleImag": nombre(pole.imag, 1),
        "TpQuatreTau": nombre(constante_de_temps(pole) * 1e3, 1),
        "TpQuatreFrequence": nombre(frequence_d_oscillation(pole), 1),
        "TpQuatrePeriode": nombre(1e3 / frequence_d_oscillation(pole), 3),
        "TpQuatreHZero": nombre(e.QUATRE_R / e.QUATRE_L, 0),
    }


def _macros_du_choix() -> dict[str, str]:
    """La resistance critique, la resistance choisie, ses deux poles reels."""
    num, den = coefficients(
        transmittance_lc_serie_sur_r(e.QUATRE_R_CHOISIE, e.QUATRE_L, e.QUATRE_C)
    )
    rapide, lent = sorted(poles_et_zeros(num, den)[0].real)
    return {
        "TpQuatreRCritique": nombre(
            float(resistance_critique(e.QUATRE_L, e.QUATRE_C)), 0
        ),
        "TpQuatreRChoisie": nombre(e.QUATRE_R_CHOISIE / 1e3, 0),
        "TpQuatrePoleRapide": nombre(rapide, 0),
        "TpQuatrePoleLent": nombre(lent, 0),
        "TpQuatreTauRapide": nombre(-1e3 / rapide, 3),
        "TpQuatreTauLent": nombre(-1e3 / lent, 3),
    }


def macros_du_circuit() -> dict[str, str]:
    """La question 4, de l'enonce au choix de la resistance."""
    return {
        **_macros_de_l_enonce(),
        **_macros_du_choix(),
        **_courbes(e.QUATRE_R, "Enonce"),
        **_courbes(e.QUATRE_R_CHOISIE, "Choisie"),
    }
