"""Macros du TP 1, questions 5 et 6 : poles, stabilite, poles dominants.

Question 5 : l'enonce dit le systeme « stabilise ». Ses poles disent le
contraire, et le critere de Routh le dit par un second chemin, sans
calculer une seule racine.

Question 6 : deux approximations de la reponse impulsionnelle, publiees
cote a cote avec la vraie. L'une garde les poles dominants au meme gain
statique ; l'autre garde leur seul mode, residu exact. Les deux echouent
au debut, et c'est l'enseignement : les poles dominants portent ici les
plus petits residus.

Les prefixes sont `TpCinq` et `TpSix`.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from polytech.signaux.application import enonce_lab_01 as e
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.modeles import coefficients, transmittance_d_une_equation
from polytech.signaux.domain.modes import contribution_des_modes, residus
from polytech.signaux.domain.poles_numeriques import (
    approximation_dominante,
    constante_de_temps,
    frequence_d_oscillation,
    poles_dominants,
    poles_et_zeros,
    reponse_impulsionnelle,
    reponse_indicielle,
    routh_du_troisieme_degre,
)

INSTANTS_CINQ = np.linspace(0.0, 10.0, 201)
INSTANTS_SIX = np.linspace(0.0, 6.0, 241)
APRES = 2.0
"""L'instant au-dela duquel le seul mode dominant decrit la reponse."""


def _macros_du_cinq() -> dict[str, str]:
    """La transmittance, ses trois poles, Routh, et les deux reponses."""
    transmittance = transmittance_d_une_equation(e.CINQ_SORTIE, e.CINQ_ENTREE)
    num, den = coefficients(transmittance)
    poles = poles_et_zeros(num, den)[0]
    instable = poles[np.argmax(poles.imag)]
    _, deux, un, _ = den
    return {
        "TpCinqTransmittance": sp.latex(transmittance.expression),
        "TpCinqPoleReel": nombre(instable.real, 4),
        "TpCinqPoleImag": nombre(instable.imag, 4),
        "TpCinqPoleTrois": nombre(float(np.min(poles.real)), 4),
        "TpCinqRouthGauche": nombre(deux * un, 0),
        "TpCinqRouth": nombre(routh_du_troisieme_degre(den), 0),
        "TpCinqImpulsion": coordonnees(
            INSTANTS_CINQ, reponse_impulsionnelle(num, den, INSTANTS_CINQ), 3
        ),
        "TpCinqIndice": coordonnees(
            INSTANTS_CINQ, reponse_indicielle(num, den, INSTANTS_CINQ), 3
        ),
    }


def _macros_des_poles_du_six() -> dict[str, str]:
    """Les poles, le dominant, sa frequence et sa constante de temps."""
    poles = poles_et_zeros(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR)[0]
    dominant = poles_dominants(poles)
    haut = dominant[np.argmax(dominant.imag)]
    valeurs, calcules = residus(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR)
    amplitude = {
        round(p.real, 6): abs(v) for v, p in zip(valeurs, calcules, strict=True)
    }
    reels = sorted(amplitude)
    return {
        "TpSixPoleUn": nombre(reels[0], 4),
        "TpSixPoleDeuxReel": nombre(reels[1], 4),
        "TpSixPoleDeuxImag": nombre(
            float(np.max(poles[np.isclose(poles.real, reels[1])].imag)), 4
        ),
        "TpSixDominantReel": nombre(haut.real, 4),
        "TpSixDominantImag": nombre(haut.imag, 4),
        "TpSixFrequence": nombre(frequence_d_oscillation(haut), 4),
        "TpSixTau": nombre(constante_de_temps(haut), 4),
        "TpSixResiduDominant": nombre(amplitude[reels[2]], 3),
        "TpSixResiduMoyen": nombre(amplitude[reels[1]], 3),
        "TpSixResiduRapide": nombre(amplitude[reels[0]], 3),
        "TpSixHZero": nombre(e.SIX_NUMERATEUR[0] / e.SIX_DENOMINATEUR[0], 5),
    }


def _macros_des_reponses_du_six() -> dict[str, str]:
    """La vraie reponse et ses deux approximations, avec leurs ecarts."""
    vraie = reponse_impulsionnelle(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR, INSTANTS_SIX)
    poles = poles_et_zeros(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR)[0]
    modale = contribution_des_modes(
        e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR, poles_dominants(poles), INSTANTS_SIX
    )
    reduite = reponse_impulsionnelle(
        *approximation_dominante(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR), INSTANTS_SIX
    )
    tard = INSTANTS_SIX >= APRES
    return {
        "TpSixCreteVraie": nombre(float(vraie.max()), 2),
        "TpSixCreteReduite": nombre(float(reduite.max()), 2),
        "TpSixModalZero": nombre(float(modale[0]), 2),
        "TpSixEcartModal": nombre(float(np.max(np.abs(vraie - modale))), 2),
        "TpSixEcartTardif": nombre(float(np.max(np.abs(vraie - modale)[tard])), 3),
        "TpSixApres": nombre(APRES, 0),
        "TpSixVraie": coordonnees(INSTANTS_SIX, vraie, 3),
        "TpSixModale": coordonnees(INSTANTS_SIX, modale, 3),
        "TpSixReduite": coordonnees(INSTANTS_SIX, reduite, 3),
    }


def macros_des_poles() -> dict[str, str]:
    """Les questions 5 et 6."""
    return {
        **_macros_du_cinq(),
        **_macros_des_poles_du_six(),
        **_macros_des_reponses_du_six(),
    }
