"""Macros des exercices resolus, chapitres 4 et 5. Prefixe `Resolus`."""

from __future__ import annotations

import numpy as np
from scipy import signal

from polytech.signaux.application import resolus_graphes as gr
from polytech.signaux.application import resolus_quadripoles as qu
from polytech.signaux.application import resolus_regimes as rg
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.modeles import coefficients
from polytech.signaux.domain.transmittance import Transmittance

_PERIODES_DE_L_ENCLENCHEMENT = 2.0
"""La constante de temps de l'illustration de 0.30, comptee en periodes."""


def _reponses(expression: object, fin: float, echelle: float = 1.0) -> tuple[str, str]:
    """Les points de l'impulsionnelle puis de l'indicielle, par `scipy`."""
    numerateur, denominateur = coefficients(Transmittance(expression=expression))
    instants = np.linspace(0.0, fin, 201)
    _, h = signal.impulse((numerateur, denominateur), T=instants)
    _, a = signal.step((numerateur, denominateur), T=instants)
    return (
        coordonnees(instants * echelle, h, 4),
        coordonnees(instants * echelle, a, 4),
    )


def _chapitre_quatre() -> dict[str, str]:
    reduits = np.linspace(0.0, 5.0, 51)
    instant, sommet = qu.sommet_de_l_impulsionnelle()
    exemple = qu.transmittance_de_l_exemple().subs(qu.DE_L_EXEMPLE)
    h_exemple, a_exemple = _reponses(exemple, 7e-3, 1e3)
    h_27, a_27 = _reponses(gr.transmittance("0.27"), 2.5)
    lente, rapide = gr.constantes_de_temps("0.27")
    lent, vif = gr.poids_des_modes_de_l_impulsionnelle("0.27")
    reel, pulsation = gr.pole_instable("0.28")
    instant_25, sommet_25 = gr.sommet_de_l_indicielle("0.25")
    return {
        "ResolusVingtDeuxMontee": coordonnees(reduits, 1 - np.exp(-reduits), 4),
        "ResolusVingtDeuxDescente": coordonnees(reduits, np.exp(-reduits), 4),
        "ResolusVingtQuatreInstant": nombre(instant * 1e3, 3),
        "ResolusVingtQuatreSommet": nombre(sommet, 1),
        "ResolusVingtQuatreDepassement": nombre(qu.sommet_de_l_indicielle(), 3),
        "ResolusVingtQuatreH": h_exemple,
        "ResolusVingtQuatreA": a_exemple,
        "ResolusVingtCinqInstant": nombre(instant_25 * 1e3, 1),
        "ResolusVingtCinqSommet": nombre(sommet_25, 4),
        "ResolusVingtSeptH": h_27,
        "ResolusVingtSeptA": a_27,
        "ResolusVingtSeptLente": nombre(lente, 3),
        "ResolusVingtSeptRapide": nombre(rapide * 1e3, 1),
        "ResolusVingtSeptPoidsLent": nombre(lent, 4),
        "ResolusVingtSeptPoidsVif": nombre(vif, 4),
        "ResolusVingtHuitReel": nombre(reel, 3),
        "ResolusVingtHuitPulsation": nombre(pulsation, 2),
        "ResolusVingtHuitPeriode": nombre(2 * np.pi / pulsation, 2),
    }


def _chapitre_cinq() -> dict[str, str]:
    periodes = np.linspace(0.0, 4.0, 321)
    regime = np.cos(2 * np.pi * periodes)
    transitoire = np.exp(-periodes / _PERIODES_DE_L_ENCLENCHEMENT)
    crete = rg.crete_relative(constante_en_periodes=_PERIODES_DE_L_ENCLENCHEMENT)
    return {
        "ResolusTrenteCourant": coordonnees(periodes, regime - transitoire, 4),
        "ResolusTrenteTransitoire": coordonnees(periodes[::8], -transitoire[::8], 4),
        "ResolusTrenteCrete": nombre(crete, 2),
        "ResolusTrenteTroisLimite": nombre(float(rg.alpha_limite()), 2),
        "ResolusTrenteCinqAttenuation": nombre(rg.attenuation_du_fondamental(), 1),
    }


def macros_resolus_systemes() -> dict[str, str]:
    """Les macros des exercices 0.22 a 0.35."""
    return {**_chapitre_quatre(), **_chapitre_cinq()}
