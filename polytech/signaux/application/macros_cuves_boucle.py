"""Macros du TP « Reservoir », deuxieme partie : les boucles fermees.

Prefixe `TpCuves`.
"""

from __future__ import annotations

import math

from polytech.signaux.application import cuves_boucle as bo
from polytech.signaux.application import cuves_courbes as co
from polytech.signaux.application import cuves_pid as pid
from polytech.signaux.application.formatage import coordonnees, nombre

_GAINS = (("Faible", 0.3), ("Unite", 1.0))
"""Les deux gains dont le corrige trace la reponse."""


def _proportionnel() -> dict[str, str]:
    pulsation, marge = bo.marge_de_gain_analytique()
    marges = bo.marges(1.0)
    macros = {
        "TpCuvesStatique": nombre(bo.gain_statique(), 3),
        "TpCuvesPulsationCritique": nombre(pulsation * 1e3, 2),
        "TpCuvesGainLimite": nombre(marge, 2),
        "TpCuvesMargeDeGain": nombre(20 * math.log10(marge), 1),
        "TpCuvesMargeDePhase": nombre(marges.de_phase, 1),
        "TpCuvesBlack": co.black(bo.PROCEDE, 2e-4, 0.05),
    }
    for nom, gain in _GAINS:
        instants, sortie = bo.reponse_indicielle(gain * bo.PROCEDE, 8000.0)
        reponse = bo.reponse(gain)
        macros[f"TpCuvesReponse{nom}"] = coordonnees(instants[::20], sortie[::20], 4)
        macros[f"TpCuvesErreur{nom}"] = nombre(100 * bo.erreur_de_regime(gain), 1)
        macros[f"TpCuvesDepassement{nom}"] = nombre(reponse.depassement, 0)
        macros[f"TpCuvesMontee{nom}"] = nombre(reponse.montee, 0)
    return macros


def _pid() -> dict[str, str]:
    _, commande = pid.regime_sous_consigne()
    marges, m = pid.marges(), pid.caracteristiques()
    estimation, simulee = pid.estimation(), pid.reponse_a_la_consigne()
    instants, sortie = bo.reponse_indicielle(pid.OUVERTE, pid.DUREE)
    temps, ecart = pid.reponse_a_la_perturbation()
    return {
        "TpCuvesCommande": nombre(float(commande), 3),
        "TpCuvesPidMargeDePhase": nombre(marges.de_phase, 1),
        "TpCuvesPidMargeDeGain": nombre(marges.de_gain_en_decibels(), 1),
        "TpCuvesPidBlack": co.black(pid.OUVERTE, 2e-4, 0.05),
        "TpCuvesPidBande": nombre(m.bande_passante * 1e3, 2),
        "TpCuvesPidMonteeBasse": nombre(estimation.montee[0], 0),
        "TpCuvesPidMonteeHaute": nombre(estimation.montee[1], 0),
        "TpCuvesPidMontee": nombre(simulee.montee, 0),
        "TpCuvesPidDepassement": nombre(simulee.depassement, 1),
        "TpCuvesPidReponse": coordonnees(instants[::20], sortie[::20], 4),
        "TpCuvesPidPerturbation": coordonnees(temps[::20], ecart[::20], 4),
        "TpCuvesPidEcartMaximal": nombre(float(abs(ecart).max()), 2),
    }


def macros_des_boucles() -> dict[str, str]:
    """Les macros du controleur proportionnel, puis du PID."""
    return {**_proportionnel(), **_pid()}
