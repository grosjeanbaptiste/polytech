"""Macros de la feuille « Performances », exercices 1 a 3.

Prefixe `ExsHuit`.
"""

from __future__ import annotations

import math

from polytech.signaux.application import marges_exs_08 as mg
from polytech.signaux.application import regimes_exs_08 as rg
from polytech.signaux.application.formatage import nombre


def _regimes() -> dict[str, str]:
    perturbe = rg.regime_un(consigne=0, perturbation=5)
    cas = [
        rg.regime_deux(consigne=2, perturbation=0),
        rg.regime_deux(consigne=0, perturbation=10),
        rg.regime_deux(consigne=2, perturbation=10),
    ]
    macros = {
        "ExsHuitUnSortie": nombre(float(perturbe.y), 2),
        "ExsHuitUnCommande": nombre(float(perturbe.u), 2),
        "ExsHuitDeuxMarge": nombre(rg.marge_de_phase_deux(), 1),
    }
    for nom, regime in zip(("Un", "Deux", "Trois"), cas, strict=True):
        macros[f"ExsHuitDeuxCommande{nom}"] = nombre(float(regime.u), 2)
    return macros


def _marges() -> dict[str, str]:
    marge, pulsation = mg.marge_de_phase(1.0)
    gain = mg.gain_pour_la_marge(60.0)
    _, croisement = mg.marge_de_phase(gain)
    ideale, de_consigne, de_perturbation = mg.decomposition()
    return {
        "ExsHuitTroisPulsation": nombre(pulsation, 3),
        "ExsHuitTroisRetard": nombre(math.degrees(pulsation), 2),
        "ExsHuitTroisMarge": nombre(marge, 2),
        "ExsHuitTroisGain": nombre(gain, 3),
        "ExsHuitTroisCroisement": nombre(croisement, 3),
        "ExsHuitTroisIdeale": nombre(float(ideale), 2),
        "ExsHuitTroisErreurDeConsigne": nombre(float(de_consigne), 2),
        "ExsHuitTroisErreurDePerturbation": nombre(float(de_perturbation), 2),
        "ExsHuitTroisSortie": nombre(float(mg.sortie_finale()), 2),
    }


def macros_des_regimes() -> dict[str, str]:
    """Les macros des exercices 1, 2 et 3."""
    return {**_regimes(), **_marges()}
