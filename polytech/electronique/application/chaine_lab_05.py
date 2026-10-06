"""Laboratoire 5 d'electronique, questions 18 a 20 : l'amplificateur entier.

Etage d'entree, egaliseur, sommateur a volume reglable, etage de sortie
en pont. Le module assemble les quatre et dit ce que le montage fait
vraiment : son gain maximal, sa bande, et le reglage du volume qui
respecte le haut-parleur.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.electronique.application import dimensionnement_lab_05 as dm
from polytech.electronique.application import egaliseur_lab_05 as eg
from polytech.electronique.application import sortie_lab_05 as so
from polytech.electronique.domain.amplification import gain_inverseur

R9: Final = 22e3
R11: Final = 100e3
"""Le sommateur : `R_9` (et `R_10`, egale) en entree, le potentiometre en retour."""
PONT: Final = 2.0
"""Le haut-parleur recoit `V_o2 - V_o1 = 2 V_i` (question 12)."""
TROIS_DECIBELS: Final = 10 * math.log10(2)


def gain_du_sommateur(*, potentiometre: float = R11) -> float:
    """Le gain de chaque voie du sommateur, de 0 a `-R_11 / R_9`."""
    return gain_inverseur(R9, potentiometre)


def bande_du_sommateur() -> float:
    """`GBW` divise par le gain de bruit, `1 + R_11 / (R_9 // R_10)`."""
    return dm.PRODUIT_GAIN_BANDE / (1 + R11 / (R9 / 2))


def gain_maximal(*, potentiometre: float = R11) -> float:
    """De la source au haut-parleur, egaliseur au maximum, hors des coupures."""
    entree = dm.gains_realises()[0]
    return entree * abs(gain_du_sommateur(potentiometre=potentiometre)) * PONT


def crete_demandee() -> float:
    """Ce que les 25 mV de la source demanderaient au haut-parleur, volume a fond."""
    return dm.SOURCE * gain_maximal()


def potentiometre_pour_le_gain(gain: float) -> float:
    """Le reglage du volume qui donne ce gain total."""
    if not 0 <= gain <= gain_maximal():
        msg = f"gain hors de portee du montage : {gain}"
        raise ValueError(msg)
    return gain / (dm.gains_realises()[0] * PONT) * R9


def _premier_ordre(frequence: float, coupure: float) -> float:
    return 1 / math.hypot(1.0, frequence / coupure)


def perte(frequence: float) -> float:
    """Ce que la chaine perd a cette frequence, en decibels sous son gain maximal."""
    entree, _ = dm.bandes_des_etages()
    passe_haut = 1 / math.hypot(1.0, so.frequence_de_coupure() / frequence)
    module = (
        _premier_ordre(frequence, entree)
        * eg.somme_au_maximum(frequence)
        * _premier_ordre(frequence, bande_du_sommateur())
        * passe_haut
    )
    return -20 * math.log10(module)


def _dichotomie(basse: float, haute: float) -> float:
    """La frequence, entre ces deux bornes, ou la perte vaut 3 dB."""
    croissante = perte(haute) > perte(basse)
    for _ in range(80):
        milieu = math.sqrt(basse * haute)
        if (perte(milieu) > TROIS_DECIBELS) == croissante:
            haute = milieu
        else:
            basse = milieu
    return math.sqrt(basse * haute)


def bande_a_trois_decibels() -> tuple[float, float]:
    """Les deux frequences ou la chaine perd 3 dB."""
    return _dichotomie(1.0, 1e3), _dichotomie(1e3, 1e6)
