"""Macros de l'exercice cote du 23 mai 2024 (Mons). Prefixe `CoteMonsVingtQuatre`."""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2024_mons_thevenin as th
from polytech.electromagnetisme.application import cote_2024_mons_tiges as tg
from polytech.electromagnetisme.application.formatage import nombre


def macros_cote_mons_vingt_quatre() -> dict[str, str]:
    """Les macros des questions 1 et 2 ; la question 3 est litterale."""
    tension, resistance = th.thevenin()
    return {
        "CoteMonsVingtQuatreResistance": nombre(float(resistance), 1),
        "CoteMonsVingtQuatreTension": nombre(float(tension), 1),
        "CoteMonsVingtQuatreCourant": nombre(float(th.courant_de_la_maille()), 1),
        "CoteMonsVingtQuatreEnergie": nombre(float(th.energie_en_microjoules()), 1),
        "CoteMonsVingtQuatreLongueur": nombre(tg.longueur_de_la_tige(), 2),
        "CoteMonsVingtQuatreChamp": nombre(tg.GRADIENT * tg.DISTANCE, 1),
        "CoteMonsVingtQuatreVitesse": nombre(tg.vitesse(), 2),
    }
