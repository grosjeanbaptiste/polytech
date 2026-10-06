"""Macros du TP 8 d'electronique : le bruit thermique d'un montage a AOP.

Le prefixe est `TpHuit`. Les densites efficaces sont publiees en
nanovolts par racine de hertz, l'unite de l'enonce.
"""

from __future__ import annotations

import math

from polytech.electronique.application import bruit_tp_08 as br
from polytech.electronique.application.formatage import nombre


def macros_du_tp_huit() -> dict[str, str]:
    """Toutes les valeurs que le corrige du TP 8 compose."""
    sortie = br.densite_en_sortie()
    de_r1, _ = br.parts_du_bruit_de_sortie()
    gauche, droite = br.Montage.NON_INVERSEUR, br.Montage.INVERSEUR
    return {
        "TpHuitDensiteKilo": nombre(br.densite_efficace(br.R1) * 1e9, 2),
        "TpHuitDensiteKiloCarree": nombre(br.densite_efficace(br.R1) ** 2 * 1e17, 3),
        "TpHuitDensiteCent": nombre(br.densite_efficace(br.R2) * 1e9, 2),
        "TpHuitSortieCarree": nombre(sortie * 1e12, 4),
        "TpHuitSortie": nombre(math.sqrt(sortie) * 1e9, 1),
        "TpHuitPartRUn": nombre(100 * de_r1, 1),
        "TpHuitEntreeGauche": nombre(br.rapportee_a_l_entree(gauche) * 1e9, 2),
        "TpHuitEntreeDroite": nombre(br.rapportee_a_l_entree(droite) * 1e9, 2),
        "TpHuitResistanceGauche": nombre(br.resistance_equivalente(gauche), 1),
        "TpHuitResistanceDroite": nombre(br.resistance_equivalente(droite), 0),
        "TpHuitSortieReduite": nombre(
            math.sqrt(br.densite_en_sortie(echelle=0.01)) * 1e9, 2
        ),
    }
