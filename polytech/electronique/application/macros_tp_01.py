"""Macros du TP 1 d'electronique : le ferrite, la boucle, le pont.

Les grandeurs sont publiees dans l'unite ou elles se lisent -- les
microhenrys du ferrite, les nanohenrys de la boucle -- pour que
l'erreur d'unite de l'officiel se voie a cote de la bonne valeur.

Le prefixe est `TpUn`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.application import enonce_tp_01 as e
from polytech.electronique.application.formatage import nombre
from polytech.electronique.domain.laplace import impedances_en_parallele


def macros_du_tp_un() -> dict[str, str]:
    """Les trois exercices chiffres du TP."""
    mu, relative, courant, inductance = e.ferrite()
    boucle, basse, haute = e.croisements()
    tension, equivalente = e.pont_de_resistances()
    return {
        "TpUnMu": nombre(mu * 1e4, 3),
        "TpUnMuRelatif": nombre(relative, 0),
        "TpUnCourant": nombre(courant * 1e3, 0),
        "TpUnLongueur": nombre(e.SPIRES * e.PAS * 1e3, 0),
        "TpUnInductance": nombre(inductance * 1e6, 0),
        "TpUnRapport": nombre(e.COTE / e.RAYON_DU_FIL, 2),
        "TpUnLog": nombre(math.log(e.COTE / e.RAYON_DU_FIL), 3),
        "TpUnRosa": nombre(boucle * 1e9, 1),
        "TpUnCroisementBas": nombre(basse / 1e6, 2),
        "TpUnCroisementHaut": nombre(haute / 1e12, 2),
        "TpUnHaut": nombre(
            float(impedances_en_parallele(sp.Integer(e.R1), sp.Integer(e.R2))), 0
        ),
        "TpUnBas": nombre(
            float(impedances_en_parallele(sp.Integer(e.R3), sp.Integer(e.R4))) / 1e3, 0
        ),
        "TpUnTension": nombre(tension, 3),
        "TpUnEquivalente": nombre(equivalente / 1e3, 1),
    }
