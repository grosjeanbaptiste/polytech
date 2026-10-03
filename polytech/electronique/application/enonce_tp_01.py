"""Les donnees chiffrees du TP 1 d'electronique, et leurs calculs.

Source : `ElectroniqueInstrumentation/TP/TP_1.pdf`.

- **Le ferrite** (exercice 2.2.5) : la partie lineaire du graphique monte
  jusqu'a 1500 gauss pour 2 oersteds, que l'enonce convertit en 0,15 T et
  160 A/m. Le noyau a 20 spires au pas de 1 mm, et 25 mm2 de section.
- **La boucle carree** (exercice 3.2) : 5 cm de cote, fil de 0,65 mm de
  rayon, et deux resistances, 10 ohms et 10 megohms.
- **Les quatre resistances** (exercice 3.3) : R1 et R2 relient la borne
  `+` de la source au noeud de droite, R3 et R4 ce noeud a la masse --
  les deux diagonales du schema se croisent sans se toucher.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedances_en_parallele,
    impedances_en_serie,
)
from polytech.electronique.domain.solenoide import (
    courant_de_saturation,
    frequence_de_croisement,
    inductance_de_rosa,
    inductance_du_solenoide,
    permeabilite_lineaire,
)

INDUCTION_DE_SATURATION: Final = 1500 * 1e-4
CHAMP_DE_SATURATION: Final = 2 * 80.0
"""Deux oersteds, a 80 A/m chacun, la conversion de l'enonce."""
PAS, SPIRES, SECTION = 1e-3, 20, 25e-6
MU0: Final = 4 * math.pi * 1e-7
COTE, RAYON_DU_FIL, RESISTANCES_DE_LA_BOUCLE = 0.05, 0.65e-3, (10.0, 10e6)
SOURCE, R1, R2, R3, R4 = 5, 1000, 1000, 10_000, 10_000


def ferrite() -> tuple[float, float, float, float]:
    """`mu`, `mu_r`, le courant de saturation et l'inductance."""
    mu = permeabilite_lineaire(INDUCTION_DE_SATURATION, CHAMP_DE_SATURATION)
    longueur = SPIRES * PAS
    return (
        mu,
        mu / MU0,
        courant_de_saturation(CHAMP_DE_SATURATION, longueur, SPIRES),
        inductance_du_solenoide(mu, SPIRES, SECTION, longueur),
    )


def croisements() -> tuple[float, float, float]:
    """L'inductance de la boucle, et les deux frequences de croisement."""
    inductance = inductance_de_rosa(COTE, RAYON_DU_FIL)
    basse, haute = (
        frequence_de_croisement(r, inductance) for r in RESISTANCES_DE_LA_BOUCLE
    )
    return inductance, basse, haute


def pont_de_resistances() -> tuple[float, float]:
    """La tension du noeud de droite, et la resistance vue par la source."""
    amont = impedances_en_parallele(sp.Integer(R1), sp.Integer(R2))
    aval = impedances_en_parallele(sp.Integer(R3), sp.Integer(R4))
    return (
        float(SOURCE * diviseur_de_tension(amont, aval)),
        float(impedances_en_serie(amont, aval)),
    )
