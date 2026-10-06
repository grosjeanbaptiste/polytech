"""Feuille « Systemes », exercice 3 : le reservoir qui se vide par gravite.

`S dh/dt = q_e - k racine(h)`. Le modele local est celui du cours ; ce
que l'exercice ajoute est la **mesure de son erreur** quand on s'eloigne
du point de repos.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from polytech.signaux.domain.reservoir import Reservoir

RESERVOIR: Final = Reservoir(section=0.5, coefficient=0.7)
DEBIT: Final = 0.75
"""Le debit d'entree au repos, en m^3/s."""
VARIATION: Final = 0.10
"""La variation relative du debit a la question 3."""
DUREE: Final = 40.0
"""En secondes : plus de vingt constantes de temps du modele local."""


def hauteur_de_repos() -> float:
    """`h^0 = (q_e / k)^2`, en metres."""
    return RESERVOIR.hauteur_de_repos(DEBIT)


def premier_ordre() -> tuple[float, float]:
    """Le gain statique, en s/m^2, et la constante de temps, en secondes."""
    return RESERVOIR.premier_ordre_local(DEBIT)


def hauteurs_apres_la_variation() -> tuple[float, float]:
    """Le nouveau regime : par le modele global, puis par le modele local."""
    gain, _ = premier_ordre()
    exacte = RESERVOIR.hauteur_de_repos(DEBIT * (1 + VARIATION))
    return exacte, hauteur_de_repos() + gain * DEBIT * VARIATION


def hauteur_simulee() -> float:
    """La hauteur atteinte en integrant le modele global apres l'echelon de debit."""
    instants = np.linspace(0.0, DUREE, 401)
    hauteurs = RESERVOIR.simuler(
        lambda _: DEBIT * (1 + VARIATION), depart=DEBIT, instants=instants
    )
    return float(hauteurs[-1])
