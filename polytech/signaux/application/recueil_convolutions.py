"""Recueil d'exercices, planches 13 et 18 : deux creneaux convolues.

Le produit de convolution de deux creneaux est un trapeze : il monte
pendant que le plus court entre dans le plus long, tient un palier tant
qu'il y est tout entier, et redescend quand il en sort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True, slots=True)
class Creneau:
    """Un creneau : son debut, sa largeur, sa hauteur."""

    debut: float
    largeur: float
    hauteur: float

    def valeurs(self, instants: np.ndarray) -> np.ndarray:
        """Le creneau, echantillonne."""
        dedans = (instants >= self.debut) & (instants < self.debut + self.largeur)
        return np.where(dedans, self.hauteur, 0.0)


Creneaux = tuple[Creneau, Creneau]

PLANCHE_13: Final[Creneaux] = (Creneau(0, 4, 1), Creneau(0, 2, 1))
"""`x = rect((t - 2)/4)` et `h = rect((t - 1)/2)`."""
PLANCHE_18: Final[Creneaux] = (Creneau(4, 2, 2), Creneau(0, 4, 1))
"""`x` vaut 2 de 4 a 6 ; `h` vaut 1 de 0 a 4."""
_PAS: Final = 1e-3


def sommets(donnees: Creneaux) -> tuple[tuple[float, float], ...]:
    """Les quatre sommets du trapeze, de gauche a droite."""
    entree, reponse = donnees
    depart = entree.debut + reponse.debut
    court = min(entree.largeur, reponse.largeur)
    long = max(entree.largeur, reponse.largeur)
    palier = entree.hauteur * reponse.hauteur * court
    return (
        (depart, 0),
        (depart + court, palier),
        (depart + long, palier),
        (depart + court + long, 0),
    )


def par_integration(donnees: Creneaux, instant: float) -> float:
    """`y(t) = integrale de x(u) h(t - u) du`, par une somme de Riemann."""
    entree, reponse = donnees
    variable = np.arange(entree.debut, entree.debut + entree.largeur, _PAS)
    produit = entree.valeurs(variable) * reponse.valeurs(instant - variable)
    return float(np.sum(produit) * _PAS)


def aire_de_la_sortie(donnees: Creneaux) -> float:
    """L'aire du trapeze : la demi-somme des bases, fois la hauteur."""
    (a, _), (b, hauteur), (c, _), (d, _) = sommets(donnees)
    return ((d - a) + (c - b)) * hauteur / 2


def produit_des_aires(donnees: Creneaux) -> float:
    """L'aire de l'entree fois celle de la reponse impulsionnelle."""
    entree, reponse = donnees
    return entree.largeur * entree.hauteur * reponse.largeur * reponse.hauteur
