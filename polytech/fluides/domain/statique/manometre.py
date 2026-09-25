"""Cours II.3 -- Manometre multi-fluides.

Le cours etablit, pour la configuration eau / huile / mercure de l'activite :

    p1 + rho_eau g h1 + rho_huile g h2 - rho_mercure g h3 = p2

Cette regle se generalise en une *marche* le long de l'appareil : on ajoute le
poids de la colonne quand on descend, on le retranche quand on remonte.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from polytech.fluides.domain.grandeurs import (
    PESANTEUR,
    Ecart,
    Hauteur,
    MasseVolumique,
    Pression,
)
from polytech.fluides.domain.statique.stevin import variation_de_pression


class Sens(Enum):
    """Sens de parcours d'un troncon, de l'amont vers l'aval."""

    DESCENTE = 1
    MONTEE = -1


@dataclass(frozen=True, slots=True)
class Troncon:
    """Portion de manometre parcourue dans un seul fluide."""

    fluide: MasseVolumique
    hauteur: Hauteur
    sens: Sens
    nom: str = "fluide"

    def contribution(self, pesanteur: float = PESANTEUR) -> Ecart:
        """Poids de la colonne, en valeur absolue."""
        return variation_de_pression(self.fluide, self.hauteur, pesanteur)


@dataclass(frozen=True, slots=True)
class Manometre:
    """Suite ordonnee de troncons, de l'amont (point 1) vers l'aval (point 2).

    Agregat : la racine porte l'invariant (au moins un troncon) et c'est par
    elle que passe tout calcul -- aucun troncon n'est interroge de l'exterieur.
    """

    troncons: tuple[Troncon, ...]
    pesanteur: float = PESANTEUR

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if not self.troncons:
            msg = "un manometre comporte au moins un troncon"
            raise ValueError(msg)

    def equation_symbolique(self) -> sp.Eq:
        """Equation litterale du manometre, telle que l'ecrit le cours II.3.

        Chaque troncon apporte ``+ rho g h`` s'il descend, ``- rho g h`` s'il
        remonte ; la suite des troncons dicte donc l'equation entiere.
        """
        g = sp.Symbol("g", positive=True)
        membre = sp.Symbol("p_1")
        for rang, troncon in enumerate(self.troncons, start=1):
            rho = sp.Symbol(f"rho_{troncon.nom}", positive=True)
            hauteur = sp.Symbol(f"h_{rang}", positive=True)
            membre += troncon.sens.value * rho * g * hauteur
        return sp.Eq(membre, sp.Symbol("p_2"))

    def contributions(self) -> tuple[Ecart, ...]:
        """Poids de chaque colonne, dans l'ordre de parcours."""
        return tuple(t.contribution(self.pesanteur) for t in self.troncons)

    def _denivelee_totale(self) -> float:
        return sum(
            t.sens.value * t.contribution(self.pesanteur).pascals for t in self.troncons
        )

    def pression_amont(self, aval: Pression) -> Pression:
        """Pression au point 1, connaissant celle du point 2."""
        return Pression(pascals=aval.pascals - self._denivelee_totale())

    def pression_aval(self, amont: Pression) -> Pression:
        """Pression au point 2, connaissant celle du point 1."""
        return Pression(pascals=amont.pascals + self._denivelee_totale())

    def part_dominante(self) -> float:
        """Part du troncon le plus lourd dans la somme des contributions."""
        poids = [c.pascals for c in self.contributions()]
        return max(poids) / sum(poids)
