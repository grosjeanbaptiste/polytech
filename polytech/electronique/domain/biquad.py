"""La cellule biquad de Tow-Thomas, et sa mise a l'echelle.

La cellule se dimensionne d'abord en valeurs *normalisees*, ou les
resistances se comptent en micro-ohms et les capacites en farads : ce sont
des nombres, pas des composants. Un facteur d'echelle les ramene ensuite
dans le domaine des pieces reelles, en multipliant les resistances et en
divisant les capacites.

Ce facteur n'est pas libre : il doit a la fois remonter la plus petite
resistance au-dessus du minimum disponible et abaisser la plus grande
capacite sous le maximum disponible. Deux contraintes, un seul nombre --
c'est la plus exigeante qui commande.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TowThomas:
    """Une cellule definie par sa pulsation, son facteur de qualite, son gain."""

    pulsation: float
    qualite: float
    gain: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.pulsation <= 0:
            msg = f"pulsation non positive : {self.pulsation}"
            raise ValueError(msg)
        if self.qualite <= 0:
            msg = f"facteur de qualite non positif : {self.qualite}"
            raise ValueError(msg)
        if self.gain <= 0:
            msg = f"gain non positif : {self.gain}"
            raise ValueError(msg)

    def r1_normalisee(self) -> float:
        """La resistance qui fixe le facteur de qualite : `Q / omega_0`."""
        return self.qualite / self.pulsation

    def r2_normalisee(self) -> float:
        """Celle qui fixe la pulsation de reference : `1 / omega_0`.

        `R_3` lui est egale ; la cellule standard les prend identiques.
        """
        return 1 / self.pulsation

    def r4_normalisee(self) -> float:
        """Celle qui fixe le gain continu : `1 / (omega_0 * G)`."""
        return 1 / (self.pulsation * self.gain)

    def facteur_d_echelle(
        self,
        resistance_minimale: float,
        capacite_maximale: float,
    ) -> float:
        """Le plus petit facteur qui satisfasse les deux contraintes.

        La plus petite resistance normalisee est `R_4` : c'est elle qui
        commande la contrainte basse sur les resistances.
        """
        if resistance_minimale <= 0 or capacite_maximale <= 0:
            msg = (
                "resistance minimale ou capacite maximale non positive : "
                f"{resistance_minimale}, {capacite_maximale}"
            )
            raise ValueError(msg)
        return max(resistance_minimale / self.r4_normalisee(), 1 / capacite_maximale)
