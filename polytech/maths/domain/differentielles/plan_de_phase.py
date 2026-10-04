"""La nature d'un point critique, lue sur les valeurs propres du linearise.

Pres d'un point ou le champ s'annule, un systeme autonome se comporte
comme son linearise `y' = A y`. Les valeurs propres de `A` decident :

* reelles de meme signe -- un **noeud**, stable si elles sont negatives ;
* reelles de signes opposes -- un **point de selle**, toujours instable ;
* complexes -- un **foyer**, stable si la partie reelle est negative, ou
  un **centre** quand elle est nulle : les trajectoires tournent sans
  s'approcher ni s'eloigner.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp


class Nature(Enum):
    """Les six portraits qu'un linearise sans valeur propre nulle peut donner."""

    NOEUD_STABLE = "noeud stable"
    NOEUD_INSTABLE = "noeud instable"
    SELLE = "point de selle"
    FOYER_STABLE = "foyer stable"
    FOYER_INSTABLE = "foyer instable"
    CENTRE = "centre"

    @property
    def est_stable(self) -> bool:
        """Vrai si une trajectoire voisine reste voisine."""
        return self in {Nature.NOEUD_STABLE, Nature.FOYER_STABLE, Nature.CENTRE}


def nature_du_point_critique(matrice: sp.Matrix) -> Nature:
    """Le portrait du linearise ; leve si une valeur propre est nulle."""
    valeurs = list(matrice.eigenvals())
    if any(sp.simplify(valeur) == 0 for valeur in valeurs):
        msg = "valeur propre nulle : le linearise ne decide pas de la nature"
        raise ValueError(msg)
    reelles = [sp.re(valeur) for valeur in valeurs]
    if all(sp.im(valeur) == 0 for valeur in valeurs):
        if all(r.is_negative for r in reelles):
            return Nature.NOEUD_STABLE
        return (
            Nature.NOEUD_INSTABLE
            if all(r.is_positive for r in reelles)
            else Nature.SELLE
        )
    if reelles[0] == 0:
        return Nature.CENTRE
    return Nature.FOYER_STABLE if reelles[0].is_negative else Nature.FOYER_INSTABLE
