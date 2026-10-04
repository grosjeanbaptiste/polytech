"""Chapitre XIV.1.4 : les matrices totalement unimodulaires.

Une matrice a coefficients entiers est **totalement unimodulaire** si les
determinants de toutes ses sous-matrices carrees -- de dimension 1, 2,
3, ... -- valent 0, 1 ou -1. Ses coefficients eux-memes ne peuvent donc
valoir que 0, 1 ou -1.

Propriete : si `T` l'est et que `d` est entier, tous les sommets de
`{x | T x <= d, x >= 0}` sont entiers. Le probleme en variables entieres
est alors un « faux » probleme entier : sa relaxation lineaire le resout.
C'est le cas de l'affectation et du transport.

Le test enumere les sous-matrices : son cout est exponentiel, il sert a
verifier une petite matrice, non a decider d'une grande.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp

PERMIS = (-1, 0, 1)


def est_totalement_unimodulaire(matrice: sp.Matrix) -> bool:
    """Vrai si tout mineur carre vaut 0, 1 ou -1."""
    for ordre in range(1, min(matrice.shape) + 1):
        for lignes in combinations(range(matrice.rows), ordre):
            for colonnes in combinations(range(matrice.cols), ordre):
                if matrice[list(lignes), list(colonnes)].det() not in PERMIS:
                    return False
    return True
