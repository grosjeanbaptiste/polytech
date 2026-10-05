"""Reduire une matrice au maximum : diagonaliser, triangulariser, ou moins encore.

Le syllabus d'algebre (X.2 a X.4) range une matrice reelle dans l'un de
trois cas :

* **diagonalisable** : la somme des multiplicites geometriques vaut `n` ;
* **triangularisable** : toutes les racines du polynome caracteristique
  sont reelles, c'est-a-dire que la somme des multiplicites algebriques
  vaut `n` ;
* **ni l'un ni l'autre** : des racines sont complexes, et la reduction
  s'arrete a un triangle suivi d'un bloc sans valeur propre reelle.

Le procede est le meme dans les trois cas (X.3.3, X.4.2) : reunir des
bases des sous-espaces propres, les prolonger en une base, calculer
`P_1^-1 A P_1` ; puis recommencer avec le bloc `A_1` qui reste en bas a
droite, tant qu'il a des valeurs propres reelles. La matrice de passage
est `P = P_1 . diag(I, P_2) ...`.

Une etape se compte par changement de base effectif : si `P_1^-1 A P_1`
est deja triangulaire -- c'est le cas des que la somme des multiplicites
geometriques atteint `n - 1` --, il n'y en a qu'une.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from polytech.maths.domain.algebre.triangularisation import (
    completer_en_base,
    semblable_par,
)


class Nature(Enum):
    """Ce que le procede peut atteindre."""

    DIAGONALISABLE = "diagonalisable"
    TRIANGULARISABLE = "triangularisable, non diagonalisable"
    NI_L_UN_NI_L_AUTRE = "ni diagonalisable ni triangularisable"


@dataclass(frozen=True, slots=True)
class Reduction:
    """La matrice de passage, `P^-1 A P`, et le nombre d'etapes qu'il a fallu."""

    passage: sp.Matrix
    reduite: sp.Matrix
    etapes: int


def _vecteurs_propres_reels(matrice: sp.Matrix) -> list[sp.Matrix]:
    return [
        vecteur
        for valeur, _, vecteurs in matrice.eigenvects()
        if valeur.is_real
        for vecteur in vecteurs
    ]


def nature_de_la_reduction(matrice: sp.Matrix) -> Nature:
    """Diagonalisable, triangularisable seulement, ou ni l'un ni l'autre."""
    reelles = [(a, len(v)) for valeur, a, v in matrice.eigenvects() if valeur.is_real]
    if sum(algebrique for algebrique, _ in reelles) < matrice.rows:
        return Nature.NI_L_UN_NI_L_AUTRE
    if sum(geometrique for _, geometrique in reelles) == matrice.rows:
        return Nature.DIAGONALISABLE
    return Nature.TRIANGULARISABLE


def reduire_au_maximum(matrice: sp.Matrix) -> Reduction:
    """Le procede du syllabus, etape apres etape, jusqu'au bloc irreductible."""
    propres = _vecteurs_propres_reels(matrice)
    if not propres:
        return Reduction(sp.eye(matrice.rows), sp.Matrix(matrice), 0)
    passage = completer_en_base(propres)
    reduite = semblable_par(matrice, passage)
    acquis = len(propres)
    if reduite.is_upper:
        return Reduction(passage, reduite, 1)
    suite = reduire_au_maximum(reduite[acquis:, acquis:])
    prolongee = sp.diag(sp.eye(acquis), suite.passage)
    finale = semblable_par(reduite, prolongee)
    return Reduction(passage * prolongee, finale, 1 + suite.etapes)
