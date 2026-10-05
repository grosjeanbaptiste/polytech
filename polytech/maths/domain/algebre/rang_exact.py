"""Le rang d'une matrice, en ecriture exacte, et selon la valeur d'un parametre.

Le syllabus d'algebre (VI.1, VI.3) donne deux lectures du rang :

* echelonner la matrice : le rang est le nombre de lignes non nulles de
  la matrice echelonnee ;
* « le rang d'une matrice est l'ordre de sa plus grande sous-matrice
  carree de determinant non nul ».

Quand la matrice porte un parametre, son rang vaut une meme valeur --
le rang generique `r` -- sauf en un nombre fini de valeurs, ou il chute.
Il ne chute que la ou **tous** les mineurs d'ordre `r` s'annulent
ensemble : ces valeurs sont les racines de leur plus grand commun
diviseur.

`rang.py` porte le rang en flottants, a pivot partiel ; ici il est exact.
"""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import (
    colonnes_des_pivots,
    echelonner,
)


def rang_par_echelonnement(matrice: sp.Matrix) -> int:
    """Le nombre de lignes non nulles de la matrice echelonnee."""
    pivots = colonnes_des_pivots(echelonner(matrice).matrice)
    return sum(1 for pivot in pivots if pivot is not None)


def _mineurs(matrice: sp.Matrix, ordre: int) -> list[sp.Expr]:
    return [
        matrice[list(lignes), list(colonnes)].det()
        for lignes in combinations(range(matrice.rows), ordre)
        for colonnes in combinations(range(matrice.cols), ordre)
    ]


def rang_par_les_mineurs(matrice: sp.Matrix) -> int:
    """L'ordre de la plus grande sous-matrice carree de determinant non nul."""
    for ordre in range(min(matrice.shape), 0, -1):
        if any(sp.simplify(mineur) != 0 for mineur in _mineurs(matrice, ordre)):
            return ordre
    return 0


def rangs_selon_le_parametre(
    matrice: sp.Matrix, parametre: sp.Symbol
) -> tuple[int, dict[sp.Expr, int]]:
    """Le rang generique, et le rang en chaque valeur reelle ou il chute."""
    generique = rang_par_les_mineurs(matrice)
    commun = sp.gcd_list([sp.expand(m) for m in _mineurs(matrice, generique)])
    particuliers = {
        valeur: rang_par_les_mineurs(matrice.subs(parametre, valeur))
        for valeur in sp.roots(commun, parametre)
        if valeur.is_real
    }
    return generique, particuliers
