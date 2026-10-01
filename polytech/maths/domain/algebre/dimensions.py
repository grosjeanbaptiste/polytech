"""La dimension des espaces vectoriels usuels -- sections VII.7 a VII.10.

« Espaces vectoriels de matrices », « de polynomes », « de fonctions »,
puis « Espaces vectoriels quelconques ». L'ordre du syllabus va du
concret au general, et le point d'arrivee est qu'un espace vectoriel
n'est pas necessairement un espace de colonnes.

Chaque etape reserve une surprise sur la dimension.

**Les matrices `n x m` : dimension `n m`**, le produit et non la somme.
La raison est la base canonique -- les matrices n'ayant qu'un seul
coefficient non nul -- et il y en a autant que de cases. Ce module la
construit plutot que de poser le produit, pour que la dimension soit un
**compte** et non une formule.

**Les polynomes de degre au plus `n` : dimension `n + 1`.** Le « plus
un » est le terme constant, et c'est l'erreur classique du chapitre.

**Les fonctions : dimension infinie.** C'est la raison d'etre de VII.10 :
tout ce qui precede s'y applique sauf ce qui suppose une base finie.
Distinguer explicitement le cas infini vaut mieux que de rendre un grand
nombre, qui laisserait croire a un espace de dimension finie tres grande.
"""

from __future__ import annotations

from typing import Final

DIMENSION_INFINIE: Final = -1
"""Le marqueur du cas infini, distinct de toute dimension finie.

Un entier negatif plutot que `math.inf` : la dimension est un cardinal,
et un flottant infini se melangerait aux comparaisons arithmetiques sans
se faire remarquer.
"""


def dimension_des_matrices(lignes: int, colonnes: int) -> int:
    """`n m` : le compte de la base canonique, et non une formule posee."""
    return len(base_canonique_des_matrices(lignes, colonnes))


def base_canonique_des_matrices(
    lignes: int,
    colonnes: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Les matrices n'ayant qu'un seul coefficient non nul.

    Il y en a autant que de cases, et c'est la raison pour laquelle la
    dimension est le **produit** des formats.
    """
    if lignes <= 0 or colonnes <= 0:
        msg = f"format nul ou negatif : {lignes} x {colonnes}"
        raise ValueError(msg)
    return tuple(
        tuple(
            tuple(1.0 if (i, j) == (li, lj) else 0.0 for j in range(colonnes))
            for i in range(lignes)
        )
        for li in range(lignes)
        for lj in range(colonnes)
    )


def dimension_des_polynomes(degre_maximal: int) -> int:
    """`n + 1` pour le degre au plus `n` : `1`, `x`, ..., `x^n`."""
    if degre_maximal < 0:
        msg = f"degre negatif : {degre_maximal} -- le polynome nul se traite a part"
        raise ValueError(msg)
    return degre_maximal + 1


def est_de_dimension_finie(dimension: int) -> bool:
    """Vrai sauf pour le marqueur du cas infini.

    La distinction porte tout le sens de VII.10 : un espace de fonctions
    n'a pas de base finie, et les demonstrations qui en supposent une
    cessent d'y valoir.
    """
    return dimension != DIMENSION_INFINIE
