"""Familles de vecteurs de `K^n` : libres, liees, generatrices, bases, coordonnees.

Le syllabus d'algebre (VII.2, VII.3) ramene chaque question au rang de la
matrice dont les lignes sont les vecteurs de la famille :

* la famille est **libre** quand ce rang egale sa longueur, **liee** sinon ;
* elle est **generatrice** de `K^n` quand ce rang vaut `n` ;
* une **base** est libre et generatrice ;
* un vecteur est **combinaison lineaire** de la famille quand l'y ajouter
  ne releve pas le rang.

D'une famille generatrice on **extrait** une base en retirant chaque
vecteur combinaison des precedents ; une famille libre se **prolonge** en
base en lui ajoutant des vecteurs qui n'en sont pas combinaison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.rang_exact import rang_par_echelonnement

if TYPE_CHECKING:
    from collections.abc import Sequence

    Vecteur = Sequence[object]
    Famille = Sequence[Vecteur]


def matrice_des_lignes(famille: Famille) -> sp.Matrix:
    """La matrice dont chaque ligne est un vecteur de la famille."""
    if not famille:
        msg = "famille vide : elle n'a ni rang ni espace ambiant"
        raise ValueError(msg)
    if len({len(vecteur) for vecteur in famille}) > 1:
        msg = "vecteurs de tailles differentes : ils ne sont pas du meme espace"
        raise ValueError(msg)
    return sp.Matrix([list(vecteur) for vecteur in famille])


def rang_de_la_famille(famille: Famille) -> int:
    """Le nombre de vecteurs lineairement independants."""
    return rang_par_echelonnement(matrice_des_lignes(famille))


def est_libre(famille: Famille) -> bool:
    """Le rang egale la longueur."""
    return rang_de_la_famille(famille) == len(famille)


def est_generatrice(famille: Famille) -> bool:
    """Le rang egale la dimension de l'espace ambiant."""
    return rang_de_la_famille(famille) == len(famille[0])


def est_une_base(famille: Famille) -> bool:
    """Libre et generatrice."""
    return est_libre(famille) and est_generatrice(famille)


def appartient(vecteur: Vecteur, famille: Famille) -> bool:
    """Vrai si ajouter le vecteur a la famille n'en releve pas le rang."""
    return rang_de_la_famille((*famille, vecteur)) == rang_de_la_famille(famille)


def relation_de_dependance(famille: Famille) -> tuple[sp.Expr, ...]:
    """Des coefficients non tous nuls d'une combinaison lineaire nulle."""
    noyau = matrice_des_lignes(famille).T.nullspace()
    if not noyau:
        msg = "famille libre : sa seule combinaison nulle a tous ses coefficients nuls"
        raise ValueError(msg)
    return tuple(noyau[0])


def coordonnees(vecteur: Vecteur, famille: Famille) -> tuple[sp.Expr, ...]:
    """Les coefficients, uniques, du vecteur dans une famille libre."""
    if not est_libre(famille):
        msg = "famille liee : les coordonnees ne seraient pas uniques"
        raise ValueError(msg)
    if not appartient(vecteur, famille):
        msg = "le vecteur n'est pas combinaison lineaire de la famille"
        raise ValueError(msg)
    colonnes = matrice_des_lignes(famille).T
    solution, _ = colonnes.gauss_jordan_solve(sp.Matrix(list(vecteur)))
    return tuple(solution)


def extraire_une_base(famille: Famille) -> tuple[Vecteur, ...]:
    """Les vecteurs qui ne sont pas combinaison de ceux deja retenus."""
    retenus: list[Vecteur] = []
    for vecteur in famille:
        if est_libre((*retenus, vecteur)):
            retenus.append(vecteur)
    return tuple(retenus)


def prolonger_en_base(famille: Famille) -> tuple[Vecteur, ...]:
    """La famille, suivie des vecteurs canoniques qui la gardent libre."""
    if not est_libre(famille):
        msg = "famille liee : elle ne se prolonge pas en une base"
        raise ValueError(msg)
    dimension = len(famille[0])
    canoniques = [
        tuple(int(i == j) for j in range(dimension)) for i in range(dimension)
    ]
    return extraire_une_base((*famille, *canoniques))
