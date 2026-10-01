"""Espaces vectoriels et applications lineaires -- chapitres VII et VIII.

VII.2 Familles liees et libres, VII.3 Familles generatrices, bases et
dimension, VII.5 Lien avec les systemes lineaires ; VIII.5 Representation
matricielle, VIII.6 Changement de base.

Le lien que VII.5 annonce est ce qui rend tout cela calculable : une
famille est libre si le systeme homogene qu'elle forme n'admet que la
solution nulle, c'est-a-dire si son **rang** egale le nombre de ses
vecteurs. Liberte, dimension engendree et rang sont donc trois lectures
du meme calcul, et c'est pourquoi ce module s'appuie sur `rang`.

Deux conventions a ne pas confondre.

**Les colonnes de la matrice sont les images** des vecteurs de la base.
L'oublier transpose la matrice -- erreur qui passe inapercue sur une
application symetrique, et qui se voit immediatement si l'on verifie que
l'image du i-eme vecteur de base est bien la i-eme colonne.

**Generatrice et libre sont independants.** Quatre vecteurs du plan sont
generateurs et lies ; deux vecteurs d'un espace de dimension trois sont
libres et non generateurs. Une base est les deux, et c'est pourquoi on la
definit par deux proprietes et non par une.
"""

from __future__ import annotations

from polytech.maths.domain.algebre.matrices import Matrice, produit, transposee
from polytech.maths.domain.algebre.rang import est_inversible, rang


def _exiger_la_famille(famille: tuple[tuple[float, ...], ...]) -> None:
    """Refuse la famille vide, dont la liberte est une convention."""
    if not famille:
        msg = "famille vide : sa liberte est une convention, non un calcul"
        raise ValueError(msg)


def est_libre(famille: tuple[tuple[float, ...], ...]) -> bool:
    """Vrai si le rang de la famille egale le nombre de ses vecteurs."""
    _exiger_la_famille(famille)
    return rang(famille) == len(famille)


def dimension_engendree(famille: tuple[tuple[float, ...], ...]) -> int:
    """La dimension du sous-espace engendre : le rang, et non le compte.

    Trois vecteurs dont deux colineaires engendrent un plan.
    """
    _exiger_la_famille(famille)
    return rang(famille)


def est_generatrice(famille: tuple[tuple[float, ...], ...], dimension: int) -> bool:
    """Vrai si la famille engendre tout l'espace de cette dimension."""
    return dimension_engendree(famille) == dimension


def matrice_d_une_application(images: tuple[tuple[float, ...], ...]) -> Matrice:
    """La matrice dont les **colonnes** sont les images donnees.

    Les images sont fournies dans l'ordre des vecteurs de la base ; la
    matrice est donc la transposee de leur empilement en lignes.
    """
    _exiger_la_famille(images)
    return transposee(images)


def applique(matrice: Matrice, vecteur: tuple[float, ...]) -> tuple[float, ...]:
    """`A x` : l'image du vecteur par l'application."""
    if len(matrice[0]) != len(vecteur):
        msg = (
            f"vecteur de taille {len(vecteur)} pour une matrice a "
            f"{len(matrice[0])} colonnes"
        )
        raise ValueError(msg)
    return tuple(
        sum(a * b for a, b in zip(ligne, vecteur, strict=True)) for ligne in matrice
    )


def _inverse_de_deux(passage: Matrice) -> Matrice:
    """L'inverse d'une matrice d'ordre deux, par la formule des cofacteurs."""
    (a, b), (c, d) = passage
    determinant = a * d - b * c
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def changement_de_base(matrice: Matrice, passage: Matrice) -> Matrice:
    """`A' = P^-1 A P` : la meme application, lue dans la nouvelle base.

    La trace et le determinant ne bougent pas -- ce sont des invariants
    de similitude, et ils gardent le calcul.
    """
    if not est_inversible(passage):
        msg = "matrice de passage singuliere : ses colonnes ne forment pas une base"
        raise ValueError(msg)
    return produit(_inverse_de_deux(passage), produit(matrice, passage))
