"""Le rang, le noyau et l'image -- chapitres VI a VIII du syllabus d'algebre.

Trois chapitres que rien ne portait : « Le rang d'une matrice », « Les
espaces vectoriels », « Les applications lineaires ». Le rang est ce qui
les relie, et c'est pourquoi un seul module suffit a leur part
calculable.

Deux resultats portent tout le reste.

**Le rang d'une matrice et celui de sa transposee sont egaux.** Il n'y a
rien d'evident la : le rang se definit par les lignes ou par les
colonnes, et c'est un theoreme que les deux donnent le meme nombre. C'est
ce qui autorise a parler du rang sans preciser lequel.

**Le theoreme du rang** -- `dim noyau + rang = nombre de colonnes`. Il
donne la dimension du noyau sans le calculer, donc le nombre de
parametres d'une solution de systeme. Sa forme utile est le critere de
Rouche : un systeme a une solution si et seulement si la colonne du
second membre ne releve pas le rang.

L'elimination est menee en flottants, avec un pivot choisi par module
maximal -- le pivot partiel. Un pivot naif sur la premiere ligne non
nulle donnerait un rang faux sur une matrice mal conditionnee, et le rang
est un entier : une erreur d'arrondi n'y est pas une imprecision, c'est
une faute.
"""

from __future__ import annotations

TOLERANCE_DU_PIVOT = 1e-10
"""En dessous, un pivot est tenu pour nul : le rang est un entier."""


def _exiger_la_matrice(matrice: tuple[tuple[float, ...], ...]) -> int:
    """Verifie la forme et rend le nombre de colonnes."""
    if not matrice:
        msg = "matrice vide : son rang n'est pas un cas du cours"
        raise ValueError(msg)
    colonnes = len(matrice[0])
    if any(len(ligne) != colonnes for ligne in matrice):
        msg = "lignes de longueurs differentes : ce n'est pas une matrice"
        raise ValueError(msg)
    return colonnes


def rang(matrice: tuple[tuple[float, ...], ...]) -> int:
    """Le nombre de pivots d'une elimination a pivot partiel."""
    colonnes = _exiger_la_matrice(matrice)
    lignes = [list(ligne) for ligne in matrice]
    pivots = 0
    for colonne in range(colonnes):
        choix = max(
            range(pivots, len(lignes)),
            key=lambda i: abs(lignes[i][colonne]),
            default=None,
        )
        if choix is None or abs(lignes[choix][colonne]) < TOLERANCE_DU_PIVOT:
            continue
        lignes[pivots], lignes[choix] = lignes[choix], lignes[pivots]
        pivot = lignes[pivots]
        for i in range(pivots + 1, len(lignes)):
            facteur = lignes[i][colonne] / pivot[colonne]
            lignes[i] = [a - facteur * b for a, b in zip(lignes[i], pivot, strict=True)]
        pivots += 1
    return pivots


def dimension_du_noyau(matrice: tuple[tuple[float, ...], ...]) -> int:
    """`colonnes - rang` : le theoreme du rang, lu dans le sens du probleme."""
    return _exiger_la_matrice(matrice) - rang(matrice)


def est_inversible(matrice: tuple[tuple[float, ...], ...]) -> bool:
    """Vrai si la matrice est carree et de rang plein.

    Le rang plein ne suffit pas : une rectangulaire a rang plein n'a pas
    d'inverse a deux cotes. Mais c'est le rang, et non le determinant,
    qui est le critere du chapitre -- parce qu'il s'etend aux
    rectangulaires, ou l'inversibilite n'a pas de sens.
    """
    colonnes = _exiger_la_matrice(matrice)
    return len(matrice) == colonnes and rang(matrice) == colonnes


def systeme_compatible(
    matrice: tuple[tuple[float, ...], ...],
    second_membre: tuple[float, ...],
) -> bool:
    """Le critere de Rouche : `rang(A) = rang(A|b)`.

    Il dit qu'une solution existe sans la calculer, et c'est la forme
    sous laquelle le theoreme du rang sert vraiment.
    """
    _exiger_la_matrice(matrice)
    if len(second_membre) != len(matrice):
        msg = (
            f"second membre de taille {len(second_membre)} pour "
            f"{len(matrice)} equations"
        )
        raise ValueError(msg)
    augmentee = tuple(
        (*ligne, valeur) for ligne, valeur in zip(matrice, second_membre, strict=True)
    )
    return rang(matrice) == rang(augmentee)
