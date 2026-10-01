"""Les chapitres VI a VIII du syllabus d'algebre : rang, noyau, image.

Trois chapitres que rien ne portait -- « Le rang d'une matrice », « Les
espaces vectoriels », « Les applications lineaires ». Le rang est ce qui
les relie : il dit la dimension de l'image, donc celle du noyau par le
theoreme du rang, donc si un systeme a une solution et combien.

Les verifications passent par **sympy**, qui calcule le rang par sa
propre elimination, et par les identites que le rang satisfait : le
theoreme du rang lui-meme, l'egalite des rangs d'une matrice et de sa
transposee -- qui n'est pas evidente et qui est le resultat du chapitre --
et le critere de compatibilite d'un systeme.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.rang import (
    dimension_du_noyau,
    est_inversible,
    rang,
    systeme_compatible,
)

CARREE_INVERSIBLE = ((2.0, 1.0), (1.0, 1.0))
CARREE_SINGULIERE = ((1.0, 2.0), (2.0, 4.0))
RECTANGLE = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (1.0, 1.0, 1.0))


def test_le_rang_concorde_avec_celui_de_sympy() -> None:
    """Chemin independant : sympy mene sa propre elimination."""
    for matrice in (CARREE_INVERSIBLE, CARREE_SINGULIERE, RECTANGLE):
        assert rang(matrice) == sp.Matrix(matrice).rank()


def test_le_rang_d_une_matrice_et_de_sa_transposee_sont_egaux() -> None:
    """Resultat du chapitre, et il n'est pas evident.

    Le rang se definit par les lignes ou par les colonnes, et les deux
    definitions donnent le meme nombre. C'est ce qui autorise a parler du
    rang sans preciser lequel, et un calcul qui ne regarderait que les
    lignes se trahirait ici.
    """
    for matrice in (CARREE_INVERSIBLE, CARREE_SINGULIERE, RECTANGLE):
        transposee = tuple(zip(*matrice, strict=True))
        assert rang(matrice) == rang(transposee)


def test_le_theoreme_du_rang_relie_noyau_et_image() -> None:
    """`dim noyau + rang = nombre de colonnes`.

    C'est l'identite centrale des trois chapitres : connaitre le rang
    donne la dimension du noyau sans le calculer, donc le nombre de
    parametres d'une solution de systeme.
    """
    for matrice in (CARREE_INVERSIBLE, CARREE_SINGULIERE, RECTANGLE):
        colonnes = len(matrice[0])
        assert dimension_du_noyau(matrice) + rang(matrice) == colonnes


def test_la_dimension_du_noyau_concorde_avec_sympy() -> None:
    """Et sympy la calcule en resolvant, non par le theoreme du rang."""
    for matrice in (CARREE_INVERSIBLE, CARREE_SINGULIERE, RECTANGLE):
        assert dimension_du_noyau(matrice) == len(sp.Matrix(matrice).nullspace())


def test_une_matrice_est_inversible_si_son_rang_est_plein() -> None:
    """Et le rang plein est le seul critere : ni le determinant, ni la taille.

    Les deux coincident pour une carree, mais le rang s'etend aux
    rectangles -- ou l'inversibilite n'a pas de sens -- et c'est pourquoi
    le chapitre le prefere.
    """
    assert est_inversible(CARREE_INVERSIBLE)
    assert not est_inversible(CARREE_SINGULIERE)
    assert not est_inversible(RECTANGLE)


def test_une_matrice_rectangulaire_n_est_jamais_inversible() -> None:
    """Meme a rang plein : il n'y a pas d'inverse a deux cotes."""
    large = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert rang(large) == 2
    assert not est_inversible(large)


def test_un_systeme_est_compatible_si_la_colonne_ne_releve_pas_le_rang() -> None:
    """Le critere de Rouche : `rang(A) = rang(A|b)`.

    C'est la forme utile du theoreme du rang : on compare le rang de la
    matrice a celui de la matrice augmentee, et l'egalite dit qu'une
    solution existe -- sans la calculer.
    """
    assert systeme_compatible(CARREE_SINGULIERE, (1.0, 2.0))
    assert not systeme_compatible(CARREE_SINGULIERE, (1.0, 3.0))


def test_un_systeme_a_matrice_inversible_est_toujours_compatible() -> None:
    """A rang plein, aucune colonne ne peut relever le rang."""
    for second in ((0.0, 0.0), (1.0, 7.0), (-3.0, 2.5)):
        assert systeme_compatible(CARREE_INVERSIBLE, second)


def test_le_rang_d_une_matrice_nulle_est_nul() -> None:
    """Cas limite, et son noyau est l'espace entier."""
    nulle = ((0.0, 0.0), (0.0, 0.0))
    assert rang(nulle) == 0
    assert dimension_du_noyau(nulle) == 2


def test_une_matrice_vide_est_refusee() -> None:
    """Le rang d'une matrice sans ligne n'est pas un cas du cours."""
    with pytest.raises(ValueError, match="matrice vide"):
        rang(())


def test_des_lignes_de_longueurs_differentes_sont_refusees() -> None:
    """Ce n'est pas une matrice, et l'elimination y produirait du bruit."""
    with pytest.raises(ValueError, match="lignes de longueurs differentes"):
        rang(((1.0, 2.0), (3.0,)))


def test_un_second_membre_de_mauvaise_taille_est_refuse() -> None:
    """Autant de composantes que d'equations, sinon le systeme n'existe pas."""
    with pytest.raises(ValueError, match="second membre de taille"):
        systeme_compatible(CARREE_SINGULIERE, (1.0, 2.0, 3.0))
