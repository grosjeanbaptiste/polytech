"""Le chapitre II du syllabus d'algebre : les operations sur les matrices.

II.1 Definitions, II.2 Operations -- addition, multiplication par un
scalaire, produit de deux matrices -- et II.3 Transposee. Rien ne les
portait : la bibliotheque avait les systemes, les determinants et le
spectre, mais pas les operations elles-memes.

Les verifications ne recopient pas les formules, elles emploient les
identites que ces operations satisfont -- et la plus instructive est
`(A B)^T = B^T A^T`, dont l'**inversion de l'ordre** est l'erreur
classique. Un produit mal transpose la romprait, et rien d'autre ne le
dirait.

sympy sert de second chemin : il calcule les memes produits par sa propre
implementation.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.matrices import (
    addition,
    identite,
    multiplication_par_un_scalaire,
    produit,
    transposee,
)

A = ((1.0, 2.0), (3.0, 4.0))
B = ((0.0, 1.0), (-1.0, 2.0))
RECTANGLE = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_l_addition_se_fait_terme_a_terme() -> None:
    """II.2.1 : l'addition n'est definie qu'entre matrices de meme format."""
    assert addition(A, B) == ((1.0, 3.0), (2.0, 6.0))


def test_la_multiplication_par_un_scalaire_multiplie_chaque_terme() -> None:
    """II.2.2, et c'est le point ou l'on confond avec le determinant.

    Multiplier une matrice `n x n` par `k` multiplie son determinant par
    `k^n`, non par `k`. Le test le verifie, parce que c'est l'erreur que
    la simplicite de l'operation invite a faire.
    """
    double = multiplication_par_un_scalaire(2.0, A)
    assert double == ((2.0, 4.0), (6.0, 8.0))
    assert sp.Matrix(double).det() == pytest.approx(4 * sp.Matrix(A).det())


def test_le_produit_concorde_avec_celui_de_sympy() -> None:
    """II.2.3, verifie par un second chemin sur trois formats."""
    for gauche, droite in ((A, B), (B, A), (RECTANGLE, transposee(RECTANGLE))):
        attendu = sp.Matrix(gauche) * sp.Matrix(droite)
        obtenu = sp.Matrix(produit(gauche, droite))
        assert obtenu == attendu


def test_le_produit_de_matrices_n_est_pas_commutatif() -> None:
    """Resultat du chapitre, et le contre-exemple suffit a l'etablir."""
    assert produit(A, B) != produit(B, A)


def test_l_identite_est_neutre_pour_le_produit() -> None:
    """Des deux cotes, ce qui n'est vrai que d'elle."""
    assert produit(A, identite(2)) == A
    assert produit(identite(2), A) == A


def test_la_transposee_de_la_transposee_rend_la_matrice() -> None:
    """II.3 : involution, comme le conjugue des complexes."""
    for matrice in (A, B, RECTANGLE):
        assert transposee(transposee(matrice)) == matrice


def test_la_transposee_d_un_produit_inverse_l_ordre() -> None:
    """`(A B)^T = B^T A^T` : l'identite qui porte le piege du chapitre.

    L'ordre s'inverse, et c'est ce qu'on oublie. Un produit mal transpose
    donnerait `A^T B^T`, qui n'a meme pas le bon format sur des matrices
    rectangulaires -- mais qui passerait inapercu sur des carrees.
    """
    assert transposee(produit(A, B)) == produit(transposee(B), transposee(A))


def test_la_transposee_change_le_format_d_une_rectangulaire() -> None:
    """Deux lignes et trois colonnes deviennent trois et deux."""
    assert len(RECTANGLE) == 2
    assert len(RECTANGLE[0]) == 3
    tournee = transposee(RECTANGLE)
    assert len(tournee) == 3
    assert len(tournee[0]) == 2


def test_une_addition_de_formats_differents_est_refusee() -> None:
    """L'addition n'est definie qu'a format egal."""
    with pytest.raises(ValueError, match="formats differents"):
        addition(A, RECTANGLE)


def test_un_produit_de_dimensions_incompatibles_est_refuse() -> None:
    """Les colonnes de gauche doivent egaler les lignes de droite."""
    with pytest.raises(ValueError, match="dimensions incompatibles"):
        produit(RECTANGLE, RECTANGLE)


def test_une_identite_d_ordre_nul_est_refusee() -> None:
    """La matrice vide n'est pas un cas du cours."""
    with pytest.raises(ValueError, match="ordre nul ou negatif"):
        identite(0)
