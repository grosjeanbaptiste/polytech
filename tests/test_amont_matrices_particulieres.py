"""Mathematiques II, chapitre 10 d'algebre et exponentielle, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire2.toml`
rattache un titre portent ce que la planche enonce -- en recopiant ses
exemples, page citee.

Sources : `AlgebreLineaire2/SLIDES/Slides_10_Matrices_carrees_particulieres.pdf`
(« ch. 10 ») et `Slides_Exponentielle_d_une_matrice.pdf` (« exp. »). Les pages
sont celles du PDF.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import exponentielle, rotation_plane
from polytech.maths.domain.algebre.diagonalisation import (
    diagonaliser_orthogonalement,
    est_diagonalisable,
)
from polytech.maths.domain.algebre.matrices_particulieres import (
    est_symetrique,
    est_une_matrice_orthogonale,
)
from polytech.maths.domain.algebre.representation import matrice_de_passage
from polytech.maths.domain.series.mac_laurin import coefficients_de_mac_laurin

X, A, THETA = sp.symbols("x a theta", real=True)
ROTATION = sp.Matrix([[0, -1], [1, 0]])
ECHANGE = sp.Matrix([[0, 1], [1, 0]])


@pytest.mark.parametrize(
    "matrice",
    [sp.Matrix([[3, 4], [-4, 3]]) / 5, ROTATION, ECHANGE, -ROTATION, -ECHANGE],
    ids=["page 13", "R90", "S90", "R270", "S270"],
)
def test_les_matrices_orthogonales_des_pages_13_et_21(matrice: sp.Matrix) -> None:
    """Ch. 10, pages 13 et 21 : `A A^t = A^t A = I`, donc `det(A) = +-1`."""
    assert est_une_matrice_orthogonale(matrice)
    assert matrice.det() in (1, -1)


def test_une_matrice_orthogonale_n_est_pas_toujours_diagonalisable() -> None:
    """Ch. 10, pages 43 et 44 : `t2 - 1` se scinde sur R, `t2 + 1` non."""
    assert est_diagonalisable(ECHANGE)
    assert not est_diagonalisable(ROTATION)


def test_la_matrice_symetrique_de_la_page_32_se_diagonalise_orthogonalement() -> None:
    """Ch. 10, pages 32 et 41 : symetrique, donc `P^t A P` diagonale."""
    matrice = sp.Matrix([[1, 4], [4, 2]])
    assert est_symetrique(matrice)
    passage, diagonale = diagonaliser_orthogonalement(matrice)
    assert est_une_matrice_orthogonale(passage)
    assert sp.simplify(passage.T * matrice * passage - diagonale).is_zero_matrix


def test_le_passage_entre_deux_bases_orthonormales_est_orthogonal() -> None:
    """Ch. 10, page 25 : dans R, `P^-1 = P^t`."""
    canonique = (sp.Matrix([1, 0]), sp.Matrix([0, 1]))
    tournee = (sp.Matrix([3, 4]) / 5, sp.Matrix([-4, 3]) / 5)
    assert est_une_matrice_orthogonale(matrice_de_passage(canonique, tournee))


def test_les_developpements_usuels_de_la_page_2() -> None:
    """Exp., page 2 : `e^x`, `sin`, `cos`, `sinh`, `cosh` en series entieres."""
    f = sp.factorial
    assert coefficients_de_mac_laurin(sp.exp(X), 4) == (1, 1, 1 / f(2), 1 / f(3))
    assert coefficients_de_mac_laurin(sp.sin(X), 4) == (0, 1, 0, -1 / f(3))
    assert coefficients_de_mac_laurin(sp.cos(X), 4) == (1, 0, -1 / f(2), 0)
    assert coefficients_de_mac_laurin(sp.sinh(X), 4) == (0, 1, 0, 1 / f(3))
    assert coefficients_de_mac_laurin(sp.cosh(X), 4) == (1, 0, 1 / f(2), 0)


def test_l_exponentielle_d_une_matrice_diagonale_et_la_serie() -> None:
    """Exp., pages 3 et 4 : `e^A = I + A + A2/2! + ...`, `diag(e^a, e^b, e^c)`."""
    diagonale = sp.diag(1, 2, 3)
    attendue = sp.diag(sp.E, sp.E**2, sp.E**3)
    assert exponentielle.par_diagonalisation(diagonale) == attendue
    assert exponentielle.par_jordan(diagonale) == attendue
    tronquee = exponentielle.par_la_serie(sp.Matrix([[0, 1], [0, 0]]), 5)
    assert tronquee == sp.Matrix([[1, 1], [0, 1]])


def test_l_exponentielle_de_la_page_9_est_en_cosinus_hyperboliques() -> None:
    """Exp., page 9 : `A = [[0, a], [a, 0]]` donne `[[ch a, sh a], [sh a, ch a]]`."""
    obtenue = exponentielle.par_diagonalisation(sp.Matrix([[0, A], [A, 0]]))
    attendue = sp.Matrix([[sp.cosh(A), sp.sinh(A)], [sp.sinh(A), sp.cosh(A)]])
    assert sp.simplify((obtenue - attendue).rewrite(sp.exp)).is_zero_matrix


def test_la_rotation_de_la_page_5_et_ses_deux_coquilles() -> None:
    """Exp., page 5 : `e^A` est la rotation d'angle `theta` ; deux ecarts au cours.

    La planche ecrit `(-1)^k theta^(2k+1)` en haut a droite de `A^(2k+1)` : c'est
    `(-1)^(k+1)`, puisque `A` y porte `-theta`. Elle ecrit aussi la diagonale
    de `e^A` en `1 + theta2/2! + ...`, sans alternance : c'est le cosinus.
    """
    generatrice = rotation_plane.generatrice(THETA)
    assert rotation_plane.coin_haut_droit(generatrice, 1) == THETA**3
    for k in range(5):
        coin = rotation_plane.coin_haut_droit(generatrice, k)
        assert coin == rotation_plane.coin_corrige(THETA, k)
        assert coin != rotation_plane.coin_imprime(THETA, k)
    serie = exponentielle.par_la_serie(generatrice, 4)
    cosinus = coefficients_de_mac_laurin(sp.cos(X), 5)
    assert serie[0, 0] == sum(c * THETA**i for i, c in enumerate(cosinus))
    rotation = rotation_plane.rotation(THETA)
    obtenue = exponentielle.par_jordan(generatrice)
    assert rotation_plane.est_la_rotation(obtenue, THETA)
    assert not rotation_plane.est_la_rotation(rotation.T, THETA)  # sens inverse
