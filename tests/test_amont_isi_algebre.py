"""Introduction aux sciences de l'ingenieur : systemes, determinants, complexes.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles de `polytech.maths` auxquels
`cursus/notions/IntroductionSciencesIngenieur.toml` rattache un titre portent
ce que le syllabus enonce -- en recopiant ses exemples, page citee.

Source : `IntroductionSciencesIngenieur/SYLLABUS/`
`Syllabus_Introduction_aux_sciences_de_l_ingenieur_2025_2026.pdf`, chapitres
A, F et O. Les pages sont celles du PDF ; la page imprimee est entre
parentheses.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import complexes, complexes_exacts
from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.polynomes_exacts import ordre_de_la_racine
from polytech.maths.domain.algebre.resolution import par_echelonnement_simple
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau
from polytech.maths.domain.algebre.systemes import SystemeLineaire

COEFFICIENTS = sp.Matrix([[2, -3, 1, 0, -3], [-1, 2, -3, 1, -2], [-2, 2, -3, -2, 4]])
SECOND_MEMBRE = sp.Matrix([-1, 2, 3])
"""Page 11 (6) : le systeme de trois equations a cinq inconnues."""

TROIS = sp.Matrix([[1, -1, 2], [0, -1, 3], [1, -1, 4]])
"""Page 56 (51) : l'exemple de l'inverse par les cofacteurs."""


def test_l_elimination_de_gauss_de_la_page_11() -> None:
    """Page 11 (6) : `x1 = -1 - 3x4 + 6x5`, `x2 = -4/7 - 2x4 + 19/7 x5`, ..."""
    solutions = par_echelonnement_simple(SystemeLineaire(COEFFICIENTS, SECOND_MEMBRE))
    septieme = sp.Rational(1, 7)
    assert solutions.particuliere == (-1, -4 * septieme, -5 * septieme, 0, 0)
    assert set(solutions.directions) == {
        (-3, -2, 0, 1, 0),
        (6, 19 * septieme, -6 * septieme, 0, 1),
    }


def test_les_solutions_du_systeme_homogene_de_la_page_12() -> None:
    """Page 12 (7) : `lambda (-3, -2, 0, 1, 0) + mu (6, 19/7, -6/7, 0, 1)`."""
    base = base_du_noyau(COEFFICIENTS)
    assert len(base) == 2
    for direction in base:
        assert (COEFFICIENTS * sp.Matrix(direction)).is_zero_matrix


def test_le_determinant_et_l_inverse_de_la_page_56() -> None:
    """Page 56 (51) : `det A = -2`, et `A^-1` par la matrice des cofacteurs."""
    determinant = Determinant(TROIS)
    assert {determinant.par_cofacteurs(ligne) for ligne in range(3)} == {-2}
    inverse = determinant.inverse()
    assert inverse == sp.Matrix([[-1, 2, -1], [3, 2, -3], [1, 0, -1]]) / -2
    assert TROIS * inverse == sp.eye(3)


def test_le_determinant_d_une_somme_n_est_pas_la_somme() -> None:
    """Page 56 (51), propriete 5 : 2 et 1, mais `det(A + B) = 1`."""
    a, b = sp.Matrix([[1, 1], [-1, 1]]), sp.Matrix([[0, -1], [1, 0]])
    assert Determinant(a).par_cofacteurs(0) == 2
    assert Determinant(b).par_cofacteurs(0) == 1
    assert Determinant(a + b).par_cofacteurs(0) == 1


def test_le_determinant_d_un_produit_est_le_produit() -> None:
    """Page 57 (52), Binet-Cauchy : -2 et 1 donnent `det(AB) = -2`."""
    a, b = sp.Matrix([[1, 2], [3, 4]]), sp.Matrix([[0, -1], [1, 2]])
    assert a * b == sp.Matrix([[2, 3], [4, 5]])
    assert Determinant(a * b).par_cofacteurs(0) == -2
    rectangulaire = sp.Matrix([[-1, 0], [0, 1], [1, 2]]) * sp.Matrix(
        [[1, 2, 3], [0, 1, 2]]
    )
    assert Determinant(rectangulaire).par_cofacteurs(0) == 0


def test_le_module_le_conjugue_et_la_forme_exponentielle() -> None:
    """Pages 145 et 146 (140 et 141) : `z z_barre = |z|^2`, `z = |z| e^(i arg z)`."""
    z = 3 + 4j
    assert z * complexes.conjugue(z) == pytest.approx(complexes.module(z) ** 2)
    module, angle = complexes.forme_exponentielle(z)
    assert module * complex(sp.exp(sp.I * angle)) == pytest.approx(z)
    assert complexes_exacts.forme_polaire(sp.Integer(-1)) == (1, sp.pi)


def test_les_trois_racines_cubiques_de_l_unite() -> None:
    """Page 147 (142), exemple 2 : `1` et `-1/2 +- i racine(3)/2`."""
    demi = sp.Rational(1, 2)
    attendues = {1, -demi + sp.sqrt(3) * sp.I / 2, -demi - sp.sqrt(3) * sp.I / 2}
    obtenues = {sp.expand_complex(r) for r in complexes_exacts.racines_de_l_unite(3)}
    assert obtenues == attendues
    assert {sp.expand_complex(r**3) for r in obtenues} == {1}


def test_la_multiplicite_des_racines_de_la_page_148() -> None:
    """Page 148 (143) : `(z - 1)^6 (z - i)^3 (z + 1 - i)^2`, multiplicites 6, 3, 2."""
    z = sp.Symbol("z")
    polynome = (z - 1) ** 6 * (z - sp.I) ** 3 * (z + 1 - sp.I) ** 2
    assert ordre_de_la_racine(polynome, sp.Integer(1), z) == 6
    assert ordre_de_la_racine(polynome, sp.I, z) == 3
    assert ordre_de_la_racine(polynome, -1 + sp.I, z) == 2
