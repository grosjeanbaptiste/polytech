"""Mathematique, exercices 1, section 2 : matrices, determinants, systemes.

Source : `Mathematique/EXS/Exs_01_Integration_matrices_determinants.pdf`,
pages 5 et 6. Les reponses officielles sont les oracles : `det A = -1 -
a^3`, la comatrice et l'inverse de la page 5 ; les solutions `(0, 0, 0)`,
`(3, 2, 0)` et la solution parametree de la page 6 ; les conditions
`a = b ou b = c ou c = a` et `2abc + bc + ca + ab = 1`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.methodes import par_cramer, par_l_inverse
from polytech.maths.domain.algebre.solutions import (
    Nature,
    nature_des_solutions,
    solution_generale,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

A, B, C = sp.symbols("a b c", real=True)
MATRICE = sp.Matrix([[1, 0, A], [0, A, 1], [A, 1, 0]])
COMATRICE = sp.Matrix([[-1, A, -(A**2)], [A, -(A**2), -1], [-(A**2), -1, A]])


def _homogene(lignes: list[list[sp.Expr]]) -> SystemeLineaire:
    return SystemeLineaire(sp.Matrix(lignes), sp.zeros(3, 1))


def _premier_parametrique() -> SystemeLineaire:
    """Page 6, 2.3.1."""
    return _homogene([[1, 1, 1], [B + C, C + A, A + B], [B * C, A * C, A * B]])


def _second_parametrique() -> SystemeLineaire:
    """Page 6, 2.3.2 : `x - a(y + z) = 0`, et de meme en `b` et `c`."""
    return _homogene([[1, -A, -A], [-B, 1, -B], [-C, -C, 1]])


def test_le_determinant_de_la_page_5() -> None:
    """`det A = |a 1; 1 0| + a |0 a; a 1| = -1 - a^3`."""
    determinant = Determinant(MATRICE)
    assert sp.expand(determinant.par_cofacteurs(0)) == -1 - A**3
    assert determinant.termes_du_developpement(0) == (-1, 0, -(A**3))


def test_la_matrice_est_inversible_sauf_pour_a_egal_moins_un() -> None:
    """`1 + a^3 != 0`, « ce qui equivaut a `a != -1` car `a` est reel »."""
    assert Determinant(MATRICE).valeurs_singulieres(A) == (-1,)


def test_la_comatrice_de_la_page_5() -> None:
    """Elle est symetrique : la comatrice et sa transposee coincident."""
    determinant = Determinant(MATRICE)
    assert determinant.comatrice() == COMATRICE
    assert determinant.comatrice() == determinant.adjointe()
    assert COMATRICE.T == COMATRICE


def test_le_produit_par_la_transposee_de_la_comatrice_rend_le_determinant() -> None:
    """Page 5 : `A tA~ = tA~ A = (-1 - a^3) I_3`."""
    attendu = (-1 - A**3) * sp.eye(3)
    assert sp.expand(MATRICE * COMATRICE.T) == attendu
    assert sp.expand(COMATRICE.T * MATRICE) == attendu


def test_l_inverse_de_la_page_5() -> None:
    """`A^-1 = A~ / (-1 - a^3)`."""
    inverse = Determinant(MATRICE).inverse()
    assert sp.simplify(inverse - COMATRICE / (-1 - A**3)) == sp.zeros(3, 3)
    assert sp.simplify(inverse * MATRICE) == sp.eye(3)


def test_une_matrice_singuliere_n_a_pas_d_inverse() -> None:
    """Pour `a = -1` le determinant s'annule."""
    with pytest.raises(ValueError, match="determinant nul"):
        Determinant(MATRICE.subs(A, -1)).inverse()


def test_le_premier_systeme_non_parametrique_n_a_que_la_solution_nulle() -> None:
    """Page 6, 2.2.1 : `(0, 0, 0)`."""
    systeme = _homogene([[1, 1, -1], [1, -1, 0], [1, 4, 1]])
    assert nature_des_solutions(systeme) is Nature.UNIQUE
    assert solution_generale(systeme) == (0, 0, 0)
    assert not systeme.admet_des_solutions_non_nulles()


def test_le_deuxieme_systeme_non_parametrique() -> None:
    """Page 6, 2.2.2 : `(3, 2, 0)`, par deux chemins."""
    systeme = SystemeLineaire(
        sp.Matrix([[1, 1, 2], [1, -1, -1], [1, 0, 1]]), sp.Matrix([5, 1, 3])
    )
    assert par_cramer(systeme) == sp.Matrix([3, 2, 0])
    assert par_l_inverse(systeme) == sp.Matrix([3, 2, 0])


def test_le_troisieme_systeme_non_parametrique() -> None:
    """Page 6, 2.2.3 : `(8a + 5b - c, -2a + b + 7c, -4a - 7b + 5c) / 18`."""
    systeme = SystemeLineaire(
        sp.Matrix([[3, -1, 2], [-1, 2, -3], [1, 2, 1]]), sp.Matrix([A, B, C])
    )
    attendu = (
        sp.Matrix([8 * A + 5 * B - C, -2 * A + B + 7 * C, -4 * A - 7 * B + 5 * C]) / 18
    )
    assert sp.simplify(par_cramer(systeme) - attendu) == sp.zeros(3, 1)
    assert sp.simplify(systeme.matrice * attendu - systeme.second_membre) == sp.zeros(
        3, 1
    )


def test_la_condition_du_premier_systeme_parametrique() -> None:
    """Page 6, 2.3.1 : des solutions non nulles si `a = b` ou `b = c` ou `c = a`."""
    condition = Determinant(_premier_parametrique().matrice).condition_de_singularite()
    produit = (A - B) * (B - C) * (C - A)
    assert (
        sp.simplify(condition.lhs - produit) == 0
        or sp.simplify(condition.lhs + produit) == 0
    )
    for egalite in ({A: B}, {B: C}, {C: A}):
        assert _premier_parametrique().matrice.subs(egalite).det() == 0


def test_la_condition_du_second_systeme_parametrique() -> None:
    """Page 6, 2.3.2 : `2abc + bc + ca + ab = 1`."""
    determinant = _second_parametrique().matrice.det()
    assert sp.expand(determinant - (1 - (2 * A * B * C + B * C + C * A + A * B))) == 0
    valeurs = {A: 1, B: 1, C: 0}
    assert (2 * A * B * C + B * C + C * A + A * B).subs(valeurs) == 1
    assert _second_parametrique().matrice.subs(valeurs).det() == 0


def test_hors_de_la_condition_il_n_y_a_que_la_solution_nulle() -> None:
    """`a = 1`, `b = 2`, `c = 3` : les trois sont distincts."""
    systeme = SystemeLineaire(
        _premier_parametrique().matrice.subs({A: 1, B: 2, C: 3}), sp.zeros(3, 1)
    )
    assert not systeme.admet_des_solutions_non_nulles()


def test_les_solutions_non_nulles_quand_la_condition_est_remplie() -> None:
    """« Expliciter ces solutions » : pour `a = b`, elles verifient le systeme."""
    matrice = _premier_parametrique().matrice.subs(B, A)
    systeme = SystemeLineaire(matrice, sp.zeros(3, 1))
    assert systeme.admet_des_solutions_non_nulles()
    solution = sp.Matrix(solution_generale(systeme))
    assert solution != sp.zeros(3, 1)
    assert sp.simplify(matrice * solution) == sp.zeros(3, 1)


def test_la_comatrice_d_une_matrice_non_symetrique_n_est_pas_sa_transposee() -> None:
    """`A = [[1, 2], [3, 4]]` : `A~ = [[4, -3], [-2, 1]]` ; c'est `tA~` qui inverse."""
    determinant = Determinant(sp.Matrix([[1, 2], [3, 4]]))
    assert determinant.comatrice() == sp.Matrix([[4, -3], [-2, 1]])
    assert determinant.inverse() == sp.Matrix([[4, -2], [-3, 1]]) / -2
    assert determinant.inverse() * determinant.matrice == sp.eye(2)
