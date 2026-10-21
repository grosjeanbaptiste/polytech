"""Syllabus d'algebre IX.6 : resoudre `A X = proj_W b`, colonnes liees comprises.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 148 a 150
(imprimees) : `A X = b` n'a de solution que si `b` est dans le sous-espace
`W` engendre par les colonnes ; sinon on resout `A X = proj_W b`, et
`b - proj_W b` est orthogonal a `W`. La projection est unique meme quand
les colonnes sont liees : seule la solution cesse de l'etre. Exercice 30
(page 150) : la solution est `(1/3, 1/3, 1/3)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.moindres_carres import (
    au_sens_des_moindres_carres,
    projection_et_residu,
)

R = sp.Rational
A30 = sp.Matrix([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]])
B30 = sp.Matrix([1, 0, 1, 1])


def test_la_projection_de_l_exercice_30_est_atteinte_par_sa_solution() -> None:
    projete, _ = projection_et_residu(A30, B30)
    solution = sp.Matrix(au_sens_des_moindres_carres(A30, B30))
    assert solution == sp.Matrix([R(1, 3), R(1, 3), R(1, 3)])
    assert A30 * solution == projete


def test_le_residu_est_orthogonal_aux_colonnes() -> None:
    _, residu = projection_et_residu(A30, B30)
    assert A30.T * residu == sp.zeros(3, 1)


def test_projection_et_residu_redonnent_le_second_membre() -> None:
    projete, residu = projection_et_residu(A30, B30)
    assert projete + residu == B30


def test_des_colonnes_liees_ont_encore_une_projection() -> None:
    """La troisieme colonne est la somme des deux premieres."""
    liees = sp.Matrix([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
    projete, residu = projection_et_residu(liees, sp.Matrix([2, 3, 4]))
    assert (projete, residu) == (sp.Matrix([2, 3, 0]), sp.Matrix([0, 0, 4]))


def test_un_second_membre_dans_w_a_un_residu_nul() -> None:
    projete, residu = projection_et_residu(A30, A30 * sp.Matrix([1, 2, 3]))
    assert residu == sp.zeros(4, 1)
    assert projete == A30 * sp.Matrix([1, 2, 3])


def test_une_matrice_nulle_projette_sur_zero() -> None:
    projete, residu = projection_et_residu(sp.zeros(2, 2), sp.Matrix([1, 1]))
    assert (projete, residu) == (sp.zeros(2, 1), sp.Matrix([1, 1]))
