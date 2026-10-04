"""Mathematique, exercices 2, seance III : les systemes d'equations differentielles.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 16
a 19. Les rappels theoriques donnent deux methodes pour `y' = A y + x`,
`y(0) = c` -- la transformee de Laplace, `Y = (pI - A)^-1 (c + X)`, et
l'exponentielle de matrice, `y = e^(At) c + int e^(A(t - tau)) x(tau)`.
Les reponses des exercices III.1 a III.5 sont les oracles.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.systemes import SystemeDifferentiel
from polytech.maths.domain.laplace.systemes import resoudre_le_systeme
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
C1, C2 = sp.symbols("C_1 C_2", real=True)
exp, cos, sin, sqrt = sp.exp, sp.cos, sp.sin, sp.sqrt


def _nulle(matrice: sp.Matrix) -> bool:
    return bool(sp.simplify(matrice) == sp.zeros(*matrice.shape))


def _iii_1() -> SystemeDifferentiel:
    return SystemeDifferentiel(sp.Matrix([[-3, sqrt(2)], [sqrt(2), -2]]))


def _iii_2() -> SystemeDifferentiel:
    return SystemeDifferentiel(sp.Matrix([[1, -1], [-4, 1]]))


def _iii_3() -> SystemeDifferentiel:
    return SystemeDifferentiel(sp.Matrix([[-7, 1], [-2, -5]]))


def test_les_modes_propres_de_l_exercice_iii_1() -> None:
    """`y = xi e^(rt)` : `r = -1` avec `(1, sqrt 2)`, `r = -4` avec `(-sqrt 2, 1)`."""
    modes = dict(_iii_1().modes_propres())
    assert set(modes) == {-1, -4}
    assert modes[-1].row_join(sp.Matrix([1, sqrt(2)])).rank() == 1
    assert modes[-4].row_join(sp.Matrix([-sqrt(2), 1])).rank() == 1


def test_chaque_mode_propre_est_une_solution() -> None:
    """`(xi e^(rt))' = A xi e^(rt)`, c'est-a-dire `A xi = r xi`."""
    systeme = _iii_1()
    for r, xi in systeme.modes_propres():
        solution = xi * exp(r * T)
        assert _nulle(sp.diff(solution, T) - systeme.matrice * solution)


def test_le_wronskien_de_l_exercice_iii_1() -> None:
    """`|e^(-t), -sqrt2 e^(-4t); sqrt2 e^(-t), e^(-4t)| = 3 e^(-5t)`, jamais nul."""
    fondamentale = sp.Matrix(
        [[exp(-T), -sqrt(2) * exp(-4 * T)], [sqrt(2) * exp(-T), exp(-4 * T)]]
    )
    assert sp.simplify(fondamentale.det() - 3 * exp(-5 * T)) == 0
    assert sp.simplify(_iii_1().wronskien(fondamentale) - 3 * exp(-5 * T)) == 0


def test_un_wronskien_nul_trahit_deux_solutions_liees() -> None:
    liees = sp.Matrix(
        [[exp(-T), 2 * exp(-T)], [sqrt(2) * exp(-T), 2 * sqrt(2) * exp(-T)]]
    )
    assert _iii_1().wronskien(liees) == 0


def test_une_matrice_qui_n_est_pas_faite_de_solutions_n_a_pas_de_wronskien() -> None:
    with pytest.raises(ValueError, match="ne sont pas des solutions"):
        _iii_1().wronskien(sp.Matrix([[exp(T), 0], [0, exp(T)]]))


def test_la_solution_de_l_exercice_iii_2_par_la_transformee_de_laplace() -> None:
    """`Y = (pI - A)^-1 c` ; valeurs propres -1 et 3."""
    attendue = sp.Matrix(
        [
            (C1 / 2 + C2 / 4) * exp(-T) + (C1 / 2 - C2 / 4) * exp(3 * T),
            (C1 + C2 / 2) * exp(-T) + (-C1 + C2 / 2) * exp(3 * T),
        ]
    )
    assert _nulle(_iii_2().par_laplace(sp.Matrix([C1, C2])) - attendue)


def test_la_solution_de_l_exercice_iii_2_par_l_exponentielle_de_matrice() -> None:
    """`y = e^(At) c` : la meme fonction, par l'autre methode des rappels."""
    systeme = _iii_2()
    conditions = sp.Matrix([C1, C2])
    par_exponentielle = systeme.exponentielle() * conditions
    assert _nulle(par_exponentielle - systeme.par_laplace(conditions))
    assert systeme.exponentielle().subs(T, 0) == sp.eye(2)


def test_l_exponentielle_de_matrice_verifie_son_equation() -> None:
    """`d/dt e^(At) = A e^(At)`."""
    systeme = _iii_2()
    exponentielle = systeme.exponentielle()
    assert _nulle(sp.diff(exponentielle, T) - systeme.matrice * exponentielle)


def test_la_solution_de_l_exercice_iii_3_a_des_valeurs_propres_complexes() -> None:
    """`r = -6 +- j` : `e^(-6t) (C1 cos t + (C2 - C1) sin t)`, et de meme pour `y2`."""
    attendue = sp.Matrix(
        [
            exp(-6 * T) * (C1 * cos(T) + (C2 - C1) * sin(T)),
            exp(-6 * T) * (C2 * cos(T) + (C2 - 2 * C1) * sin(T)),
        ]
    )
    systeme = _iii_3()
    assert set(dict(systeme.modes_propres())) == {-6 + sp.I, -6 - sp.I}
    assert _nulle(systeme.par_laplace(sp.Matrix([C1, C2])) - attendue)
    assert _nulle(systeme.exponentielle() * sp.Matrix([C1, C2]) - attendue)


def test_un_systeme_non_homogene_par_les_deux_methodes() -> None:
    """Rappels, page 16 : `e^(At) c + int_0^t e^(A(t - tau)) x(tau) d tau`.

    `y1' = -y1 + 1`, `y2' = -2 y2`, depuis `(0, 3)` : `(1 - e^(-t), 3 e^(-2t))`.
    """
    systeme = SystemeDifferentiel(sp.Matrix([[-1, 0], [0, -2]]))
    conditions, force = sp.Matrix([0, 3]), sp.Matrix([1, 0])
    attendue = sp.Matrix([1 - exp(-T), 3 * exp(-2 * T)])
    assert _nulle(systeme.par_exponentielle(conditions, force) - attendue)
    assert _nulle(systeme.par_laplace(conditions, force) - attendue)


def test_une_matrice_non_carree_ne_definit_pas_un_systeme() -> None:
    with pytest.raises(ValueError, match="matrice carree attendue"):
        SystemeDifferentiel(sp.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_le_systeme_du_second_ordre_de_l_exercice_iii_4() -> None:
    """`x'' + y' + x = e^t`, `x' + y'' = 1`, conditions initiales nulles."""
    x, y = sp.Function("x")(T), sp.Function("y")(T)
    equations = (
        sp.Eq(x.diff(T, 2) + y.diff(T) + x, exp(T)),
        sp.Eq(x.diff(T) + y.diff(T, 2), 1),
    )
    solution = resoudre_le_systeme(equations, {x: (0, 0), y: (0, 0)})
    assert sp.simplify(solution[x] - (-(T**3) / 6 - T - 1 + exp(T))) == 0
    assert sp.simplify(solution[y] - (T**4 / 24 + T**2 + T + 1 - exp(T))) == 0


def test_le_systeme_de_l_exercice_iii_5() -> None:
    """`x' + x + 3y' + 7y = H(t)`, `-2x + y' + y = e^(-t)`, `x(0) = y(0) = 0`."""
    x, y = sp.Function("x")(T), sp.Function("y")(T)
    equations = (
        sp.Eq(x.diff(T) + x + 3 * y.diff(T) + 7 * y, 1),
        sp.Eq(-2 * x + y.diff(T) + y, exp(-T)),
    )
    solution = resoudre_le_systeme(equations, {x: (0,), y: (0,)})
    en_x = R(1, 15) - exp(-T) / 2 - exp(-3 * T) / 6 + R(3, 5) * exp(-5 * T)
    en_y = R(2, 15) + exp(-3 * T) / 6 - R(3, 10) * exp(-5 * T)
    assert sp.simplify(solution[x] - en_x) == 0
    assert sp.simplify(solution[y] - en_y) == 0


def test_la_solution_rendue_verifie_chaque_equation_du_systeme() -> None:
    """Le controle ne depend pas de l'inversion : on reporte dans III.4."""
    x, y = sp.Function("x")(T), sp.Function("y")(T)
    equations = (
        sp.Eq(x.diff(T, 2) + y.diff(T) + x, exp(T)),
        sp.Eq(x.diff(T) + y.diff(T, 2), 1),
    )
    solution = resoudre_le_systeme(equations, {x: (0, 0), y: (0, 0)})
    for equation in equations:
        reste = (equation.lhs - equation.rhs).subs(solution).doit()
        assert sp.simplify(reste) == 0


def test_il_faut_autant_de_conditions_que_l_ordre_de_chaque_inconnue() -> None:
    x, y = sp.Function("x")(T), sp.Function("y")(T)
    equations = (sp.Eq(x.diff(T, 2) + y, 0), sp.Eq(y.diff(T) + x, 0))
    with pytest.raises(ValueError, match="conditions initiales pour"):
        resoudre_le_systeme(equations, {x: (0,), y: (0,)})


def test_les_conditions_initiales_non_nulles_entrent_dans_le_systeme_transforme() -> (
    None
):
    """L'exercice III.2 repris equation par equation : `y1(0) = C1`, `y2(0) = C2`."""
    y1, y2 = sp.Function("y_1")(T), sp.Function("y_2")(T)
    equations = (sp.Eq(y1.diff(T), y1 - y2), sp.Eq(y2.diff(T), -4 * y1 + y2))
    solution = resoudre_le_systeme(equations, {y1: (C1,), y2: (C2,)})
    par_la_matrice = _iii_2().par_laplace(sp.Matrix([C1, C2]))
    assert sp.simplify(solution[y1] - par_la_matrice[0]) == 0
    assert sp.simplify(solution[y2] - par_la_matrice[1]) == 0
