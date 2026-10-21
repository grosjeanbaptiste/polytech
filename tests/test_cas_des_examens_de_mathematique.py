"""Les cas particuliers que la fiche de « Mathematique » reprend, refaits.

Chaque cas de `cursus/fiches/Mathematique.toml` qui se calcule l'est ici,
par la bibliotheque ou par sympy, et recoupe par un second chemin : la
substitution dans l'equation, le rang d'une matrice, une quadrature.
"""

from __future__ import annotations

import math

import mpmath
import sympy as sp

from polytech.maths.domain.algebre.discussion_par_le_rang import (
    Nature,
    discuter_par_le_rang,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

x, t, m = sp.symbols("x t m")


def test_janvier_2022_2_compatible_pour_m_egal_moins_2_seulement() -> None:
    a = sp.Matrix([[1, 1, -1], [3, 1, -1], [1, -2, 2]])
    assert a.det() == 0
    assert a.rank() == 2
    augmentee = a.row_join(sp.Matrix([1, 1, m]))
    rangs = {v: augmentee.subs(m, v).rank() for v in (-3, -2, 0, 1, 7)}
    assert rangs == {-3: 3, -2: 2, 0: 3, 1: 3, 7: 3}
    systeme = SystemeLineaire(a, sp.Matrix([1, 1, m]))
    discussion = discuter_par_le_rang(systeme, m)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {sp.Integer(-2): Nature.INDETERMINE}


def test_janvier_2021_4_la_recurrence_des_integrales() -> None:
    def i(n: int) -> sp.Expr:
        return sp.integrate(1 / (x**2 + 1) ** n, (x, 0, 1))

    for n in (1, 2, 3, 4):
        assert (
            sp.simplify(2 * n * i(n + 1) - (2 * n - 1) * i(n) - sp.Rational(1, 2**n))
            == 0
        )


def test_janvier_2022_1_l_integrale_par_parties() -> None:
    exacte = sp.integrate(sp.log(1 + t) / t**2, (t, 1, 2))
    assert sp.simplify(exacte - sp.log(8 * sp.sqrt(3) / 9)) == 0
    numerique = mpmath.quad(lambda u: mpmath.log(1 + u) / u**2, [1, 2])
    assert math.isclose(float(numerique), math.log(8 * math.sqrt(3) / 9), rel_tol=1e-12)
    assert sp.apart(1 / (t * (1 + t)), t) == 1 / t - 1 / (t + 1)


def test_janvier_2025_1_le_facteur_sans_racine_reelle() -> None:
    assert sp.factor(x**4 - 1) == (x - 1) * (x + 1) * (x**2 + 1)
    attendue = sp.log(sp.Rational(3, 2)) + 2 * sp.atan(3) - 2 * sp.atan(2)
    numerique = mpmath.quad(lambda u: 4 * u**2 / (u**4 - 1), [2, 3])
    assert math.isclose(float(numerique), float(attendue), rel_tol=1e-12)


def test_janvier_2021_3_la_famille_des_solutions_de_y_prime_moins_2ty() -> None:
    c = sp.Symbol("C")
    y = c * sp.exp(t**2) - sp.Rational(1, 2)
    assert sp.simplify(sp.diff(y, t) - 2 * t * y - t) == 0


def test_janvier_2022_3_la_particuliere_est_un_polynome() -> None:
    y = x**2 / 3 + 8 * x / 9 + sp.Rational(26, 27) - sp.exp(x) + sp.exp(3 * x) / 27
    assert sp.simplify(sp.diff(y, x, 2) - 4 * sp.diff(y, x) + 3 * y - x**2) == 0
    assert y.subs(x, 0) == 0
    assert sp.diff(y, x).subs(x, 0) == 0


def test_janvier_2025_3_les_conditions_initiales_non_nulles() -> None:
    y = t**2 + 2 + sp.exp(7 * t) - 3 * sp.exp(-2 * t)
    second = -14 * t**2 - 10 * t - 26
    assert sp.simplify(sp.diff(y, t, 2) - 5 * sp.diff(y, t) - 14 * y - second) == 0
    assert (y.subs(t, 0), sp.diff(y, t).subs(t, 0)) == (0, 13)
    assert set(sp.roots(sp.Symbol("r") ** 2 - 5 * sp.Symbol("r") - 14)) == {7, -2}
