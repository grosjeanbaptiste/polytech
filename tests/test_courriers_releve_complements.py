"""Complements du releve des courriers : les valeurs que ses tests n'epinglaient pas.

Chaque test pointe une affirmation d'un courrier (`travaux/Transversal/
courriers-aux-enseignants/`) que les tests cites par `releve-source.md`
laissaient sans valeur exacte : la bibliotheque donne la valeur, le test
compare, et la valeur imprimee par l'enseignant est declaree differente.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import racines_de_l_unite
from polytech.maths.domain.algebre.solutions import solution_generale
from polytech.maths.domain.algebre.spectre import Spectre
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.edp.euler import EquationDEuler
from polytech.maths.domain.integration.hermite import inverse_d_une_puissance

X = sp.Symbol("x", real=True)
R = sp.Rational


def test_m8_les_vecteurs_propres_du_courrier_ont_un_produit_scalaire_de_un() -> None:
    jordan = sp.Matrix([[1, 0, 1], [2, -2, -1], [0, 0, 1]])
    couples = dict(Spectre(jordan).couples_propres())
    assert couples[-2] == sp.Matrix([0, 1, 0])
    assert couples[1] == sp.Matrix([R(3, 2), 1, 0])
    assert couples[-2].dot(couples[1]) == 1


def test_m13_le_coefficient_croise_vaut_quatre_fois_le_discriminant_sur_b() -> None:
    a, h, b = 1, 3, 2
    equation = EquationDEuler(a, h, b)
    premiere, seconde = equation.pentes()
    _, croise, _ = equation.coefficients_transformes(1, premiere, 1, seconde)
    assert egales(croise, 4 * (a * b - h**2) / b)
    assert not egales(croise, 2 * (a * b - h**2) / b)


def test_m27_la_forme_en_t_est_la_primitive_de_un_sur_un_plus_t2_cube() -> None:
    t = X
    forme = (
        R(3, 8) * sp.atan(t) + R(3, 8) * t / (1 + t**2) + R(1, 4) * t / (1 + t**2) ** 2
    )
    assert egales(forme, inverse_d_une_puissance(3))
    assert egales(sp.diff(forme, t), 1 / (1 + t**2) ** 3)


def test_m30_pour_a_egal_trois_x_vaut_deux_y_plus_quatre() -> None:
    systeme = SystemeLineaire(sp.Matrix([[3, -6], [2, -4]]), sp.Matrix([12, 8]))
    x, y = solution_generale(systeme)
    assert egales(x, 2 * y + 4)
    assert not egales(x, 2 * y + 2)
    assert egales(2 * x - 4 * y, 8)
    assert not egales(2 * (2 * y + 2) - 4 * y, 8)


def test_i17_a_pour_n_egal_un_la_somme_des_racines_de_l_unite_vaut_un() -> None:
    assert sum(racines_de_l_unite(1)) == 1
    for n in (2, 3, 6):
        assert egales(sum(racines_de_l_unite(n)), 0)
