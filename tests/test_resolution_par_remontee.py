"""Syllabus d'algebre III.3, exemple 2 : echelonner, puis remonter.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 34 et 35
(imprimees). `3x - 2y + 7z = 1`, `-6x + 4y + 3z = 2` : `l2 -> l2 + 2 l1`
donne `17z = 4` ; « de la deuxieme equation, on deduit que z = 4/17 et, de
la premiere, que x = (-11 + 34y)/51 ». `y` est libre : `y = alpha`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.remontee import par_remontee
from polytech.maths.domain.algebre.resolution import par_echelonnement_simple
from polytech.maths.domain.algebre.solutions import PARAMETRES
from polytech.maths.domain.algebre.systemes import SystemeLineaire

ALPHA, BETA = PARAMETRES[:2]
EXEMPLE_2 = SystemeLineaire(sp.Matrix([[3, -2, 7], [-6, 4, 3]]), sp.Matrix([1, 2]))


def test_la_remontee_rend_la_solution_du_cours() -> None:
    attendue = ((-11 + 34 * ALPHA) / 51, ALPHA, sp.Rational(4, 17))
    obtenue = par_remontee(EXEMPLE_2)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(obtenue, attendue, strict=True))


def test_la_remontee_concorde_avec_l_echelonnement_simple() -> None:
    """Second chemin : la forme generale de l'echelonnement simple."""
    simple = par_echelonnement_simple(EXEMPLE_2).forme_generale()
    obtenue = par_remontee(EXEMPLE_2)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(obtenue, simple, strict=True))


def test_deux_variables_libres_prennent_deux_parametres() -> None:
    systeme = SystemeLineaire(sp.Matrix([[1, 1, 1]]), sp.Matrix([3]))
    assert par_remontee(systeme) == (3 - ALPHA - BETA, ALPHA, BETA)


def test_un_systeme_de_cramer_n_a_aucun_parametre() -> None:
    systeme = SystemeLineaire(sp.Matrix([[2, 1], [1, 3]]), sp.Matrix([3, 4]))
    assert par_remontee(systeme) == (1, 1)


def test_un_systeme_impossible_leve() -> None:
    systeme = SystemeLineaire(sp.Matrix([[1, 1], [2, 2]]), sp.Matrix([1, 3]))
    with pytest.raises(ValueError, match="0 = "):
        par_remontee(systeme)


def test_une_equation_redondante_ne_change_rien() -> None:
    systeme = SystemeLineaire(sp.Matrix([[1, 1], [2, 2]]), sp.Matrix([1, 2]))
    assert par_remontee(systeme) == (1 - ALPHA, ALPHA)


def test_trop_de_variables_libres_pour_les_parametres_leve() -> None:
    systeme = SystemeLineaire(sp.Matrix([[1, 1, 1, 1, 1, 1]]), sp.Matrix([0]))
    with pytest.raises(ValueError, match="parametres"):
        par_remontee(systeme)
