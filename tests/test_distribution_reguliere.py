"""La distribution associee a une fonction, et les fonctions a support compact.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
sections 10.7.1 a 10.7.3 : `D(R)` est l'espace des fonctions lisses a support
compact, et `<T_f, phi> = int f phi` (l'integrale se limite au support).
`exp(-x^2)` est dans `S(R)` mais pas dans `D(R)` (exemple du cours).
"""

import pytest
import sympy as sp

from polytech.maths.domain.fourier.distribution_reguliere import (
    appliquer_la_distribution_reguliere,
    support_compact,
)
from polytech.maths.domain.fourier.distributions import FONCTION_TEST
from polytech.maths.domain.symboles import VARIABLE as X

PARABOLE = sp.Piecewise((1 - X**2, sp.Abs(X) < 1), (0, True))


def test_la_fonction_test_du_cours_a_un_support_compact() -> None:
    assert support_compact(FONCTION_TEST) == sp.Interval.open(-1, 1)


def test_l_exponentielle_de_moins_x_carre_n_a_pas_de_support_compact() -> None:
    assert support_compact(sp.exp(-(X**2))) is None


def test_la_fonction_nulle_a_un_support_vide() -> None:
    assert support_compact(sp.Integer(0)) == sp.S.EmptySet


def test_la_distribution_de_x_carre_sur_la_parabole() -> None:
    """`int_{-1}^{1} x^2 (1 - x^2) dx = 2/3 - 2/5 = 4/15`."""
    assert appliquer_la_distribution_reguliere(X**2, PARABOLE) == sp.Rational(4, 15)


def test_la_distribution_associee_a_un_est_l_aire_sous_phi() -> None:
    """`int_{-1}^{1} (1 - x^2) dx = 4/3`."""
    assert appliquer_la_distribution_reguliere(sp.Integer(1), PARABOLE) == sp.Rational(
        4, 3
    )


def test_la_distribution_est_lineaire_en_phi() -> None:
    a, b = sp.symbols("a b", real=True)
    autre = sp.Piecewise((X + 1, sp.Abs(X) < 1), (0, True))
    gauche = appliquer_la_distribution_reguliere(X, a * PARABOLE + b * autre)
    droite = a * appliquer_la_distribution_reguliere(X, PARABOLE)
    droite += b * appliquer_la_distribution_reguliere(X, autre)
    assert sp.simplify(gauche - droite) == 0


def test_une_fonction_sans_support_compact_est_refusee() -> None:
    with pytest.raises(ValueError, match="pas a support compact"):
        appliquer_la_distribution_reguliere(X, sp.exp(-(X**2)))


def test_un_support_borne_d_un_seul_cote_n_est_pas_compact() -> None:
    demi_droite = sp.Piecewise((sp.exp(-X), X > 0), (0, True))
    assert support_compact(demi_droite) is None


def test_la_distribution_sur_la_fonction_nulle_est_nulle() -> None:
    assert appliquer_la_distribution_reguliere(X**2, sp.Integer(0)) == 0
