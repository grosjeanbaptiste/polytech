"""Notes de 2011, page 49 : la primitive d'une fraction, racine par racine.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, page 49
(« Deuxieme methode ») : pour `x / (x^4 - 1)`, chaque racine simple `a` du
denominateur contribue `G(a) ln|x - a|`, avec `G(x) = f(x) (x - a)`. Pour
la paire `+- i`, `gamma = G(i) = i / (-2 . 2i) = -1/4`, et la contribution
est `-1/4 ln(x^2 + 1)`. La premiere methode de la meme page donne
`1/4 ln|x - 1| + 1/4 ln|x + 1| - 1/4 ln(x^2 + 1)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.contributions import (
    contribution_d_une_racine_reelle,
    contribution_de_racines_conjuguees,
)
from polytech.maths.domain.symboles import VARIABLE as X

F = X / (X**4 - 1)


def test_la_racine_un_contribue_un_quart_de_ln() -> None:
    assert (
        contribution_d_une_racine_reelle(F, sp.Integer(1)) == sp.log(sp.Abs(X - 1)) / 4
    )


def test_la_racine_moins_un_contribue_un_quart_de_ln() -> None:
    assert (
        contribution_d_une_racine_reelle(F, sp.Integer(-1)) == sp.log(sp.Abs(X + 1)) / 4
    )


def test_la_paire_plus_ou_moins_i_contribue_moins_un_quart() -> None:
    attendue = -sp.log(X**2 + 1) / 4
    assert sp.simplify(contribution_de_racines_conjuguees(F, sp.I) - attendue) == 0


def test_la_somme_des_contributions_est_une_primitive() -> None:
    """Second chemin : deriver la somme rend la fraction."""
    somme = (
        contribution_d_une_racine_reelle(F, sp.Integer(1))
        + contribution_d_une_racine_reelle(F, sp.Integer(-1))
        + contribution_de_racines_conjuguees(F, sp.I)
    )
    for point in (2, 3, sp.Rational(5, 2)):
        assert sp.simplify((sp.diff(somme, X) - F).subs(X, point)) == 0


def test_une_paire_hors_de_l_axe_imaginaire_garde_son_arctangente() -> None:
    """`1 / (x^2 + 2x + 2)` : racines `-1 +- i`, primitive `arctg(x + 1)`."""
    f = 1 / (X**2 + 2 * X + 2)
    contribution = contribution_de_racines_conjuguees(f, -1 + sp.I)
    assert sp.simplify(sp.diff(contribution, X) - f) == 0


def test_une_paire_dont_gamma_a_partie_reelle_et_imaginaire() -> None:
    f = (X + 3) / ((X - 1) * (X**2 + 2 * X + 2))
    somme = contribution_d_une_racine_reelle(f, sp.Integer(1)) + (
        contribution_de_racines_conjuguees(f, -1 + sp.I)
    )
    for point in (2, 3):
        assert sp.simplify((sp.diff(somme, X) - f).subs(X, point)) == 0


def test_une_racine_double_est_refusee() -> None:
    with pytest.raises(ValueError, match="multiplicite 2"):
        contribution_d_une_racine_reelle(1 / (X - 1) ** 2, sp.Integer(1))


def test_une_valeur_qui_n_est_pas_racine_est_refusee() -> None:
    with pytest.raises(ValueError, match="n'est pas racine"):
        contribution_d_une_racine_reelle(F, sp.Integer(2))


def test_une_racine_reelle_n_a_pas_de_conjuguee() -> None:
    with pytest.raises(ValueError, match="partie imaginaire"):
        contribution_de_racines_conjuguees(F, sp.Integer(1))


def test_une_paire_double_est_refusee() -> None:
    with pytest.raises(ValueError, match="multiplicite 2"):
        contribution_de_racines_conjuguees(1 / (X**2 + 1) ** 2, sp.I)
