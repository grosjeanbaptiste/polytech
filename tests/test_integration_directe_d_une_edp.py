"""Syllabus de mathematiques appliquees 9.1, page 156 : integrer directement.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
page 156 (imprimee) : « en integrant doublement par rapport a x », `u_xx = 0`
donne `u = x f(y) + g(y)` ; « en integrant par rapport a y », `u_y = f(y)`
donne `u = F(y) + G(x)`. Chaque integration ajoute une **fonction
arbitraire** des autres variables, et non une constante.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.edp.integration_directe import par_integrations
from polytech.maths.domain.edp.operateurs import ORDONNEE as Y
from polytech.maths.domain.symboles import VARIABLE as X


def _arbitraires(u: sp.Expr) -> list[sp.Expr]:
    return sorted(u.atoms(sp.core.function.AppliedUndef), key=str)


def test_u_xx_nul_donne_x_f_de_y_plus_g_de_y() -> None:
    u = par_integrations(sp.Integer(0), (X, X), (X, Y))
    f, g = _arbitraires(u)
    assert {f.args, g.args} == {(Y,)}
    assert sp.diff(u, X, 2) == 0
    assert sp.expand(u).coeff(X, 1) in (f, g)


def test_u_y_egal_f_de_y_ajoute_une_fonction_de_x() -> None:
    u = par_integrations(2 * Y, (Y,), (X, Y))
    (g,) = _arbitraires(u)
    assert g.args == (X,)
    assert sp.diff(u, Y) == 2 * Y
    assert sp.expand(u - g) == Y**2


def test_u_xy_nul_donne_f_de_x_plus_g_de_y() -> None:
    u = par_integrations(sp.Integer(0), (X, Y), (X, Y))
    assert sp.diff(u, X, Y).doit() == 0
    assert {a.args for a in _arbitraires(u)} == {(X,), (Y,)}


def test_chaque_integration_a_sa_propre_fonction() -> None:
    u = par_integrations(sp.Integer(0), (X, X, X), (X, Y))
    assert len(_arbitraires(u)) == 3
    assert sp.diff(u, X, 3) == 0


def test_une_variable_a_integrer_hors_des_variables_est_refusee() -> None:
    t = sp.Symbol("t")
    with pytest.raises(ValueError, match="pas une variable"):
        par_integrations(sp.Integer(0), (t,), (X, Y))


def test_au_moins_une_integration() -> None:
    with pytest.raises(ValueError, match="aucune integration"):
        par_integrations(sp.Integer(0), (), (X, Y))
