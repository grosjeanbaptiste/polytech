"""Mener au bout trois methodes du premier ordre que le code laissait en chemin.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`
(pages imprimees) :
- exemple 1.3 (pages 11 et 12) : `y' = 2xy / (x^2 - y^2)` est homogene de
  degre 0 ; `z = y/x` la separe, et l'integration donne des « cercles de
  rayon |C|/2 centres en (0, C/2) » : `x^2 + y^2 = C y` ;
- exemple 1.18 (page 33) : `y' = 1 / (x + y)^2`, la surface parametree
  par `x = u`, `y = v - u`, `z = 1/v^2` ; `dy = z dx` donne
  `v - arctan v = u + C`, soit `y = arctan(x + y) + C` ;
- exemple 1.20 (pages 36 et 37) : `y = x y'^2 + y'^2` ; « en eliminant z,
  on obtient la solution generale » `y = (C + sqrt(x + 1))^2`.

Une integrale premiere `G` est constante le long des solutions : on le
verifie par `G_x + G_y f = 0`, second chemin independant de l'integration.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.clairaut import PENTE as Z
from polytech.maths.domain.differentielles.homogenes import integrale_premiere
from polytech.maths.domain.differentielles.lagrange import EquationDeLagrange
from polytech.maths.domain.differentielles.premier_ordre import (
    INCONNUE as Y,
)
from polytech.maths.domain.differentielles.premier_ordre import (
    VARIABLE as X,
)
from polytech.maths.domain.differentielles.surface import integrer_sur_la_surface

U, V = sp.symbols("u v", real=True)
C = sp.Symbol("C", real=True)


def _constante_le_long(g: sp.Expr, f: sp.Expr) -> bool:
    return bool(sp.simplify(sp.diff(g, X) + sp.diff(g, Y) * f) == 0)


def test_l_integrale_premiere_de_l_exemple_1_3_est_constante() -> None:
    f = 2 * X * Y / (X**2 - Y**2)
    assert _constante_le_long(integrale_premiere(f), f)


def test_l_exemple_1_3_donne_les_cercles_du_cours() -> None:
    """`G` ne depend que de `(x^2 + y^2) / y` : leurs jacobiens sont lies."""
    g = integrale_premiere(2 * X * Y / (X**2 - Y**2))
    cercles = (X**2 + Y**2) / Y
    jacobien = sp.Matrix([g, cercles]).jacobian([X, Y]).det()
    assert sp.simplify(jacobien) == 0


def test_une_homogene_lineaire_en_z() -> None:
    """`y' = y/x + 1` : `x z' = 1`, `G = y/x - ln x`."""
    f = Y / X + 1
    g = integrale_premiere(f)
    assert _constante_le_long(g, f)


def test_une_equation_non_homogene_est_refusee() -> None:
    with pytest.raises(ValueError, match="homogene"):
        integrale_premiere(X + Y**2)


def test_z_egal_f_de_z_n_a_pas_d_integrale_par_z() -> None:
    """`y' = y/x` : `f(z) - z = 0`, les droites `y = c x`.

    Le cours y voit une equation a variables separees, pas ce changement.
    """
    with pytest.raises(ValueError, match="f\\(z\\) = z"):
        integrale_premiere(Y / X)


def test_la_surface_de_l_exemple_1_18_s_integre() -> None:
    relation = integrer_sur_la_surface((U, V - U, 1 / V**2), U, V)
    assert sp.simplify(relation - (V - sp.atan(V) - U)) == 0


def test_revenue_en_x_et_y_la_relation_est_celle_du_cours() -> None:
    relation = integrer_sur_la_surface((U, V - U, 1 / V**2), U, V)
    en_xy = relation.subs({V: X + Y, U: X}, simultaneous=True)
    assert sp.simplify(en_xy - (Y - sp.atan(X + Y))) == 0
    assert _constante_le_long(en_xy, 1 / (X + Y) ** 2)


def test_un_facteur_constant_de_la_pente_est_garde() -> None:
    """`x = u`, `y = v`, `z = 2uv` : `y' = 2xy`, `G = ln y - x^2`."""
    relation = integrer_sur_la_surface((U, V, 2 * U * V), U, V)
    en_xy = relation.subs({U: X, V: Y}, simultaneous=True)
    assert _constante_le_long(en_xy, 2 * X * Y)


def test_une_surface_qui_ne_se_separe_pas_est_refusee() -> None:
    with pytest.raises(ValueError, match="separe"):
        integrer_sur_la_surface((U, V, U + V**2 + sp.sin(U * V)), U, V)


def test_l_elimination_de_z_de_l_exemple_1_20() -> None:
    """Deux branches, `(sqrt(x + 1) +- sqrt(C + 1))^2` : la forme du cours."""
    x = sp.Symbol("x", positive=True)
    branches = EquationDeLagrange(Z**2, Z**2).solution_generale(C)
    cours = {(sp.sqrt(x + 1) + s * sp.sqrt(C + 1)) ** 2 for s in (1, -1)}
    obtenues = {b.subs(X, x) for b in branches}
    assert len(obtenues) == 2
    for branche in obtenues:
        assert any(sp.simplify(sp.expand(branche - c)) == 0 for c in cours)


def test_chaque_branche_verifie_l_equation_de_lagrange() -> None:
    for branche in EquationDeLagrange(Z**2, Z**2).solution_generale(C):
        pente = sp.diff(branche, X)
        assert sp.simplify(branche - (X * pente**2 + pente**2)) == 0
