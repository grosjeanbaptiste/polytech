"""Mathematiques appliquees, 9.1, 9.2 et 9.4 : lire une EDP avant de la resoudre.

Source : `Syllabus_Mathematiques_appliquees_2016_2017.pdf`, pages 155 a
157 et 161.

- 9.1 : deux exemples, « respectivement du premier et second ordre »,
  lineaires et homogenes ; quatre solutions generales a fonctions
  arbitraires, dont deux que le cours laisse a verifier ;
- 9.2 : les equations de la chaleur, de Helmholtz, de Poisson, de
  Schrodinger et de Klein-Gordon, ecrites avec le laplacien ;
- 9.4, exercice : le type de `y u_xx + 2x u_xy + y u_yy = 0` depend du
  point.

Coquille deja consignee a la taxonomie : l'exemple 4 de la page 156
imprime `u = f(x, y) + g(x - y)` ; c'est `f(x + y)`, et le test le montre.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.edp.caracterisation import (
    caracteriser_l_edp,
    est_solution_de_l_edp,
    type_au_point,
    types_selon_le_point,
)
from polytech.maths.domain.edp.classification import Type
from polytech.maths.domain.edp.physique import (
    de_helmholtz,
    de_klein_gordon,
    de_la_chaleur,
    de_poisson,
    de_schrodinger,
)

X, Y, T = sp.symbols("x y t", real=True)
U = sp.Function("u")(X, Y)
F, G = sp.Function("f"), sp.Function("g")


def test_le_premier_exemple_est_du_premier_ordre_lineaire_homogene() -> None:
    equation = 3 * Y**2 * sp.diff(U, X) + sp.diff(U, Y) - 2 * U
    assert caracteriser_l_edp(equation, U) == (1, True, True)


def test_le_second_exemple_est_du_second_ordre() -> None:
    equation = sp.diff(U, X, 2) + F(X, Y) * sp.diff(U, Y, 2)
    assert caracteriser_l_edp(equation, U) == (2, True, True)


def test_un_second_membre_rend_l_equation_non_homogene() -> None:
    equation = sp.diff(U, X, 2) + sp.diff(U, Y, 2) - F(X, Y)
    assert caracteriser_l_edp(equation, U) == (2, True, False)


def test_un_produit_de_derivees_n_est_pas_lineaire() -> None:
    equation = U * sp.diff(U, X) + sp.diff(U, Y)
    assert caracteriser_l_edp(equation, U) == (1, False, True)
    assert caracteriser_l_edp(sp.diff(U, X) ** 2 - U, U)[1] is False
    assert caracteriser_l_edp(sp.sin(U) + sp.diff(U, X), U)[1] is False


def test_une_expression_sans_derivee_n_est_pas_une_edp() -> None:
    with pytest.raises(ValueError, match="aucune derivee partielle"):
        caracteriser_l_edp(U - X, U)


def test_les_quatre_solutions_generales_de_la_page_156() -> None:
    assert est_solution_de_l_edp(sp.diff(U, X, 2), U, F(Y) * X + G(Y))
    primitive = sp.Function("F")
    equation = sp.diff(U, Y) - sp.diff(primitive(Y), Y)
    assert est_solution_de_l_edp(equation, U, primitive(Y) + G(X))
    assert est_solution_de_l_edp(U - Y * sp.diff(U, Y), U, Y * F(X))
    ondes = sp.diff(U, X, 2) - sp.diff(U, Y, 2)
    assert est_solution_de_l_edp(ondes, U, F(X + Y) + G(X - Y))


def test_la_coquille_de_l_exemple_4_n_est_pas_une_solution() -> None:
    ondes = sp.diff(U, X, 2) - sp.diff(U, Y, 2)
    assert not est_solution_de_l_edp(ondes, U, F(X, Y) + G(X - Y))


def test_le_type_de_l_exercice_depend_du_point() -> None:
    types = types_selon_le_point(Y, X, Y)
    assert types[Type.ELLIPTIQUE] == (Y**2 - X**2 > 0)
    assert types[Type.PARABOLIQUE] == sp.Eq(Y**2 - X**2, 0)
    assert types[Type.HYPERBOLIQUE] == (Y**2 - X**2 < 0)
    assert type_au_point(Y, X, Y, {X: 1, Y: 2}) is Type.ELLIPTIQUE
    assert type_au_point(Y, X, Y, {X: 2, Y: 1}) is Type.HYPERBOLIQUE
    assert type_au_point(Y, X, Y, {X: 1, Y: -1}) is Type.PARABOLIQUE


def test_un_point_incomplet_leve() -> None:
    with pytest.raises(ValueError, match="signe du discriminant indetermine"):
        type_au_point(Y, X, Y, {X: 1})


def test_la_chaleur_de_la_barre_du_9_3() -> None:
    # U_t = 3 U_xx : K sin(k pi x / 2) e^(-3 k^2 pi^2 t / 4), ici k = 2.
    mode = sp.sin(sp.pi * X) * sp.exp(-3 * sp.pi**2 * T)
    assert sp.simplify(de_la_chaleur(mode, [X], T, 3)) == 0
    assert sp.simplify(de_la_chaleur(mode, [X], T, 2)) != 0


def test_helmholtz_a_pour_solutions_les_modes_propres() -> None:
    mode = sp.sin(2 * X) * sp.sin(3 * Y)
    assert sp.simplify(de_helmholtz(mode, [X, Y], 13)) == 0


def test_poisson_retranche_la_source() -> None:
    assert de_poisson(X**2 + Y**2, [X, Y], 4) == 0
    assert de_poisson(X**2 + Y**2, [X, Y], X) == 4 - X


def test_schrodinger_d_un_puits_plat() -> None:
    # Delta u + alpha (E - V) u = 0 : sin(kx) avec alpha (E - V) = k^2.
    assert sp.simplify(de_schrodinger(sp.sin(3 * X), [X], 1, 10, 1)) == 0


def test_klein_gordon_ajoute_lambda_carre_u_au_dalembertien() -> None:
    # u = cos(kx - wt), w^2 / c^2 = k^2 - lambda^2 : k = 5, w = 3, lambda = 4.
    onde = sp.cos(5 * X - 3 * T)
    assert sp.simplify(de_klein_gordon(onde, [X], T, 1, 4)) == 0
    assert sp.simplify(de_klein_gordon(onde, [X], T, 1, 3)) != 0


def test_l_ordre_est_celui_de_la_plus_haute_derivee() -> None:
    chaleur = sp.diff(U, X, 2) - sp.diff(U, Y)
    assert caracteriser_l_edp(chaleur, U)[0] == 2
    assert caracteriser_l_edp(sp.diff(U, X, Y, Y) + sp.diff(U, X), U)[0] == 3
