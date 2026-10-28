"""Syllabus de mathematiques appliquees : complements 1.A et geometrie 2.1.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 32 a 44. La methode de la surface (exemples 1.18, 1.21 a 1.23),
Clairaut (1.19), Lagrange (1.20), les familles de courbes et leurs
trajectoires (exemples 2.1 et 2.2), les enveloppes (2.1.5).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.clairaut import (
    PENTE,
    EquationDeClairaut,
)
from polytech.maths.domain.differentielles.enveloppes import enveloppe
from polytech.maths.domain.differentielles.familles import (
    equation_d_ordre_n,
    pente_de_la_famille,
    pente_isogonale,
    pente_orthogonale,
    pentes_de_la_famille,
    valeurs_de_la_constante,
)
from polytech.maths.domain.differentielles.lagrange import (
    EquationDeLagrange,
    est_singuliere,
)
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE
from polytech.maths.domain.differentielles.surface import forme_sur_la_surface
from polytech.maths.domain.symboles import CONSTANTE as C

X, Y, Z = VARIABLE, INCONNUE, PENTE
U, V = sp.symbols("u v", real=True)
A = sp.Symbol("a", positive=True)
R = sp.Rational


def _pente_implicite(courbe: sp.Expr) -> sp.Expr:
    return sp.idiff(courbe, Y, X)


def test_la_methode_de_la_surface_sur_l_exemple_1_18() -> None:
    """`y' = 1/(x + y)^2`, `x = u`, `x + y = v` : `dv = du (v^2 + 1)/v^2`."""
    en_u, en_v = forme_sur_la_surface((U, V - U, 1 / V**2), U, V)
    assert sp.simplify(-en_u / en_v - (V**2 + 1) / V**2) == 0


def test_la_solution_implicite_de_l_exemple_1_18() -> None:
    """`y = arctan(x + y) + C` : sa pente est bien `1/(x + y)^2`."""
    pente = _pente_implicite(Y - sp.atan(X + Y) - C)
    assert sp.simplify(pente - 1 / (X + Y) ** 2) == 0


def test_la_methode_de_la_surface_sur_l_exemple_1_22() -> None:
    """`x = sin y' + y'` : `dy = z (cos z + 1) dz`, que l'on integre en `z`."""
    en_z, en_y = forme_sur_la_surface((sp.sin(Z) + Z, Y, Z), Z, Y)
    assert en_y == 1
    assert sp.simplify(-en_z - Z * (sp.cos(Z) + 1)) == 0
    y = Z**2 / 2 + Z * sp.sin(Z) + sp.cos(Z) + C
    assert sp.simplify(sp.diff(y, Z) / sp.diff(sp.sin(Z) + Z, Z) - Z) == 0


def test_les_cercles_de_l_exemple_1_21_et_leur_enveloppe() -> None:
    """`y^2 (1 + y'^2) = a^2` : `(x + c)^2 + y^2 = a^2`, d'enveloppe `y = +-a`."""
    cercles = (X + C) ** 2 + Y**2 - A**2
    pente = _pente_implicite(cercles)
    sur_le_cercle = (Y**2 * (1 + pente**2) - A**2).subs(Y**2, A**2 - (X + C) ** 2)
    assert sp.simplify(sur_le_cercle.subs(Y, sp.sqrt(A**2 - (X + C) ** 2))) == 0
    assert set(enveloppe(cercles, C)) == {(-C, A), (-C, -A)}


def test_les_deux_methodes_de_l_exemple_1_23_s_accordent() -> None:
    """`x y'^2 = y` : `y = (sqrt x + K)^2` verifie (1.53) avec `c = K^2`.

    Le syllabus ecrit « avec `K = c` » (page 38) : c'est `K^2 = c`.
    """
    k = sp.Symbol("K", real=True)
    x = sp.Symbol("x", positive=True)
    y = (sp.sqrt(x) + k) ** 2
    assert sp.simplify(x * sp.diff(y, x) ** 2 - y) == 0

    def relation(c: sp.Expr) -> sp.Expr:
        return sp.expand((x - y) ** 2 - 2 * c * (x + y) + c**2)

    assert relation(k**2) == 0
    assert relation(k) != 0


def test_les_droites_d_une_equation_de_clairaut() -> None:
    """(1.48) : `y = x y' + phi(y')` a pour solutions les droites `y = c x + phi(c)`."""
    equation = EquationDeClairaut(A * Z / sp.sqrt(1 + Z**2))
    droite = equation.droite(C)
    assert droite == C * X + A * C / sp.sqrt(1 + C**2)
    assert sp.simplify(droite - X * sp.diff(droite, X) - equation.phi_en(C)) == 0


def test_l_enveloppe_de_l_exemple_1_19() -> None:
    """`x = -a (1 + c^2)^(-3/2)`, `y = a c^3 (1 + c^2)^(-3/2)`."""
    equation = EquationDeClairaut(A * Z / sp.sqrt(1 + Z**2))
    x, y = equation.solution_singuliere(C)
    assert sp.simplify(x + A * (1 + C**2) ** R(-3, 2)) == 0
    assert sp.simplify(y - A * C**3 * (1 + C**2) ** R(-3, 2)) == 0


def test_la_solution_singuliere_de_clairaut_verifie_l_equation() -> None:
    """Sa pente au parametre `c` est `c`, et `y = x c + phi(c)` y tient."""
    equation = EquationDeClairaut(A * Z / sp.sqrt(1 + Z**2))
    x, y = equation.solution_singuliere(C)
    assert sp.simplify(sp.diff(y, C) / sp.diff(x, C) - C) == 0
    assert sp.simplify(y - x * C - equation.phi_en(C)) == 0


def test_la_solution_singuliere_de_clairaut_est_l_enveloppe_des_droites() -> None:
    """Page 35 : `G = y - c x - phi(c)` et `dG/dc = 0` redonnent les memes equations."""
    equation = EquationDeClairaut(Z**2)
    (obtenue,) = enveloppe(Y - equation.droite(C), C)
    assert obtenue == equation.solution_singuliere(C) == (-2 * C, -(C**2))


def test_l_equation_lineaire_en_x_de_l_exemple_1_20() -> None:
    """`y = x y'^2 + y'^2` : `x' - x 2z/(z - z^2) = 2/(1 - z)`."""
    p, q = EquationDeLagrange(Z**2, Z**2).equation_en_x()
    assert sp.simplify(p + 2 * Z / (Z - Z**2)) == 0
    assert sp.simplify(q - 2 / (1 - Z)) == 0


def test_la_solution_x_de_z_de_l_exemple_1_20() -> None:
    """`x = -1 + K / (z - 1)^2`, la constante `K` valant ici `C + 1`."""
    x = EquationDeLagrange(Z**2, Z**2).x_de_z(C)
    assert sp.simplify(x - (-1 + (C + 1) / (Z - 1) ** 2)) == 0


def test_les_solutions_lineaires_de_l_exemple_1_20() -> None:
    """Les zeros de `z - psi(z)` : `z = 0` et `z = 1`, d'ou `y = 0` et `y = x + 1`."""
    equation = EquationDeLagrange(Z**2, Z**2)
    assert equation.pentes_constantes() == (0, 1)
    assert equation.solutions_lineaires() == (0, X + 1)


def test_seule_la_premiere_solution_lineaire_est_singuliere() -> None:
    """`y = (C + sqrt(x + 1))^2` : `y = x + 1` en sort pour `C = 0`, `y = 0` non."""
    x = sp.Symbol("x", positive=True)
    generale = (C + sp.sqrt(x + 1)) ** 2
    assert sp.simplify(generale - (x + 1) * sp.diff(generale, x) ** 2) == 0
    assert est_singuliere(sp.Integer(0), generale, C)
    assert not est_singuliere(x + 1, generale, C)


def test_clairaut_est_le_cas_de_lagrange_ou_psi_est_l_identite() -> None:
    """Page 36 : alors `z - psi(z)` est nul partout, et (1.52) n'a plus de sens."""
    with pytest.raises(ValueError, match="equation de Clairaut"):
        EquationDeLagrange(Z, Z**2)


def test_l_equation_differentielle_des_paraboles_de_l_exemple_2_1() -> None:
    """`y = C x^2` : `y' = 2 C x = 2 y / x`."""
    assert pente_de_la_famille(Y - C * X**2, C) == 2 * Y / X


def test_l_equation_differentielle_des_cercles_de_la_page_41() -> None:
    """`x^2 + y^2 + C x = 0` : `y' = (y^2 - x^2) / (2 x y)`."""
    pente = pente_de_la_famille(X**2 + Y**2 + C * X, C)
    assert sp.simplify(pente - (Y**2 - X**2) / (2 * X * Y)) == 0


def test_les_trajectoires_orthogonales_de_l_exemple_2_2() -> None:
    """`y' = -x / (2y)`, de solution les ellipses `x^2/2 + y^2 = K`."""
    orthogonale = pente_orthogonale(pente_de_la_famille(Y - C * X**2, C))
    assert sp.simplify(orthogonale + X / (2 * Y)) == 0
    assert sp.simplify(_pente_implicite(X**2 / 2 + Y**2 - C) - orthogonale) == 0


def test_une_trajectoire_isogonale_coupe_sous_l_angle_voulu() -> None:
    """(2.5) : `y_T' = (y' + tan a) / (1 - y' tan a)`."""
    pente, angle = R(1, 2), sp.pi / 6
    coupee = pente_isogonale(pente, angle)
    entre_les_deux = (coupee - pente) / (1 + coupee * pente)
    assert sp.simplify(entre_les_deux - sp.tan(angle)) == 0
    assert pente_isogonale(0, sp.pi / 4) == 1


def test_l_isogonale_a_angle_droit_est_l_orthogonale() -> None:
    """Page 43 : pour `alpha = pi/2`, `y_T' = -1/y'`."""
    alpha = sp.Symbol("alpha")
    limite = sp.limit(pente_isogonale(Y, alpha), alpha, sp.pi / 2)
    assert sp.simplify(limite - pente_orthogonale(Y)) == 0


def test_une_famille_a_deux_parametres_donne_une_equation_d_ordre_deux() -> None:
    """2.1.4 : `y = C1 cos x + C2 sin x` est la solution generale de `y'' = -y`."""
    c1, c2 = sp.symbols("C1 C2")
    equation = equation_d_ordre_n(c1 * sp.cos(X) + c2 * sp.sin(X), (c1, c2))
    y = sp.Function("y")(X)
    assert equation == sp.Eq(y.diff(X, 2), -y)


def test_une_famille_a_un_parametre_donne_une_equation_d_ordre_un() -> None:
    """`y = C x^2`, vue comme famille a `n = 1` parametre : `y' = 2 y / x`."""
    y = sp.Function("y")(X)
    assert equation_d_ordre_n(C * X**2, (C,)) == sp.Eq(y.diff(X), 2 * y / X)


def test_une_famille_sans_enveloppe_le_dit() -> None:
    """Les droites paralleles `y = x + c` ne se touchent nulle part."""
    with pytest.raises(ValueError, match="aucune enveloppe"):
        enveloppe(Y - X - C, C)


def test_une_equation_de_lagrange_ou_psi_et_phi_different() -> None:
    """`y = 2 x y' + y'^2` : `x' + 2x/z = -2`, `x = C/z^2 - 2z/3`, et `dy = z dx`."""
    equation = EquationDeLagrange(2 * Z, Z**2)
    assert equation.equation_en_x() == (2 / Z, -2)
    x = equation.x_de_z(C)
    assert sp.simplify(x - (C / Z**2 - 2 * Z / 3)) == 0
    y = x * 2 * Z + Z**2
    assert sp.simplify(sp.diff(y, Z) - Z * sp.diff(x, Z)) == 0


def test_les_droites_d_une_equation_de_lagrange_portent_psi_en_pente() -> None:
    """`y = x y'^2 + y' + 1` : `z_0 = 0` donne `y = 1`, `z_0 = 1` donne `y = x + 2`."""
    equation = EquationDeLagrange(Z**2, Z + 1)
    assert equation.solutions_lineaires() == (1, X + 2)
    for droite in equation.solutions_lineaires():
        pente = sp.diff(droite, X)
        assert sp.expand(droite - X * pente**2 - pente - 1) == 0


def test_une_famille_a_deux_racines_donne_deux_pentes() -> None:
    """`y = (racine x + K)^2` : deux valeurs de `K`, pentes `+- racine(y/x)`."""
    k = sp.Symbol("K", real=True)
    pentes = pentes_de_la_famille(Y - (sp.sqrt(X) + k) ** 2, k)
    assert len(pentes) == 2
    assert {sp.simplify(p**2 - Y / X) for p in pentes} == {0}
    assert sp.simplify(pentes[0] + pentes[1]) == 0


def test_un_parametre_absent_de_la_famille_leve() -> None:
    with pytest.raises(ValueError, match="ne se tire pas"):
        pentes_de_la_famille(Y - X, sp.Symbol("K", real=True))


def test_la_constante_de_l_integrale_complete_est_k_carre() -> None:
    k, c = sp.Symbol("K", real=True), sp.Symbol("c", real=True)
    y = (sp.sqrt(X) + k) ** 2
    relation = (X - Y) ** 2 - 2 * c * (X + Y) + c**2
    assert valeurs_de_la_constante(relation, y, c) == (k**2,)
    assert valeurs_de_la_constante(Y - c, sp.Integer(3), c) == (3,)
