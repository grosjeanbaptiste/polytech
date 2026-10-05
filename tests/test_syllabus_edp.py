"""Syllabus de mathematiques appliquees, chapitre 9 : les EDP.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 155 a 169. Les solutions generales de la page 156, les equations de
la physique mathematique, la barre refroidie resolue par une serie de
Fourier (9.3), la classification (9.4), l'equation d'Euler (9.5), le
principe du maximum (9.7), la corde vibrante (9.8) et la demi-bande (9.9).

Une coquille du syllabus est etablie : `u = f(x, y) + g(x - y)` page 156,
pour `f(x + y) + g(x - y)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.edp.classification import (
    Condition,
    Type,
    conditions_adaptees,
    discriminant,
    type_a_deux_variables,
    type_par_la_forme_quadratique,
)
from polytech.maths.domain.edp.demi_bande import (
    amplitude_de_la_demi_bande,
    mode_de_la_demi_bande,
)
from polytech.maths.domain.edp.euler import EquationDEuler
from polytech.maths.domain.edp.operateurs import (
    ORDONNEE,
    dalembertien,
    est_harmonique,
    laplacien,
)
from polytech.maths.domain.edp.separation import (
    constante_de_separation,
    corde_vibrante,
    diffusion_dans_une_barre,
    mode_de_diffusion,
)
from polytech.maths.domain.fourier.cosinus import transformee_en_cosinus
from polytech.maths.domain.fourier.parite import prolongement_impair
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

Y = ORDONNEE
R = sp.Rational
F, G = sp.Function("F"), sp.Function("G")
C = sp.Symbol("c", positive=True)


def test_les_solutions_generales_de_la_page_156() -> None:
    """Des fonctions arbitraires, et non des constantes, y servent de parametres."""
    assert sp.diff(F(Y) * X + G(Y), X, 2) == 0
    primitive = sp.Integral(F(Y), Y)
    assert sp.diff(primitive + G(X), Y).doit() == F(Y)
    produit = Y * F(X)
    assert sp.simplify(produit - Y * sp.diff(produit, Y)) == 0


def test_la_solution_generale_de_l_equation_des_ondes_normalisee() -> None:
    """`u_xx = u_yy` : `u = f(x + y) + g(x - y)`."""
    u = F(X + Y) + G(X - Y)
    assert sp.simplify(sp.diff(u, X, 2).doit() - sp.diff(u, Y, 2).doit()) == 0


def test_la_forme_imprimee_page_156_n_est_pas_solution() -> None:
    """Le syllabus ecrit `f(x, y) + g(x - y)` : une `f` de deux variables echoue."""
    quelconque = X**2 * Y + G(X - Y)
    ecart = sp.diff(quelconque, X, 2).doit() - sp.diff(quelconque, Y, 2).doit()
    assert sp.simplify(ecart) == 2 * Y


def test_le_laplacien_et_les_fonctions_harmoniques() -> None:
    """9.2 : `Delta u = u_xx + u_yy` ; ses zeros sont les fonctions harmoniques."""
    assert laplacien(X**2 * Y**3, (X, Y)) == 2 * Y**3 + 6 * X**2 * Y
    assert est_harmonique(X**2 - Y**2, (X, Y))
    assert est_harmonique(sp.exp(X) * sp.cos(Y), (X, Y))
    assert not est_harmonique(X**2 + Y**2, (X, Y))


def test_le_dalembertien_annule_les_ondes_progressives() -> None:
    """`Delta u - u_tt / c^2 = 0` pour `F(x - ct) + G(x + ct)`."""
    onde = F(X - C * T) + G(X + C * T)
    assert sp.simplify(dalembertien(onde, (X,), T, C).doit()) == 0
    assert dalembertien(X**2 * T**2, (X,), T, C) == 2 * T**2 - 2 * X**2 / C**2


def test_les_equations_de_la_chaleur_et_de_helmholtz() -> None:
    """`Delta u = u_t / k` et `Delta u + lambda u = 0`, sur une solution de chacune."""
    k, lam = sp.symbols("k lambda", positive=True)
    chaleur = sp.exp(-k * T) * sp.sin(X)
    assert sp.simplify(laplacien(chaleur, (X,)) - sp.diff(chaleur, T) / k) == 0
    stationnaire = sp.sin(sp.sqrt(lam) * X)
    assert sp.simplify(laplacien(stationnaire, (X,)) + lam * stationnaire) == 0


def test_la_constante_de_separation_de_la_barre() -> None:
    """9.3 : `a'' = lambda a`, `a(0) = a(2) = 0` impose `lambda = -k^2 pi^2 / 4`."""
    for rang in (1, 2, 3):
        valeur = constante_de_separation(rang, 2)
        assert valeur == -(rang**2) * sp.pi**2 / 4
        mode = sp.sin(rang * sp.pi * X / 2)
        assert sp.simplify(sp.diff(mode, X, 2) - valeur * mode) == 0
        assert mode.subs(X, 0) == mode.subs(X, 2) == 0


def test_un_mode_de_la_barre_de_la_page_159() -> None:
    """`U = sin(k pi x / 2) e^(-3 k^2 pi^2 t / 4)` verifie `U_t = 3 U_xx`."""
    for rang in (1, 2):
        mode = mode_de_diffusion(rang, longueur=2, diffusivite=3)
        attendu = sp.sin(rang * sp.pi * X / 2) * sp.exp(
            -R(3, 4) * rang**2 * sp.pi**2 * T
        )
        assert sp.simplify(mode - attendu) == 0
        assert sp.simplify(sp.diff(mode, T) - 3 * sp.diff(mode, X, 2)) == 0
        assert mode.subs(X, 0) == 0
        assert mode.subs(X, 2) == 0


def test_la_solution_de_la_barre_refroidie() -> None:
    """9.3 : `U = somme 4 (-1)^(k+1) / (pi k) sin(k pi x / 2) e^(-3 k^2 pi^2 t / 4)`."""
    solution = diffusion_dans_une_barre(X, longueur=2, diffusivite=3, ordre=3)
    attendue = sum(
        4
        * (-1) ** (k + 1)
        / (sp.pi * k)
        * mode_de_diffusion(k, longueur=2, diffusivite=3)
        for k in (1, 2, 3)
    )
    assert sp.simplify(solution - attendue) == 0
    assert sp.simplify(sp.diff(solution, T) - 3 * sp.diff(solution, X, 2)) == 0
    assert solution.subs(X, 0) == 0
    assert sp.simplify(solution.subs(X, 2)) == 0


def test_la_condition_initiale_de_la_barre_est_une_serie_de_fourier() -> None:
    """Lien avec l'exemple 8.6 : en `t = 0`, la serie de sinus de `x` sur `[0, 2]`."""
    solution = diffusion_dans_une_barre(X, longueur=2, diffusivite=3, ordre=4)
    serie = prolongement_impair(X, 2)
    assert sp.simplify(solution.subs(T, 0) - serie.somme_partielle(4)) == 0


def test_la_barre_refroidit_vers_zero() -> None:
    """La glace fondante aux deux bouts l'emporte : chaque mode s'eteint."""
    solution = diffusion_dans_une_barre(X, longueur=2, diffusivite=3, ordre=3)
    assert sp.limit(solution.subs(X, 1), T, sp.oo) == 0


@pytest.mark.parametrize(
    ("coefficients", "attendu"),
    [
        ((1, 0, -1 / C**2), Type.HYPERBOLIQUE),
        ((1, 0, 1), Type.ELLIPTIQUE),
        ((1, 0, 0), Type.PARABOLIQUE),
        ((1, 2, 4), Type.PARABOLIQUE),
    ],
)
def test_le_type_des_equations_du_9_4(
    coefficients: tuple[object, object, object], attendu: Type
) -> None:
    """Ondes, Laplace, chaleur : le signe de `ab - h^2`, comme pour les coniques."""
    assert type_a_deux_variables(*coefficients) is attendu


def test_l_exercice_a_coefficients_non_constants() -> None:
    """`y u_xx + 2x u_xy + y u_yy = 0` : `ab - h^2 = y^2 - x^2` change de signe."""
    assert discriminant(Y, X, Y) == Y**2 - X**2
    points = {
        (0, 1): Type.ELLIPTIQUE,
        (1, 0): Type.HYPERBOLIQUE,
        (1, 1): Type.PARABOLIQUE,
    }
    for (x, y), attendu in points.items():
        ici = {X: x, Y: y}
        assert type_a_deux_variables(Y.subs(ici), X.subs(ici), Y.subs(ici)) is attendu


def test_un_type_qui_depend_du_point_ne_se_decide_pas_en_bloc() -> None:
    with pytest.raises(ValueError, match="signe du discriminant indetermine"):
        type_a_deux_variables(Y, X, Y)


def test_le_type_par_les_valeurs_propres() -> None:
    """Page 160 : ondes a deux dimensions, Laplace a trois, chaleur a deux."""
    assert type_par_la_forme_quadratique(sp.diag(1, 1, -1 / C**2)) is Type.HYPERBOLIQUE
    assert type_par_la_forme_quadratique(sp.eye(3)) is Type.ELLIPTIQUE
    assert type_par_la_forme_quadratique(sp.diag(1, 1, 0)) is Type.PARABOLIQUE
    assert type_par_la_forme_quadratique(-sp.eye(2)) is Type.ELLIPTIQUE


def test_les_conditions_aux_frontieres_qui_conviennent() -> None:
    """Page 165 : Dirichlet ou Neumann pour l'elliptique, Cauchy pour l'hyperbolique."""
    assert conditions_adaptees(Type.ELLIPTIQUE) == (
        Condition.DIRICHLET,
        Condition.NEUMANN,
    )
    assert conditions_adaptees(Type.HYPERBOLIQUE) == (Condition.CAUCHY,)


def test_le_syllabus_ne_dit_rien_des_conditions_du_type_parabolique() -> None:
    with pytest.raises(ValueError, match="le syllabus ne le dit pas"):
        conditions_adaptees(Type.PARABOLIQUE)


def test_le_changement_de_variables_de_l_equation_d_euler() -> None:
    """(9.7) : les trois coefficients transformes, contre la derivation en chaine."""
    a, h, b, p, q, r, s = 2, 3, -1, 1, 2, 3, -1
    xi, eta = sp.symbols("xi eta")
    quelconque = xi**3 * eta + xi * eta**2 + sp.sin(xi) * sp.exp(eta)
    en_x_y = quelconque.subs({xi: p * X + q * Y, eta: r * X + s * Y}, simultaneous=True)
    direct = EquationDEuler(a, h, b).membre(en_x_y)
    c1, c2, c3 = EquationDEuler(a, h, b).coefficients_transformes(p, q, r, s)
    transforme = (
        c1 * sp.diff(quelconque, xi, 2)
        + c2 * sp.diff(quelconque, xi, eta)
        + c3 * sp.diff(quelconque, eta, 2)
    ).subs({xi: p * X + q * Y, eta: r * X + s * Y}, simultaneous=True)
    assert sp.simplify(direct - transforme) == 0


def test_le_coefficient_croise_devient_deux_ab_moins_h2_sur_b() -> None:
    """(9.7) puis (9.11) : il ne reste que le terme croise, `4 (ab - h^2) / b`.

    Le syllabus ecrit `2 (ab - h^2) / b` : il a laisse tomber le 2 de (9.7),
    sans consequence puisque le membre de droite est nul.
    """
    equation = EquationDEuler(1, 3, 2)
    premiere, seconde = equation.pentes()
    c1, c2, c3 = equation.coefficients_transformes(1, premiere, 1, seconde)
    assert sp.simplify(c1) == 0
    assert sp.simplify(c3) == 0
    assert sp.simplify(c2 - 4 * discriminant(1, 3, 2) / 2) == 0


def test_l_equation_des_ondes_par_l_equation_d_euler() -> None:
    """Cas 1 : `a = 1, h = 0, b = -1/c^2`, pentes `+-c`, `u = F(x + ct) + G(x - ct)`."""
    equation = EquationDEuler(1, 0, -1 / C**2)
    assert set(equation.pentes()) == {C, -C}
    assert equation.type() is Type.HYPERBOLIQUE
    generale = equation.solution_generale(F, G)
    assert sp.simplify(equation.membre(generale).doit()) == 0
    assert generale.atoms(sp.Function) == {F(X + C * Y), G(X - C * Y)} or (
        generale.atoms(sp.Function) == {F(X - C * Y), G(X + C * Y)}
    )


def test_l_equation_de_laplace_par_l_equation_d_euler() -> None:
    """Cas 2 : pentes `+-j`, `u = F(x + jy) + G(x - jy)` -- des arguments complexes."""
    equation = EquationDEuler(1, 0, 1)
    assert set(equation.pentes()) == {sp.I, -sp.I}
    assert equation.type() is Type.ELLIPTIQUE
    assert sp.simplify(equation.membre(equation.solution_generale(F, G)).doit()) == 0
    assert est_harmonique(sp.re(sp.expand((X + sp.I * Y) ** 3)), (X, Y))


def test_le_cas_parabolique_de_la_page_164() -> None:
    """`u_xx + 4 u_xy + 4 u_yy = 0` : `lambda = -1/2` double, `u = F + y G`."""
    equation = EquationDEuler(1, 2, 4)
    assert equation.pentes() == (-R(1, 2), -R(1, 2))
    assert equation.type() is Type.PARABOLIQUE
    generale = equation.solution_generale(F, G)
    assert generale == F(X - Y / 2) + Y * G(X - Y / 2)
    assert sp.simplify(equation.membre(generale).doit()) == 0


def test_le_principe_du_maximum() -> None:
    """9.7 : une fonction harmonique n'a pas d'extremum strict a l'interieur.

    `Delta u = 0` rend la trace de la hessienne nulle : en un point critique,
    les deux courbures sont de signes opposes. Et `v = u + eps (x^2 + y^2)` a
    `Delta v = 4 eps`.
    """
    u = X**3 - 3 * X * Y**2 + X**2 - Y**2
    assert est_harmonique(u, (X, Y))
    hessienne = sp.hessian(u, (X, Y))
    assert hessienne.trace() == 0
    for point in sp.solve([sp.diff(u, X), sp.diff(u, Y)], (X, Y), dict=True):
        assert hessienne.subs(point).det() <= 0
    eps = sp.Symbol("epsilon", positive=True)
    assert laplacien(u + eps * (X**2 + Y**2), (X, Y)) == 4 * eps


def test_les_extremes_d_une_fonction_harmonique_sont_au_bord() -> None:
    """Sur le carre `[0, 1]^2`, `x^2 - y^2` va de -1 a 1, valeurs prises sur le bord."""
    u = X**2 - Y**2
    bord = [u.subs(X, 0), u.subs(X, 1), u.subs(Y, 0), u.subs(Y, 1)]
    extremes = [
        sp.calculus.util.function_range(e, e.free_symbols.pop(), sp.Interval(0, 1))
        for e in bord
    ]
    assert max(e.sup for e in extremes) == 1
    assert min(e.inf for e in extremes) == -1
    assert -1 < u.subs({X: R(1, 2), Y: R(1, 3)}) < 1


def test_un_mode_de_la_corde_vibrante() -> None:
    """9.8 : `f = sin(pi x / l)`, sans vitesse : `u = cos(pi c t / l) sin(pi x / l)`."""
    longueur = sp.Symbol("l", positive=True)
    forme = sp.sin(sp.pi * X / longueur)
    corde = corde_vibrante(forme, sp.Integer(0), longueur=longueur, celerite=C, ordre=3)
    attendue = sp.cos(sp.pi * C * T / longueur) * forme
    assert sp.simplify(corde - attendue) == 0
    assert sp.simplify(dalembertien(corde, (X,), T, C)) == 0


def test_une_corde_lancee_sans_deplacement_initial() -> None:
    """`g = sin(2 pi x / l)` : `D_2 = l / (2 pi c)`, et `u_t(x, 0) = g`."""
    longueur = sp.Symbol("l", positive=True)
    vitesse = sp.sin(2 * sp.pi * X / longueur)
    corde = corde_vibrante(
        sp.Integer(0), vitesse, longueur=longueur, celerite=C, ordre=3
    )
    attendue = (
        longueur / (2 * sp.pi * C) * sp.sin(2 * sp.pi * C * T / longueur) * vitesse
    )
    assert sp.simplify(corde - attendue) == 0
    assert sp.simplify(sp.diff(corde, T).subs(T, 0) - vitesse) == 0
    assert corde.subs(T, 0) == 0


def test_la_corde_pincee_en_son_milieu() -> None:
    """`f` en triangle : `C_k = (2/l) int f sin(k pi x / l)`, nul pour `k` pair."""
    forme = sp.Piecewise((X, X < 1), (2 - X, True))
    corde = corde_vibrante(forme, sp.Integer(0), longueur=2, celerite=1, ordre=3)
    premier = sp.integrate(forme * sp.sin(sp.pi * X / 2), (X, 0, 2))
    assert premier == 8 / sp.pi**2
    attendue = (
        8
        / sp.pi**2
        * (
            sp.cos(sp.pi * T / 2) * sp.sin(sp.pi * X / 2)
            - sp.cos(3 * sp.pi * T / 2) * sp.sin(3 * sp.pi * X / 2) / 9
        )
    )
    assert sp.simplify(corde - attendue) == 0
    assert corde.subs(X, 0) == 0
    assert sp.simplify(corde.subs(X, 2)) == 0


def test_un_mode_borne_de_la_demi_bande() -> None:
    """9.9 : `cos(lambda x) ch(lambda y)`, harmonique, a derivees nulles aux bords."""
    lam = sp.Symbol("lambda", positive=True)
    mode = mode_de_la_demi_bande(lam)
    assert mode == sp.cos(lam * X) * sp.cosh(lam * Y)
    assert est_harmonique(mode, (X, Y))
    assert sp.diff(mode, X).subs(X, 0) == 0
    assert sp.diff(mode, Y).subs(Y, 0) == 0


def test_la_transformee_en_cosinus_de_la_fenetre() -> None:
    """Page 169 : `int_0^1 cos(lambda t) dt = sin(lambda) / lambda`."""
    lam = sp.Symbol("lambda", positive=True)
    fenetre = sp.Piecewise((1, X <= 1), (0, True))
    assert sp.simplify(transformee_en_cosinus(fenetre, lam) - sp.sin(lam) / lam) == 0


def test_l_amplitude_de_la_demi_bande() -> None:
    """`A(lambda) = (2/pi) sin(lambda) / (lambda ch lambda)` pour la fenetre."""
    lam = sp.Symbol("lambda", positive=True)
    fenetre = sp.Piecewise((1, X <= 1), (0, True))
    amplitude = amplitude_de_la_demi_bande(fenetre, lam)
    assert sp.simplify(amplitude - 2 * sp.sin(lam) / (sp.pi * lam * sp.cosh(lam))) == 0


def test_la_solution_de_la_demi_bande_retrouve_la_fenetre_en_y_egal_un() -> None:
    """`u(x, 1) = (2/pi) int cos(lambda x) sin(lambda) / lambda` : 1, puis 0.

    `cos(lambda x) sin(lambda) = (sin((1 + x) lambda) + sin((1 - x) lambda)) / 2`, et
    `int_0^oo sin(a lambda) / lambda = pi/2` fois le signe de `a`.
    """
    lam = sp.Symbol("lambda", positive=True)

    def au_bord(x: sp.Expr) -> sp.Expr:
        termes = (sp.sin((1 + x) * lam) / lam, sp.sin((1 - x) * lam) / lam)
        return sum(sp.integrate(t, (lam, 0, sp.oo)) for t in termes) / sp.pi

    assert au_bord(R(1, 2)) == 1
    assert au_bord(sp.Integer(2)) == 0
    produit = sp.cos(lam * X) * sp.sin(lam)
    somme = (sp.sin((1 + X) * lam) + sp.sin((1 - X) * lam)) / 2
    assert sp.simplify(sp.expand_trig(produit - somme)) == 0


def test_une_equation_d_euler_sans_terme_en_u_yy_est_refusee() -> None:
    with pytest.raises(ValueError, match="b nul"):
        EquationDEuler(1, 2, 0)
