"""Les réponses des examens de Mathématiques III, refaites deux fois.

La fiche de révision (`cursus/fiches/MathematiquesIngenieur3.toml`) cite ces
résultats ; chacun est calculé ici par la bibliothèque (les résidus, le
problème de Cauchy), puis recoupé par un second chemin qui ne la doit pas :
une intégrale numérique sur la paramétrisation, une substitution dans
l'équation. Les sujets sont lus sur l'image des pages : leur texte converti
perd les formules.
"""

from __future__ import annotations

import cmath
import math

import mpmath
import sympy as sp

from polytech.maths.domain.analyse_complexe.quotient import residu_d_un_quotient
from polytech.maths.domain.analyse_complexe.residus import (
    integrale_par_les_residus,
    residu_d_un_pole,
    residu_d_un_pole_simple,
)
from polytech.maths.domain.analyse_complexe.singularites import ordre_du_pole
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire

z, t = sp.symbols("z t")


def _sur_le_cercle(
    f: sp.Expr, centre: complex, rayon: float, tours: int = 1
) -> complex:
    """L'intégrale de `f` sur le cercle, par la paramétrisation, sans résidu."""
    g = sp.lambdify(z, f, "mpmath")

    def integrande(u: float) -> complex:
        point = rayon * mpmath.e ** (1j * u)
        return complex(g(centre + point) * 1j * point)

    # Le cercle frole parfois un pole : l'intervalle est decoupe, pour la precision.
    return complex(
        mpmath.quad(integrande, mpmath.linspace(0, 2 * mpmath.pi * tours, 65))
    )


def test_l_examen_blanc_n_entoure_que_0_et_tourne_deux_fois() -> None:
    """3 décembre 2025, 1 (b)(i) : le corrigé officiel conclut 4 pi i."""
    f = 1 / (z + z**2)
    centre, rayon = 1 + 1j, 2.0
    assert abs(0 - centre) < rayon < abs(-1 - centre)
    par_les_residus = 2 * integrale_par_les_residus(f, z, [0])
    assert sp.simplify(par_les_residus - 4 * sp.pi * sp.I) == 0
    assert cmath.isclose(
        _sur_le_cercle(f, centre, rayon, tours=2), 4j * math.pi, abs_tol=1e-9
    )


def test_l_integrale_de_1_sur_1_plus_x_puissance_n_par_le_secteur() -> None:
    """Janvier et août 2023 : (1 - e^(2 i pi/n)) I = 2 pi i Res(f, e^(i pi/n))."""
    for n in (2, 3, 8):
        pole = sp.exp(sp.I * sp.pi / n)
        secteur = (
            2 * sp.pi * sp.I * residu_d_un_quotient(sp.Integer(1), 1 + z**n, z, pole)
        )
        par_le_secteur = complex(sp.N(secteur / (1 - sp.exp(2 * sp.I * sp.pi / n))))
        numerique = float(mpmath.quad(lambda x, n=n: 1 / (1 + x**n), [0, mpmath.inf]))
        assert cmath.isclose(par_le_secteur, numerique, rel_tol=1e-10)
        assert math.isclose(
            numerique, (math.pi / n) / math.sin(math.pi / n), rel_tol=1e-10
        )


def test_l_inverse_de_laplace_de_janvier_2023_par_les_residus() -> None:
    """4 b) et c) : le décalage et les résidus en 1 + 2i et 1 - 2i concordent."""
    image = z / ((z - 1) ** 2 + 4)
    poles = (1 + 2 * sp.I, 1 - 2 * sp.I)
    temps = sp.Symbol("t", positive=True)  # reel : l'exponentielle se separe
    termes = (residu_d_un_pole_simple(image * sp.exp(z * temps), z, p) for p in poles)
    par_les_residus = sp.expand_complex(sum(termes, sp.Integer(0)))
    par_le_decalage = sp.exp(temps) * (sp.cos(2 * temps) + sp.sin(2 * temps) / 2)
    assert sp.simplify(par_les_residus - par_le_decalage) == 0


def test_le_probleme_de_cauchy_de_l_examen_blanc() -> None:
    """Examen blanc, question 3 : u''' - 4u'' + u' + 6u = -3.

    Conditions initiales : u(0) = -1, u'(0) = 0, u''(0) = -2.
    """
    u = ProblemeLineaire((1, -4, 1, 6), sp.Integer(-3), (-1, 0, -2)).solution()
    (temps,) = u.free_symbols  # le temps de la bibliotheque porte ses hypotheses
    u = u.subs(temps, t)
    attendue = (
        -sp.exp(3 * t) / 4 + sp.exp(2 * t) / 6 - sp.Rational(1, 2) - 5 * sp.exp(-t) / 12
    )
    assert sp.simplify(u - attendue) == 0
    equation = (
        sp.diff(attendue, t, 3)
        - 4 * sp.diff(attendue, t, 2)
        + sp.diff(attendue, t)
        + 6 * attendue
    )
    assert sp.simplify(equation + 3) == 0
    assert [sp.diff(attendue, t, k).subs(t, 0) for k in range(3)] == [-1, 0, -2]


def test_le_logarithme_de_janvier_2026_sur_le_petit_cercle() -> None:
    """3 b) : Log z / (z^2 + 1)^2, pôle double en i, cercle |z - i| = 1/2."""
    g = sp.log(z) / (z**2 + 1) ** 2
    assert ordre_du_pole(g, z, sp.I) == 2
    par_les_residus = complex(sp.N(integrale_par_les_residus(g, z, [sp.I])))
    assert cmath.isclose(
        par_les_residus, complex(sp.N(sp.pi * (-2 + sp.I * sp.pi) / 4)), abs_tol=1e-12
    )
    assert cmath.isclose(_sur_le_cercle(g, 1j, 0.5), par_les_residus, abs_tol=1e-9)


def test_la_transformee_de_1_sur_1_plus_x_carre() -> None:
    """Août 2023 : racine(pi/2) e^(-|s|), convention unitaire du cours."""
    for s in (0.5, 1.0, 2.0):
        # Integrande pair et oscillant : deux fois la demi-droite, par quadosc.
        demi = mpmath.quadosc(
            lambda x, s=s: mpmath.cos(s * x) / (1 + x**2), [0, mpmath.inf], omega=s
        )
        numerique = 2 * float(demi)
        assert math.isclose(
            numerique / math.sqrt(2 * math.pi),
            math.sqrt(math.pi / 2) * math.exp(-s),
            rel_tol=1e-8,
        )


# --- Les cas particuliers que la fiche reprend -------------------------------


def test_janvier_2026_1b_la_fonction_n_est_autre_que_cos_z() -> None:
    x, y = sp.symbols("x y", real=True)
    f = sp.cos(x) * sp.cosh(y) - sp.I * sp.sin(x) * sp.sinh(y)
    assert sp.simplify(sp.expand_complex(sp.cos(x + sp.I * y)) - f) == 0


def test_janvier_2023_2_la_serie_de_la_couronne_vaut_f_jusqu_a_4() -> None:
    """1/(z-1) en 1/z, 1/(z+4i) en z/4i : la serie vaut f pour 1 < |z| < 4."""
    f = 1 / ((z - 1) * (z + 4 * sp.I))
    coefficient = 1 / (1 + 4 * sp.I)  # 1/((z-1)(z+4i)) = c (1/(z-1) - 1/(z+4i))
    assert sp.simplify(coefficient * (1 / (z - 1) - 1 / (z + 4 * sp.I)) - f) == 0
    for point in (1.5, 2.5j, -3.5):
        un = sum(point ** (-k - 1) for k in range(3000))
        deux = sum((-point / (4j)) ** k for k in range(3000)) / (4j)
        valeur = complex(coefficient) * (un - deux)
        assert cmath.isclose(valeur, complex(f.subs(z, point)), rel_tol=1e-9)
    assert f.subs(z, 0) == 1 / (-4 * sp.I)  # 0 n'est pas une singularite


def test_examen_blanc_fourier_1_le_resultat_depend_du_rayon() -> None:
    f = sp.cos(z) / (z * (z - sp.pi) ** 2)
    assert cmath.isclose(_sur_le_cercle(f, 0, 2.0), 2j / math.pi, abs_tol=1e-9)
    assert cmath.isclose(_sur_le_cercle(f, 0, 4.0), 4j / math.pi, abs_tol=1e-9)
    assert sp.series(1 / (1 - z) ** 3, z, 0, 4).removeO().coeff(z, -1) == 0


def test_aout_2023_3_l_integrale_de_1_sur_z_depend_de_rho() -> None:
    assert cmath.isclose(_sur_le_cercle(1 / z, 2j, 3.0), 2j * math.pi, abs_tol=1e-9)
    assert cmath.isclose(_sur_le_cercle(1 / z, 2j, 1.0), 0, abs_tol=1e-9)


def test_l_integrale_du_conjugue_vaut_2_pi_i_rho_carre_par_tour() -> None:
    assert cmath.isclose(
        _sur_le_cercle(sp.conjugate(z), 2j, 1.5), 2j * math.pi * 1.5**2, abs_tol=1e-9
    )
    assert cmath.isclose(
        _sur_le_cercle(sp.conjugate(z), 1 + 1j, 2.0, tours=2),
        16j * math.pi,
        abs_tol=1e-9,
    )


def test_janvier_2026_1d_un_pole_d_ordre_3_de_residu_i() -> None:
    a = 1 + sp.I * sp.sqrt(2)
    g = sp.I / (z - a) + 1 / (z - a) ** 3
    assert ordre_du_pole(g, z, a) == 3
    assert sp.simplify(residu_d_un_pole(g, z, a, ordre=3) - sp.I) == 0


def test_janvier_2026_4a_les_coefficients_du_prolongement_impair() -> None:
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", integer=True, positive=True)
    b = 2 / sp.pi * sp.integrate(x**3 * sp.sin(k * x), (x, 0, sp.pi))
    assert sp.simplify(b - (-1) ** k * (12 - 2 * sp.pi**2 * k**2) / k**3) == 0


def test_examen_blanc_fourier_2_la_regularisee_de_e_x_en_pi() -> None:
    x = sp.Symbol("x", real=True)
    n = sp.Symbol("n", integer=True, positive=True)
    a = sp.integrate(sp.exp(x) * sp.cos(n * x), (x, -sp.pi, sp.pi)) / sp.pi
    a0 = sp.integrate(sp.exp(x), (x, -sp.pi, sp.pi)) / sp.pi
    en_pi = a0 / 2 + sp.Sum(sp.simplify(a * (-1) ** n), (n, 1, sp.oo)).doit()
    assert sp.simplify((en_pi - sp.cosh(sp.pi)).rewrite(sp.exp)) == 0
    somme = sp.Sum(1 / (n**2 + 1), (n, 1, sp.oo))
    assert math.isclose(
        float(somme.evalf()), (math.pi / math.tanh(math.pi) - 1) / 2, rel_tol=1e-12
    )
    alternee = sp.Sum((-1) ** n / (n**2 + 1), (n, 1, sp.oo))
    assert math.isclose(
        float(alternee.evalf()), (math.pi / math.sinh(math.pi) - 1) / 2, rel_tol=1e-12
    )


def test_aout_2023_4_parseval_sur_la_valeur_absolue() -> None:
    n = sp.Symbol("n", integer=True, nonnegative=True)
    impairs = sp.Sum(1 / (2 * n + 1) ** 4, (n, 0, sp.oo)).doit()
    assert sp.simplify(impairs - sp.pi**4 / 96) == 0
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", integer=True, positive=True)
    a = sp.simplify(2 / sp.pi * sp.integrate(x * sp.cos(k * x), (x, 0, sp.pi)))
    assert sp.simplify(a.subs(k, 3) + 4 / (9 * sp.pi)) == 0
    assert a.subs(k, 2) == 0


def test_examen_blanc_fourier_3_les_poles_de_l_image() -> None:
    """L(u(3)) = z^3 U + z^2 + 2 ; les poles de U sont 0, -1, 2 et 3."""
    u0, u1, u2 = -1, 0, -2
    image = sp.Symbol("U")
    derivees = [
        image,
        z * image - u0,
        z**2 * image - z * u0 - u1,
        z**3 * image - z**2 * u0 - z * u1 - u2,
    ]
    assert sp.expand(derivees[3] - (z**3 * image + z**2 + 2)) == 0
    equation = derivees[3] - 4 * derivees[2] + derivees[1] + 6 * derivees[0] + 3 / z
    (solution,) = sp.solve(equation, image)
    assert set(sp.roots(sp.denom(sp.together(solution)), z)) == {0, -1, 2, 3}
