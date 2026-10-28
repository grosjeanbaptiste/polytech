"""Syllabus de mathematiques appliquees, chapitre 5 : le point fixe de Banach.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 87 a 99. Les transformations contractantes et leurs iterees (les
deux tables des pages 89 et 94), la borne d'erreur (5.1), la norme
uniforme, les equations integrales et la methode de Picard (5.4, 5.5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import (
    egales,
    egales_arrondies,
    eloignees,
    proches,
    suites_proches,
)
from polytech.maths.domain.differentielles.cauchy import ProblemeDeCauchy
from polytech.maths.domain.differentielles.picard import (
    constante_de_lipschitz,
    contraction_de_picard,
    demi_largeur_garantie,
    voisinage_de_cauchy,
)
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE as Y
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as X
from polytech.maths.domain.point_fixe.contraction import (
    borne_de_l_erreur,
    constante_de_contraction,
    iterer,
)
from polytech.maths.domain.series.uniforme import distance_uniforme

if TYPE_CHECKING:
    from collections.abc import Callable

R = sp.Rational
DEMI_COSINUS = sp.cos(X) / 2
INVERSE_DU_COSH = 1 / sp.cosh(X)
TABLE_DU_COSINUS = (
    0.5,
    0.438791,
    0.452633,
    0.449649,
    0.450299,
    0.450158,
    0.450189,
    0.450182,
    0.450183,
    0.450183,
)
TABLE_DU_COSH = (
    0.6480543,
    0.8213958,
    0.7370586,
    0.7787255,
    0.7582408,
    0.7683419,
    0.7633676,
    0.7658189,
    0.7646114,
    0.7652063,
    0.7649132,
    0.7650576,
    0.7649865,
    0.7650215,
    0.7650043,
    0.7650128,
    0.7650086,
    0.7650106,
    0.7650096,
)


IMPRIME_PAGE_94 = 0.765009873  # x_inf imprime page 94 du syllabus de mathematique


def _demi_cosinus(x: float) -> float:
    return math.cos(x) / 2


def _inverse_du_cosh(x: float) -> float:
    return 1 / math.cosh(x)


def test_les_iterees_du_demi_cosinus() -> None:
    """Page 89 : de `x_0 = 0`, `x_1 = 0,5`, ... `x_10 = 0,450183`.

    Trois valeurs de la table ne sont pas l'arrondi a six decimales, d'une
    unite sur la derniere : `x_5 = 0,4502998`, `x_9 = 0,4501839` et
    `x_10 = 0,4501836` y sont ecrits `0,450299`, `0,450183`, `0,450183`.
    """
    suite = iterer(_demi_cosinus, 0.0, 10)
    assert suite[0] == 0
    assert suites_proches(suite[1:], TABLE_DU_COSINUS, 1e-6)
    mal_arrondies = {
        rang
        for rang, imprimee in enumerate(TABLE_DU_COSINUS, start=1)
        if not egales_arrondies(suite[rang], imprimee, 6)
    }
    assert mal_arrondies == {5, 9, 10}


def test_la_limite_des_iterees_est_un_point_fixe() -> None:
    """`u = T(u)` : 0,450183 annule `x - cos(x)/2`."""
    u = iterer(_demi_cosinus, 0.0, 60)[-1]
    assert proches(u, 0.450183, 1e-6)
    assert proches(_demi_cosinus(u), u, 1e-15)


def test_le_point_fixe_ne_depend_pas_du_depart() -> None:
    """Theoreme 5.1 : un et un seul point fixe, quel que soit `x_0`."""
    arrivees = [iterer(_demi_cosinus, depart, 80)[-1] for depart in (0, 1, 1.5)]
    assert suites_proches(arrivees, [arrivees[0]] * len(arrivees), 1e-12)


def test_le_demi_cosinus_est_contractant_de_constante_un_demi() -> None:
    """`|T(x) - T(y)| = |x - y| |sin c| / 2 <= |x - y| / 2`."""
    assert constante_de_contraction(DEMI_COSINUS, X, sp.Interval(0, sp.pi / 2)) == R(
        1, 2
    )


def test_l_inverse_du_cosh_est_contractant() -> None:
    """Page 93 : `T'' = 0` en `sinh x = 1`, ou `|T'| = 1/2` est maximal."""
    seconde = sp.diff(INVERSE_DU_COSH, X, 2)
    assert egales(seconde, (sp.sinh(X) ** 2 - 1) / sp.cosh(X) ** 3)
    constante = constante_de_contraction(INVERSE_DU_COSH, X, sp.Interval(0, sp.oo))
    assert egales(constante.rewrite(sp.exp), R(1, 2))


def test_les_iterees_de_l_inverse_du_cosh() -> None:
    """Page 94 : de `x_0 = 1`, la table jusqu'a `x_19`."""
    suite = iterer(_inverse_du_cosh, 1.0, 19)
    assert suites_proches(suite[1:], TABLE_DU_COSH, 5e-8)


def test_le_point_fixe_imprime_page_94_est_faux_a_la_huitieme_decimale() -> None:
    """Le syllabus ecrit `x_inf = 0,765009873...` ; c'est `0,765009955`.

    Second chemin : la racine de `x - 1/cosh x`, cherchee par Newton, et le
    residu de l'equation en chacune des deux valeurs.
    """
    limite = iterer(_inverse_du_cosh, 1.0, 200)[-1]
    racine = float(sp.nsolve(X - INVERSE_DU_COSH, X, 0.765))
    assert proches(limite, 0.765009955, 5e-10)
    assert proches(limite, racine, 1e-14)
    assert eloignees(IMPRIME_PAGE_94, _inverse_du_cosh(IMPRIME_PAGE_94), 1e-7)
    assert proches(limite, _inverse_du_cosh(limite), 1e-15)


@pytest.mark.parametrize(
    ("transformation", "depart", "limite"),
    [(_demi_cosinus, 0.0, 0.4501836112948736), (_inverse_du_cosh, 1.0, 0.7650098729)],
)
def test_la_borne_de_l_erreur_tient_a_chaque_rang(
    transformation: Callable[[float], float], depart: float, limite: float
) -> None:
    """Remarque 5.1 : `d(x_n, u) <= K^n d(x_0, x_1) / (1 - K)`, avec `K = 1/2`."""
    suite = iterer(transformation, depart, 15)
    premier_pas = abs(suite[1] - suite[0])
    for rang, valeur in enumerate(suite):
        assert proches(
            valeur, limite, borne_de_l_erreur(0.5, premier_pas, rang) + 1e-10
        )


def test_la_borne_decroit_d_autant_plus_vite_que_k_est_petit() -> None:
    assert borne_de_l_erreur(R(1, 2), 1, 3) == R(1, 4)
    assert borne_de_l_erreur(R(1, 10), 1, 3) == R(1, 900)


def test_une_transformation_qui_ne_contracte_pas_n_a_pas_de_borne() -> None:
    """`K < 1` est l'hypothese : `x -> 2x` eloigne les points."""
    assert constante_de_contraction(2 * X, X, sp.Interval(0, 1)) == 2
    for constante in (2, 1, -R(1, 2)):
        with pytest.raises(ValueError, match="transformation non contractante"):
            borne_de_l_erreur(constante, 1, 3)


def test_une_homothetie_de_rapport_inferieur_a_un_a_son_centre_pour_point_fixe() -> (
    None
):
    """Exemple 1, page 88, dans le plan."""
    centre = sp.Matrix([2, -1])

    def homothetie(point: sp.Matrix) -> sp.Matrix:
        return centre + R(1, 3) * (point - centre)

    assert homothetie(centre) == centre
    arrivee = iterer(homothetie, sp.Matrix([10, 7]), 40)[-1]
    assert (arrivee - centre).norm() < R(1, 10**15)


def test_la_distance_entre_fonctions_est_celle_de_la_convergence_uniforme() -> None:
    """Definition 5.5 et propriete 5.3.2 : symetrie, inegalite triangulaire."""
    intervalle = sp.Interval(0, 1)
    f, g, h = X**2, X, sp.Integer(1)

    def d(u: sp.Expr, v: sp.Expr) -> sp.Expr:
        return distance_uniforme(u, v, X, intervalle)

    assert d(f, g) == R(1, 4)
    assert d(f, g) == d(g, f)
    assert d(f, f) == 0
    assert d(f, g) <= d(f, h) + d(h, g)
    assert d(3 * f, 0) == 3 * d(f, 0)


def test_l_equation_integrale_de_la_page_94() -> None:
    """`y = 1 + int_0^t tau^2 y` : `1`, `1 + t^3/3`, `1 + t^3/3 + t^6/(3.6)`, ..."""
    probleme = ProblemeDeCauchy(X**2 * Y, x0=0, y0=1)
    iterees = probleme.iterees_de_picard(3)
    assert iterees[0] == 1
    assert iterees[1] == 1 + X**3 / 3
    assert iterees[2] == 1 + X**3 / 3 + X**6 / (3 * 6)
    assert iterees[3] == 1 + X**3 / 3 + X**6 / (3 * 6) + X**9 / (3 * 6 * 9)


def test_le_point_fixe_de_l_equation_integrale() -> None:
    """`e^(t^3/3)` verifie l'equation integrale, donc `u' = t^2 u`, `u(0) = 1`."""
    probleme = ProblemeDeCauchy(X**2 * Y, x0=0, y0=1)
    limite = sp.exp(X**3 / 3)
    assert probleme.verifie_l_equation_integrale(limite)
    assert probleme.est_solution(limite)
    developpement = sp.series(limite, X, 0, 10).removeO()
    assert sp.expand(probleme.iterees_de_picard(3)[-1] - developpement) == 0


def test_la_transformation_integrale_contracte_d_un_tiers_sur_zero_un() -> None:
    """Page 94 : `||T y - T z|| <= d(y, z) t^3 / 3 <= d(y, z) / 3` sur `[0, 1]`."""
    constante = contraction_de_picard(X**2 * Y, 0)
    delta = sp.Symbol("delta", positive=True)
    assert constante == delta**3 / 3
    assert constante.subs(delta, 1) == R(1, 3)


def test_la_contraction_se_constate_sur_deux_fonctions() -> None:
    """Deux fonctions : leurs images par `T` sont au moins trois fois plus proches."""
    intervalle = sp.Interval(0, 1)
    probleme = ProblemeDeCauchy(X**2 * Y, x0=0, y0=1)
    y, z = sp.Integer(1) + X, sp.Integer(1) - X**2

    def image(fonction: sp.Expr) -> sp.Expr:
        muette = sp.Symbol("u")
        integrande = probleme.second_membre.subs(Y, fonction).subs(X, muette)
        return 1 + sp.integrate(integrande, (muette, 0, X))

    avant = distance_uniforme(y, z, X, intervalle)
    apres = distance_uniforme(image(y), image(z), X, intervalle)
    assert apres <= avant / 3


def test_les_iterees_de_l_equation_5_2() -> None:
    """`y' = t y`, `y(0) = 1` : `1`, `1 + t^2/2`, `1 + t^2/2 + t^4/(2.4)`, ..."""
    probleme = ProblemeDeCauchy(X * Y, x0=0, y0=1)
    assert probleme.iterees_de_picard(2) == (1, 1 + X**2 / 2, 1 + X**2 / 2 + X**4 / 8)
    assert probleme.est_solution(sp.exp(X**2 / 2))
    assert contraction_de_picard(X * Y, 0).subs(
        sp.Symbol("delta", positive=True), 1
    ) == R(1, 2)


def test_le_theoreme_ne_garantit_la_solution_que_pres_du_depart() -> None:
    """Remarque 5.3 : `t^2/2 < 1`, soit `|t| < racine(2)` ; et `t^3/3 < 1` page 94."""
    assert demi_largeur_garantie(X * Y, 0) == sp.sqrt(2)
    assert demi_largeur_garantie(X**2 * Y, 0) == sp.cbrt(3)


def test_l_exemple_5_1() -> None:
    """`y' = a y`, `y(0) = C` : constante de Lipschitz `|a|`, limite `C e^(at)`."""
    a, c = sp.symbols("a C", real=True)
    probleme = ProblemeDeCauchy(a * Y, x0=0, y0=c)
    attendue = (c, c + a * c * X, c + a * c * X + a**2 * c * X**2 / 2)
    assert probleme.iterees_de_picard(2) == tuple(sp.expand(e) for e in attendue)
    assert probleme.est_solution(c * sp.exp(a * X))
    assert constante_de_lipschitz(a * Y, sp.S.Reals, sp.S.Reals) == sp.Abs(a)
    assert demi_largeur_garantie(3 * Y, 0) == R(1, 3)


def test_la_constante_de_lipschitz_d_un_second_membre_non_lineaire() -> None:
    """`f = y^2` sur `y` dans `[1 - h, 1 + h]`, `h = 1` : `|df/dy| = 2|y| <= 4`."""
    assert constante_de_lipschitz(Y**2, sp.S.Reals, sp.Interval(0, 2)) == 4
    assert constante_de_lipschitz(X * Y, sp.Interval(0, 1), sp.S.Reals) == 1


def test_une_derivee_qui_mele_les_deux_variables_est_refusee() -> None:
    with pytest.raises(ValueError, match="depend a la fois de x et de y"):
        constante_de_lipschitz(X * Y**2, sp.Interval(0, 1), sp.Interval(0, 1))


def test_le_voisinage_du_theoreme_de_cauchy() -> None:
    """Theoreme 5.4 : `|t - t_0| <= h_1 = h / M` et `<= h_2 = K / alpha`."""
    assert voisinage_de_cauchy(
        demi_cote=1, majorant=4, lipschitz=3, contraction=R(1, 2)
    ) == (
        R(1, 4),
        R(1, 6),
    )


def test_une_contraction_qui_n_est_pas_inferieure_a_un_est_refusee() -> None:
    """« Soit `K` un nombre quelconque tel que `0 <= K < 1` »."""
    for contraction in (1, -R(1, 2)):
        with pytest.raises(ValueError, match="K doit verifier 0 <= K < 1"):
            voisinage_de_cauchy(
                demi_cote=1, majorant=4, lipschitz=3, contraction=contraction
            )


def test_la_constante_de_picard_demande_un_second_membre_lineaire() -> None:
    """Pour `y' = y^2`, `|df/dy|` depend de `y` : c'est `constante_de_lipschitz`."""
    with pytest.raises(ValueError, match="non lineaire en y"):
        contraction_de_picard(Y**2, 0)


def test_la_contraction_se_mesure_depuis_le_point_de_depart() -> None:
    """`y' = t y` depuis `t_0 = 1` : `K = int_0^delta (1 + s) ds`."""
    delta = sp.Symbol("delta", positive=True)
    assert sp.expand(contraction_de_picard(X * Y, 1) - (delta + delta**2 / 2)) == 0
    assert demi_largeur_garantie(X * Y, 1) == sp.sqrt(3) - 1
