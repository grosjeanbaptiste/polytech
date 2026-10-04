"""Mathematique, troisieme jeu de planches : la transformee de Laplace.

Oracles (`Mathematique/SLIDES/Slides_03_Transformee_de_Laplace.pdf`) :
l'origine par les series (planches 6-7) ; `L(1) = 1/p` (planche 10) ;
`L(x')` et `L(x'')` (planches 12 a 14) ; `x'' + 3x' + 2x = 1`,
`X = 1/(p(p^2 + 3p + 2))` (planches 15-16) ; les conditions d'existence
(planches 19 a 23) ; la table des planches 25 a 33 ; les proprietes des
planches 35 a 42.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.laplace.definition import (
    NOYAU,
    integrale_tronquee,
    serie_generatrice,
    transformee_par_l_integrale,
    version_continue_de_la_serie,
)
from polytech.maths.domain.laplace.existence import (
    est_de_type_exponentiel,
    est_localement_integrable,
    verifie_les_conditions_d_existence,
)
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import (
    d_une_fonction_periodique,
    de_l_integrale,
    de_la_convolution,
    de_la_derivee_n,
    du_produit_par_une_exponentielle,
    du_retard,
    par_la_formule_d_euler,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.signaux.domain import symboles as signaux
from polytech.signaux.domain.systemes import convolution
from polytech.signaux.domain.transformees import transformee_de_laplace

R = sp.Rational
A, OMEGA = sp.symbols("a omega", positive=True)
BORNE = sp.Symbol("beta", positive=True)


def test_le_noyau_de_laplace_est_une_exponentielle_decroissante() -> None:
    """Planche 5 : `K(p, t) = e^(-pt)`."""
    assert sp.exp(-P * T) == NOYAU


def test_la_serie_generatrice_d_une_suite() -> None:
    """Planche 6 : `a(n) = 1` donne `1/(1 - x)`, `a(n) = 1/n!` donne `e^x`."""
    x = sp.Symbol("x", positive=True)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    geometrique = serie_generatrice(sp.Integer(1), n, x)
    assert geometrique.subs(x, R(1, 3)) == R(3, 2)
    assert sp.simplify(serie_generatrice(1 / sp.factorial(n), n, x) - sp.exp(x)) == 0


def test_la_version_continue_de_la_serie_est_une_transformee_de_laplace() -> None:
    """Planche 7 : `int a(t) x^t dt` avec `x = e^(-s)` est `int a(t) e^(-st) dt`."""
    x = sp.Symbol("x", positive=True)
    continue_en_x = version_continue_de_la_serie(T, x)
    en_s = continue_en_x.subs(x, sp.exp(-P))
    assert sp.simplify(en_s - Transformee.de(T).image) == 0
    assert sp.simplify(en_s - 1 / P**2) == 0


def test_la_transformee_de_un_par_son_integrale() -> None:
    """Planche 10 : `[e^(-pt) / (-p)]_0^oo = 1/p`."""
    assert (
        sp.simplify(
            integrale_tronquee(sp.Integer(1), BORNE) - (1 - sp.exp(-P * BORNE)) / P
        )
        == 0
    )
    assert transformee_par_l_integrale(sp.Integer(1)) == 1 / P


def test_l_integrale_ne_converge_que_pour_une_partie_reelle_positive() -> None:
    """La planche 10 ecrit `Re(p) < 0` ; c'est `Re(p) > 0` qu'il faut.

    `int_0^beta e^(-st) dt = (1 - e^(-s beta))/s` tend vers `1/s` si
    `s > 0` et vers l'infini si `s < 0` -- la planche 23 dit d'ailleurs
    `Re(p) > alpha`.
    """
    s = sp.Symbol("s", real=True)
    tronquee = integrale_tronquee(sp.Integer(1), BORNE).subs(P, s)
    assert sp.limit(tronquee.subs(s, 2), BORNE, sp.oo) == R(1, 2)
    assert sp.limit(tronquee.subs(s, -2), BORNE, sp.oo) == sp.oo


def test_l_echelon_de_heaviside_a_la_transformee_de_un() -> None:
    """Planche 11 : la transformee ne voit que `t >= 0`."""
    assert Transformee.de(sp.Heaviside(T)).image == 1 / P


def test_la_transformee_est_lineaire() -> None:
    """Planche 9."""
    f, g = sp.exp(-2 * T), sp.sin(3 * T)
    somme = Transformee.de(5 * f - 4 * g).image
    separees = 5 * Transformee.de(f).image - 4 * Transformee.de(g).image
    assert sp.simplify(somme - separees) == 0


def test_la_transformee_des_derivees_des_planches_12_a_14() -> None:
    """`L(x') = pX - x(0+)` ; `L(x'') = p^2 X - p x(0+) - x'(0+)`."""
    image, x0, v0 = sp.symbols("X x_0 v_0")
    assert de_la_derivee_n(image, (x0,)) == P * image - x0
    assert (
        sp.expand(de_la_derivee_n(image, (x0, v0)) - (P**2 * image - P * x0 - v0)) == 0
    )


def test_la_formule_de_la_derivee_se_verifie_sur_une_fonction() -> None:
    """`x = cos(2t)` : `L(x'') = L(-4 cos 2t)`, avec `x(0) = 1`, `x'(0) = 0`."""
    fonction = sp.cos(2 * T)
    par_la_formule = de_la_derivee_n(Transformee.de(fonction).image, (1, 0))
    directe = Transformee.de(sp.diff(fonction, T, 2)).image
    assert sp.simplify(par_la_formule - directe) == 0


def test_la_derivee_d_ordre_n_de_la_planche_35() -> None:
    """`L(D^n f) = p^n L(f) - p^(n-1) f(0+) - ... - f^(n-1)(0+)`, pour `n = 3`."""
    image, f0, f1, f2 = sp.symbols("F f_0 f_1 f_2")
    attendu = P**3 * image - P**2 * f0 - P * f1 - f2
    assert sp.expand(de_la_derivee_n(image, (f0, f1, f2)) - attendu) == 0


def test_l_exemple_des_planches_15_et_16() -> None:
    """`x'' + 3x' + 2x = 1`, conditions nulles : `X = 1/(p(p^2 + 3p + 2))`."""
    probleme = ProblemeLineaire((1, 3, 2), sp.Integer(1), (0, 0))
    inconnue = probleme.inconnue
    equation = P**2 * inconnue + 3 * P * inconnue + 2 * inconnue - 1 / P
    assert sp.expand(probleme.equation_en_image() - equation) == 0
    assert sp.simplify(probleme.image() - 1 / (P * (P**2 + 3 * P + 2))) == 0


def test_la_transformee_inverse_de_l_exemple() -> None:
    """Planche 16 : « ensuite `X(p) -> x(t)` » ; `x = 1/2 - e^(-t) + e^(-2t)/2`."""
    probleme = ProblemeLineaire((1, 3, 2), sp.Integer(1), (0, 0))
    attendu = R(1, 2) - sp.exp(-T) + sp.exp(-2 * T) / 2
    assert sp.simplify(probleme.solution() - attendu) == 0
    assert probleme.solution_verifiee()


def test_des_conditions_initiales_non_nulles_entrent_dans_l_equation() -> None:
    """`x'' + x = 0`, `x(0) = 1`, `x'(0) = 0` : `x = cos t`."""
    probleme = ProblemeLineaire((1, 0, 1), sp.Integer(0), (1, 0))
    assert sp.simplify(probleme.image() - P / (P**2 + 1)) == 0
    assert sp.simplify(probleme.solution() - sp.cos(T)) == 0


def test_il_faut_autant_de_conditions_initiales_que_l_ordre() -> None:
    with pytest.raises(ValueError, match="conditions initiales pour"):
        ProblemeLineaire((1, 3, 2), sp.Integer(1), (0,))


def test_les_deux_fonctions_qui_posent_probleme_planche_19() -> None:
    """`1/t` n'est pas integrable pres de 0 ; `e^(t^2)` croit trop vite."""
    assert not est_localement_integrable(1 / T)
    assert est_localement_integrable(sp.exp(T**2))
    assert not est_de_type_exponentiel(sp.exp(T**2), 100)
    assert not verifie_les_conditions_d_existence(1 / T, 1)
    assert not verifie_les_conditions_d_existence(sp.exp(T**2), 100)


def test_les_fonctions_courantes_verifient_les_conditions_d_existence() -> None:
    """Planches 25 a 33 : `e^(at)`, `sin(wt)`, `t^n` -- « (!) » pour ce dernier."""
    assert verifie_les_conditions_d_existence(sp.exp(3 * T), 4)
    assert not est_de_type_exponentiel(sp.exp(3 * T), 2)
    assert verifie_les_conditions_d_existence(sp.sin(5 * T), R(1, 10))
    assert verifie_les_conditions_d_existence(T**7, R(1, 10))


@pytest.mark.parametrize(
    ("original", "image"),
    [
        (sp.exp(A * T), 1 / (P - A)),
        (sp.sin(OMEGA * T), OMEGA / (P**2 + OMEGA**2)),
        (sp.cos(OMEGA * T), P / (P**2 + OMEGA**2)),
        (T**3, sp.factorial(3) / P**4),
        (T**0, 1 / P),
    ],
)
def test_la_table_des_transformees_courantes(original: sp.Expr, image: sp.Expr) -> None:
    """Planches 26, 29, 31, 33 et le bilan de la planche 42."""
    assert sp.simplify(Transformee.de(original).image - image) == 0


def test_la_transformee_de_l_exponentielle_par_son_integrale() -> None:
    """Planche 26 : `1/(p - a)` pour `Re(p) > a` ; ici `a = 2` et `p = q + 2`."""
    q = sp.Symbol("q", positive=True)
    tronquee = integrale_tronquee(sp.exp(2 * T), BORNE).subs(P, q + 2)
    assert sp.limit(tronquee, BORNE, sp.oo) == 1 / q


def test_le_sinus_par_la_formule_d_euler() -> None:
    """Planches 28-29 : `sin(wt) = (e^(jwt) - e^(-jwt)) / 2j`, d'ou `w/(p^2 + w^2)`."""
    assert (
        sp.simplify(par_la_formule_d_euler(sp.sin, OMEGA) - OMEGA / (P**2 + OMEGA**2))
        == 0
    )
    assert (
        sp.simplify(par_la_formule_d_euler(sp.cos, OMEGA) - P / (P**2 + OMEGA**2)) == 0
    )


def test_la_formule_d_euler_ne_vaut_que_pour_le_sinus_et_le_cosinus() -> None:
    with pytest.raises(ValueError, match="sinus ou cosinus"):
        par_la_formule_d_euler(sp.tan, OMEGA)


def test_la_multiplication_par_une_exponentielle_decale_l_image() -> None:
    """Planche 35 : `L[e^(alpha t) f] = F(p - alpha)` ; ici `f = t^2`, `alpha = 3`."""
    image = Transformee.de(T**2).image
    decalee = du_produit_par_une_exponentielle(image, 3)
    assert sp.simplify(decalee - 2 / (P - 3) ** 3) == 0
    q = sp.Symbol("q", positive=True)
    directe = sp.integrate(sp.exp(3 * T) * T**2 * sp.exp(-(q + 3) * T), (T, 0, sp.oo))
    assert sp.simplify(decalee.subs(P, q + 3) - directe) == 0


def test_l_integrale_d_une_fonction_divise_l_image_par_p() -> None:
    """Planches 36-37 : `L(If) = L(f)/p`, `L(I^n f) = L(f)/p^n`."""
    image = Transformee.de(sp.cos(T)).image
    assert sp.simplify(de_l_integrale(image) - Transformee.de(sp.sin(T)).image) == 0
    deux_fois = Transformee.de(1 - sp.cos(T)).image
    assert sp.simplify(de_l_integrale(image, ordre=2) - deux_fois) == 0


def test_le_retard_multiplie_l_image_par_une_exponentielle() -> None:
    """Planches 38-39 : `L(tau_h f) = e^(-ph) L(f)` ; `f = t`, `h = 2`."""
    retardee = du_retard(Transformee.de(T).image, 2)
    assert retardee == sp.exp(-2 * P) / P**2
    directe = sp.integrate((T - 2) * sp.exp(-P * T), (T, 2, sp.oo))
    assert sp.simplify(retardee - directe) == 0


def test_un_retard_negatif_n_est_pas_une_fonction_causale() -> None:
    with pytest.raises(ValueError, match="retard positif"):
        du_retard(1 / P, -1)


def test_la_transformee_d_une_fonction_periodique() -> None:
    """Planches 40-41 : `F = G / (1 - e^(-pT))` ; la dent de scie de periode 1.

    Verification par un autre chemin : les deux premieres periodes,
    integrees directement, valent `G (1 + e^(-p))`.
    """
    motif = sp.integrate(T * sp.exp(-P * T), (T, 0, 1))
    periodique = d_une_fonction_periodique(motif, 1)
    assert sp.simplify(periodique * (1 - sp.exp(-P)) - motif) == 0
    seconde = sp.integrate((T - 1) * sp.exp(-P * T), (T, 1, 2))
    assert sp.simplify(motif + seconde - motif * (1 + sp.exp(-P))) == 0


def test_la_transformee_d_un_produit_de_convolution() -> None:
    """Planche 42 : `L(f * g) = L(f) L(g)` ; la convolution est celle des signaux."""
    t = signaux.TEMPS
    f, g = sp.exp(-t), t
    produit = convolution(f, g)
    image_du_produit = transformee_de_laplace(produit).subs(signaux.IMAGE, P)
    images = [transformee_de_laplace(x).subs(signaux.IMAGE, P) for x in (f, g)]
    assert sp.simplify(image_du_produit - de_la_convolution(*images)) == 0
    assert sp.simplify(de_la_convolution(*images) - 1 / (P**2 * (P + 1))) == 0


def test_les_deux_contextes_calculent_la_meme_transformee() -> None:
    """Le cours de signaux et celui de mathematique parlent de la meme `L`."""
    t = signaux.TEMPS
    for original in (sp.exp(-3 * T), T**2, sp.sin(2 * T)):
        chez_les_signaux = transformee_de_laplace(original.subs(T, t))
        en_maths = Transformee.de(original).image
        assert sp.simplify(chez_les_signaux.subs(signaux.IMAGE, P) - en_maths) == 0


def test_chaque_derivee_recoit_les_conditions_initiales_de_son_ordre() -> None:
    """`x'' + 3x' + 2x = 0`, `x(0) = 1`, `x'(0) = 0` : `x = 2 e^(-t) - e^(-2t)`.

    `L(x')` ne retranche que `x(0+)`, `L(x'')` retranche `p x(0+) + x'(0+)`.
    """
    probleme = ProblemeLineaire((1, 3, 2), sp.Integer(0), (1, 0))
    assert sp.simplify(probleme.image() - (P + 3) / (P**2 + 3 * P + 2)) == 0
    assert sp.simplify(probleme.solution() - (2 * sp.exp(-T) - sp.exp(-2 * T))) == 0
    assert probleme.solution_verifiee()


def test_verifier_l_equation_sans_les_conditions_ne_suffit_pas() -> None:
    """`e^(-t)` verifie `x'' + 3x' + 2x = 0` mais part avec la pente -1, non 0."""
    probleme = ProblemeLineaire((1, 3, 2), sp.Integer(0), (1, 0))
    assert not probleme.est_solution(sp.exp(-T))
    assert not probleme.est_solution(sp.cos(T))
    assert probleme.est_solution(2 * sp.exp(-T) - sp.exp(-2 * T))


def test_la_version_continue_d_une_suite_constante() -> None:
    """`a(t) = 1` : `int_0^oo x^t dt = -1/ln x`, soit `1/p` pour `x = e^(-p)`."""
    x = sp.Symbol("x", positive=True)
    continue_en_x = version_continue_de_la_serie(sp.Integer(1), x)
    assert sp.simplify(continue_en_x + 1 / sp.log(x)) == 0
    assert sp.simplify(continue_en_x.subs(x, sp.exp(-P)) - 1 / P) == 0
