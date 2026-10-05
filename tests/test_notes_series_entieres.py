"""Notes de cours de 1993, paragraphes 5.4, 6.4 et 6.5 : series binomiale et exemples.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 111 a
129 des notes (pages 125 a 143 du fichier, un scan lu en image). La serie
binomiale et l'equation `(1 + x) y' = a y` qui la caracterise, puis les
treize exemples du paragraphe 6.5 ; l'exemple 5.4.2 d'une serie qui
converge uniformement sans que sa serie derivee le fasse.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.integration.quadrature import (
    par_gauss,
    par_les_trapezes,
    par_simpson,
)
from polytech.maths.domain.series.alternees import somme_alternee
from polytech.maths.domain.series.binomiale import (
    coefficient_binomial,
    developpement_binomial,
    quotient_binomial,
)
from polytech.maths.domain.series.entieres import INDICE as N
from polytech.maths.domain.series.entieres import SerieEntiere
from polytech.maths.domain.series.produit import produit_de_cauchy
from polytech.maths.domain.series.uniforme import distance_uniforme
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
A = sp.Symbol("a", real=True)


def _mac_laurin(fonction: sp.Expr, ordre: int) -> sp.Expr:
    return sp.series(fonction, X, 0, ordre + 1).removeO()


def _integrande(x: float) -> float:
    """`(1 - e^(-x^2)) / x^2`, prolongee par 1 en 0."""
    return 1.0 if x == 0 else (1 - math.exp(-x * x)) / (x * x)


def test_les_coefficients_de_la_serie_binomiale() -> None:
    """6.4 : `(1 + x)^a = 1 + a x + a(a-1)/2! x^2 + ... + a(a-1)...(a-n+1)/n! x^n`."""
    assert coefficient_binomial(A, 0) == 1
    assert coefficient_binomial(A, 1) == A
    assert sp.expand(coefficient_binomial(A, 2) - A * (A - 1) / 2) == 0
    assert sp.expand(coefficient_binomial(A, 3) - A * (A - 1) * (A - 2) / 6) == 0


def test_les_coefficients_sont_ceux_de_mac_laurin() -> None:
    """`f^(n)(0) = a (a - 1) ... (a - n + 1)` pour `f = (1 + x)^a`."""
    for exposant in (R(1, 2), -R(1, 2), -1, R(5, 3)):
        attendu = _mac_laurin((1 + X) ** exposant, 5)
        assert sp.expand(developpement_binomial(exposant, 5) - attendu) == 0


def test_la_recurrence_issue_de_l_equation_differentielle() -> None:
    """Page 123 : `(1 + x) y' = a y` impose `n b_n = (a - n + 1) b_(n-1)`, `b_0 = 1`."""
    for n in range(1, 6):
        gauche = n * coefficient_binomial(A, n)
        droite = (A - n + 1) * coefficient_binomial(A, n - 1)
        assert sp.expand(gauche - droite) == 0
    somme = (1 + X) ** A
    assert sp.simplify((1 + X) * sp.diff(somme, X) - A * somme) == 0


def test_le_rayon_de_convergence_de_la_serie_binomiale_vaut_un() -> None:
    """Page 122 : `|a_(n+1) / a_n| = |a - n| / (n + 1)` tend vers 1."""
    quotient = quotient_binomial(A)
    assert sp.simplify(quotient - (A - N) / (N + 1)) == 0
    assert sp.limit(sp.Abs(quotient.subs(A, R(1, 2))), N, sp.oo) == 1
    rapport = coefficient_binomial(A, 4) / coefficient_binomial(A, 3)
    assert sp.simplify(rapport - quotient.subs(N, 3)) == 0


def test_un_exposant_entier_naturel_donne_un_polynome() -> None:
    """`(1 + x)^3` : les coefficients s'annulent au-dela du rang 3."""
    assert developpement_binomial(3, 6) == 1 + 3 * X + 3 * X**2 + X**3
    assert coefficient_binomial(3, 4) == 0


def test_les_exemples_1_a_3_du_6_5() -> None:
    """`1/(1 + x)`, puis `ln(1 + x)` et `arctan x` par integration terme a terme."""
    geometrique = SerieEntiere((-1) ** N)
    assert developpement_binomial(-1, 4) == geometrique.somme_partielle(X, 4)
    logarithme = geometrique.primitive()
    assert (
        sp.expand(logarithme.somme_partielle(X, 6) - _mac_laurin(sp.log(1 + X), 6)) == 0
    )
    carres = sum((-1) ** n * X ** (2 * n) for n in range(4))
    arctangente = sp.integrate(carres, X)
    assert sp.expand(arctangente - _mac_laurin(sp.atan(X), 7)) == 0


def test_la_serie_du_logarithme_au_bord() -> None:
    """Exemple 2 : en `x = 1` elle converge vers `ln 2` ; en `x = -1` elle diverge."""
    domaine = SerieEntiere((-1) ** N).primitive().domaine_de_convergence()
    assert domaine == sp.Interval.Lopen(-1, 1)
    k = sp.Symbol("k", integer=True, positive=True)
    assert sp.Sum((-1) ** (k + 1) / k, (k, 1, sp.oo)).doit() == sp.log(2)


def test_l_inverse_de_la_racine_de_un_plus_x() -> None:
    """Exemple 4 : `1 - x/2 + (1.3)/(2.2) x^2/2! - (1.3.5)/(2.2.2) x^3/3! + ...`."""
    attendu = 1 - X / 2 + R(3, 4) * X**2 / 2 - R(15, 8) * X**3 / 6
    assert sp.expand(developpement_binomial(-R(1, 2), 3) - attendu) == 0


def test_la_serie_de_l_arc_sinus() -> None:
    """Exemple 5 : `x + (1/2) x^3/3 + (1.3)/(2^2 2!) x^5/5 + ...`."""
    derivee = developpement_binomial(-R(1, 2), 3).subs(X, -(X**2))
    serie = sp.integrate(sp.expand(derivee), X)
    attendu = X + R(1, 2) * X**3 / 3 + R(3, 8) * X**5 / 5 + R(15, 48) * X**7 / 7
    assert sp.expand(serie - attendu) == 0
    assert sp.expand(serie - _mac_laurin(sp.asin(X), 7)) == 0


def test_la_serie_de_l_arc_sinus_approche_pi() -> None:
    """Exemple 5 : `pi / 6 = arcsin(1/2) = 1/2 + (1/2)^3 / (2.3) + ...`."""
    derivee = developpement_binomial(-R(1, 2), 12).subs(X, -(X**2))
    serie = sp.integrate(sp.expand(derivee), X)
    assert float(6 * serie.subs(X, R(1, 2))) == pytest.approx(math.pi, abs=1e-8)


def test_l_exemple_6_par_les_fractions_simples() -> None:
    """`1/((x - 1)(x - 2)) = somme (1 - (1/2)^(n+1)) x^n`, pour `|x| < 1`."""
    serie = SerieEntiere(1 - R(1, 2) ** (N + 1))
    attendu = _mac_laurin(1 / ((X - 1) * (X - 2)), 6)
    assert sp.expand(serie.somme_partielle(X, 6) - attendu) == 0
    assert serie.rayon_par_le_quotient() == 1


def test_l_exemple_7_par_deux_methodes() -> None:
    """`e^x sin x = somme 2^(n/2) sin(n pi / 4) x^n / n!`, et par produit de Cauchy."""
    directe = SerieEntiere(2 ** (N / 2) * sp.sin(N * sp.pi / 4) / sp.factorial(N))
    attendu = _mac_laurin(sp.exp(X) * sp.sin(X), 7)
    assert sp.expand(directe.somme_partielle(X, 7) - attendu) == 0
    exponentielle = SerieEntiere(1 / sp.factorial(N))
    sinus = SerieEntiere(sp.sin(N * sp.pi / 2) / sp.factorial(N))
    produit = produit_de_cauchy(exponentielle, sinus, 7)
    assert produit == tuple(directe.coefficient(n) for n in range(8))


def test_l_exemple_8_par_le_produit_de_cauchy() -> None:
    """`e^x / (1 - x)` : le coefficient de `x^n` est `1 + 1/1! + ... + 1/n!`."""
    produit = produit_de_cauchy(
        SerieEntiere(1 / sp.factorial(N)), SerieEntiere(sp.Integer(1)), 6
    )
    attendus = tuple(
        sum(R(1, math.factorial(k)) for k in range(n + 1)) for n in range(7)
    )
    assert produit == attendus
    assert float(produit[6] / produit[5]) == pytest.approx(1, abs=1e-3)
    assert float(produit[6]) == pytest.approx(math.e, abs=1e-3)


def test_l_integrale_de_l_exemple_9_par_sa_serie_alternee() -> None:
    """`I = int_0^1 (1 - e^(-x^2))/x^2 dx = somme (-1)^n / ((2n + 1)(n + 1)!)`.

    Les termes de la page 127 : 1 ; -0,1666 ; 0,0334 ; -5,9e-3 ; 9,25e-4 ;
    -1,26e-4 (le troisieme vaut 1/30, soit 0,0333 et non 0,0334). Le reste d'une
    serie alternee est majore par le premier terme neglige.
    """
    terme = (-1) ** N / ((2 * N + 1) * sp.factorial(N + 1))
    termes = [float(terme.subs(N, n)) for n in range(6)]
    attendus = [1, -0.1666, 0.0334, -5.9e-3, 9.25e-4, -1.26e-4]
    assert termes == pytest.approx(attendus, abs=1e-4)
    assert termes[2] == pytest.approx(0.0333, abs=5e-5)
    rang, somme, borne = somme_alternee(terme, N, R(5, 10**4))
    assert rang == 4
    assert borne == R(1, 7920)
    exacte = 0.8615277068
    assert abs(float(somme) - exacte) <= float(borne)
    assert 0.8613 < exacte < 0.8619


def test_la_somme_partielle_annoncee_page_127_vient_de_termes_arrondis() -> None:
    """Les notes ecrivent `u_0 + ... + u_4 = 0,8618` ; la somme exacte est 0,86164.

    0,8618 est la somme des cinq valeurs arrondies qu'elles affichent. Le scan
    a ete relu a plus haute resolution avant de l'ecrire.
    """
    terme = (-1) ** N / ((2 * N + 1) * sp.factorial(N + 1))
    _, somme, _ = somme_alternee(terme, N, R(5, 10**4))
    assert float(somme) == pytest.approx(0.86164, abs=5e-6)
    arrondis = 1 - 0.1666 + 0.0334 - 5.9e-3 + 9.25e-4
    assert arrondis == pytest.approx(0.8618, abs=5e-5)


def test_la_table_des_methodes_numeriques_de_la_page_127() -> None:
    """Trapezes 0,81606, Simpson 0,86188, Gauss 0,86153 sur un seul intervalle.

    La seconde ligne (0,85976 et 0,86153) est celle de cinq sous-intervalles :
    avec deux, les trapezes donnent 0,85043. La table l'intitule « 2 ».
    """
    assert par_les_trapezes(_integrande, 0, 1, 1) == pytest.approx(0.81606, abs=5e-6)
    assert par_simpson(_integrande, 0, 1, 1) == pytest.approx(0.86188, abs=5e-6)
    assert par_gauss(_integrande, 0, 1, 1) == pytest.approx(0.86153, abs=5e-6)
    assert par_les_trapezes(_integrande, 0, 1, 5) == pytest.approx(0.85976, abs=5e-6)
    assert par_simpson(_integrande, 0, 1, 5) == pytest.approx(0.86153, abs=5e-6)
    assert par_gauss(_integrande, 0, 1, 5) == pytest.approx(0.86153, abs=5e-6)
    assert par_les_trapezes(_integrande, 0, 1, 2) == pytest.approx(0.85043, abs=5e-6)


def test_les_quadratures_sont_exactes_sur_les_polynomes_de_leur_degre() -> None:
    """Trapezes : degre 1. Simpson : degre 3. Gauss a trois points : degre 5."""
    assert par_les_trapezes(lambda x: 3 * x + 1, 0, 2, 1) == pytest.approx(8)
    assert par_simpson(lambda x: x**3, 0, 2, 1) == pytest.approx(4)
    assert par_gauss(lambda x: x**5, 0, 2, 1) == pytest.approx(64 / 6)
    assert par_les_trapezes(lambda x: x**2, 0, 2, 1) != pytest.approx(8 / 3)
    assert par_simpson(lambda x: x**4, 0, 2, 1) != pytest.approx(32 / 5)


def test_un_nombre_de_sous_intervalles_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="au moins un sous-intervalle"):
        par_les_trapezes(_integrande, 0, 1, 0)


def test_l_exemple_10_et_la_limite_qu_il_donne() -> None:
    """`argsh x = x - (1/2) x^3/3 + ...` ; `lim (argsh x - arcsin x) / x^3 = -1/3`."""
    derivee = developpement_binomial(-R(1, 2), 3).subs(X, X**2)
    serie = sp.integrate(sp.expand(derivee), X)
    assert sp.expand(serie - _mac_laurin(sp.asinh(X), 7)) == 0
    assert sp.limit((sp.asinh(X) - sp.asin(X)) / X**3, X, 0) == -R(1, 3)


def test_la_somme_de_l_exemple_11() -> None:
    """`somme_(n>=1) (-1)^n x^(2n+1)/(4n^2 - 1) = x/2 - (x^2 + 1)/2 arctan x`."""
    fermee = X / 2 - (X**2 + 1) / 2 * sp.atan(X)
    partielle = sum((-1) ** n * X ** (2 * n + 1) / (4 * n**2 - 1) for n in range(1, 5))
    assert sp.expand(partielle - _mac_laurin(fermee, 9)) == 0
    k = sp.Symbol("k", integer=True, positive=True)
    en_un = sp.Sum((-1) ** k / (4 * k**2 - 1), (k, 1, sp.oo)).doit()
    assert sp.simplify(en_un - (R(1, 2) - sp.pi / 4)) == 0


def test_l_integrale_de_l_exemple_12() -> None:
    """`int_0^(pi/2) sin x / x = pi/2 - (pi/2)^3/(3.3!) + (pi/2)^5/(5.5!) - ...`."""
    demi = sp.pi / 2
    serie = sum(
        (-1) ** n * demi ** (2 * n + 1) / ((2 * n + 1) * sp.factorial(2 * n + 1))
        for n in range(8)
    )
    exacte = sp.Si(demi)
    assert float(serie) == pytest.approx(float(exacte), abs=1e-10)


def test_le_developpement_de_un_sur_x_autour_de_cinq() -> None:
    """Exemple 13 : `1/x = somme (-1)^n (x - 5)^n / 5^(n+1)`."""
    y = sp.Symbol("y")
    serie = SerieEntiere((-1) ** N / 5 ** (N + 1))
    attendu = sp.series(1 / (5 + y), y, 0, 6).removeO()
    assert sp.expand(serie.somme_partielle(y, 5) - attendu) == 0
    assert serie.rayon_par_le_quotient() == 5


def test_la_serie_telescopique_de_l_exemple_5_4_2() -> None:
    """`somme (x^n/n - x^(n+1)/(n+1)) = x` sur `[0, 1]` : `|S - S_n| <= 1/(n + 1)`."""
    for rang in (1, 4, 9):
        partielle = sum(X**n / n - X ** (n + 1) / (n + 1) for n in range(1, rang + 1))
        assert sp.expand(partielle - (X - X ** (rang + 1) / (rang + 1))) == 0
        ecart = distance_uniforme(partielle, X, X, sp.Interval(0, 1))
        assert ecart == R(1, rang + 1)


def test_sa_serie_derivee_ne_converge_pas_uniformement() -> None:
    """`somme (x^(n-1) - x^n)` a pour sommes `1 - x^n` : limite 0 en 1, non `S' = 1`."""
    for rang in (1, 4, 9):
        derivee = sum(X ** (n - 1) - X**n for n in range(1, rang + 1))
        assert sp.expand(derivee - (1 - X**rang)) == 0
        assert derivee.subs(X, 1) == 0
        assert distance_uniforme(derivee, 1, X, sp.Interval(0, 1)) == 1
    assert sp.diff(X, X) == 1


def test_une_serie_a_termes_positifs_n_a_pas_de_reste_alterne() -> None:
    """La majoration du reste ne vaut que si les signes alternent."""
    with pytest.raises(ValueError, match="serie non alternee"):
        somme_alternee(1 / sp.factorial(N), N, R(1, 100))
