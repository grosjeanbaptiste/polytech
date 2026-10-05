"""Syllabus d'algebre, chapitre IV : les polynomes.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 48 a 54. Les
exemples resolus de la division (calcul ecrit, Horner, deductions
successives), de la recherche des racines et du lien entre racines et
coefficients, puis les « Solutions du 4eme chapitre » : chaque reponse
officielle des exercices 1 a 6 est recalculee.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.polynomes import (
    division_de_horner,
    fonctions_symetriques_des_racines,
)
from polytech.maths.domain.algebre.polynomes_exacts import (
    division,
    ordre_de_la_racine,
    tableau_de_horner,
)
from polytech.maths.domain.algebre.racines import (
    candidats_entiers,
    racines_du_bicarre,
)
from polytech.maths.domain.algebre.racines_exactes import (
    factoriser_dans_r,
    racines_du_bicarre_dans_c,
    racines_rationnelles,
    valeur_symetrique_des_racines,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
I = sp.I  # noqa: E741 -- l'unite imaginaire
QUINTIQUE = (6, -23, 24, 1, -12, 4)


def test_la_division_ecrite_de_la_page_48() -> None:
    """Quotient `x - 1/2`, reste `3/4 x^2 - 9/4 x + 3/2`."""
    dividende = X**4 - 2 * X**3 + 3 * X**2 - 4 * X + 2
    diviseur = X**3 - R(3, 2) * X**2 + R(3, 2) * X - 1
    quotient, reste = division(dividende, diviseur)
    assert quotient == X - R(1, 2)
    assert reste == R(3, 4) * X**2 - R(9, 4) * X + R(3, 2)
    assert sp.expand(quotient * diviseur + reste - dividende) == 0


def test_le_tableau_de_horner_de_la_page_49() -> None:
    """`4x^3 - 7x^2 + 3x - 5` par `x - 2` : les lignes `8 2 10` et `4 1 5 5`."""
    produits, sommes = tableau_de_horner((4, -7, 3, -5), 2)
    assert produits == (8, 2, 10)
    assert sommes == (4, 1, 5, 5)
    quotient, reste = division(4 * X**3 - 7 * X**2 + 3 * X - 5, X - 2)
    assert (quotient, reste) == (4 * X**2 + X + 5, 5)


def test_le_tableau_de_horner_concorde_avec_la_division_existante() -> None:
    """Lien : le schema flottant de `polynomes` rend le meme quotient, le meme reste."""
    quotient, reste = division_de_horner((4.0, -7.0, 3.0, -5.0), 2.0)
    _, sommes = tableau_de_horner((4, -7, 3, -5), 2)
    assert quotient == pytest.approx(sommes[:-1])
    assert reste == pytest.approx(sommes[-1])


def test_la_division_par_deductions_successives_des_pages_49_et_50() -> None:
    """`a = -3`, `b = -2`, `c = 3`, `d = 0`, et le reste `-4`."""
    dividende = 3 * X**5 - 6 * X**4 + X**3 + 5 * X**2 - 3 * X - 4
    quotient, reste = division(dividende, X - 1)
    assert quotient == 3 * X**4 - 3 * X**3 - 2 * X**2 + 3 * X
    assert reste == -4


def test_la_division_par_zero_est_refusee() -> None:
    with pytest.raises(ValueError, match="diviseur nul"):
        division(X**2 + 1, sp.Integer(0))


def test_le_bicarre_de_la_page_50() -> None:
    """`x^4 + x^2 - 2 = 0` : `y = 1` ou `-2`, d'ou `x = +-1` ou `+- racine(2) i`."""
    racines = racines_du_bicarre_dans_c(1, 1, -2)
    assert set(racines) == {1, -1, sp.sqrt(2) * I, -sp.sqrt(2) * I}
    assert len(racines) == 4


def test_dans_les_reels_seuls_les_y_positifs_sont_retenus() -> None:
    """Lien : le bicarre reel de `racines` ne garde que `+-1`."""
    assert set(racines_du_bicarre(1.0, 1.0, -2.0)) == {-1.0, 1.0}
    reelles = [r for r in racines_du_bicarre_dans_c(1, 1, -2) if r.is_real]
    assert set(reelles) == {-1, 1}


def test_un_y_nul_ne_donne_que_x_nul() -> None:
    """`x^4 - 4x^2 = 0` : `y = 0` ou `4`, d'ou `0`, `2` et `-2`."""
    assert set(racines_du_bicarre_dans_c(1, -4, 0)) == {0, 2, -2}
    assert len(racines_du_bicarre_dans_c(1, -4, 0)) == 3


def test_les_racines_du_polynome_a_coefficients_entiers_de_la_page_51() -> None:
    """`8x^4 - 18x^3 - 3x^2 + 17x - 6` : `-1`, `2`, `3/4` et `1/2`."""
    coefficients = (8, -18, -3, 17, -6)
    assert candidats_entiers(coefficients) == (-6, -3, -2, -1, 1, 2, 3, 6)
    polynome = sp.Poly(coefficients, X).as_expr()
    assert polynome.subs(X, 1) == -2
    assert polynome.subs(X, -1) == 0
    assert division(polynome, X + 1) == (8 * X**3 - 26 * X**2 + 23 * X - 6, 0)
    assert division(8 * X**3 - 26 * X**2 + 23 * X - 6, X - 2) == (
        8 * X**2 - 10 * X + 3,
        0,
    )
    assert racines_rationnelles(coefficients) == (-1, R(1, 2), R(3, 4), 2)


def test_le_cas_particulier_de_la_page_51() -> None:
    """`3x^3 + 7x^2 - 7x - 3 = (x - 1)(x + 3)(3x + 1)` : racines 1, -3 et -1/3."""
    assert racines_rationnelles((3, 7, -7, -3)) == (-3, -R(1, 3), 1)
    produit = (X - 1) * (X + 3) * (3 * X + 1)
    assert sp.expand(produit - (3 * X**3 + 7 * X**2 - 7 * X - 3)) == 0


def test_un_terme_constant_nul_donne_la_racine_zero() -> None:
    """`x^3 - x` : zero, puis les racines de `x^2 - 1`."""
    assert racines_rationnelles((1, 0, -1, 0)) == (-1, 0, 1)


def test_le_polynome_nul_n_a_pas_de_liste_de_racines() -> None:
    with pytest.raises(ValueError, match="tout nombre en est racine"):
        racines_rationnelles((0, 0, 0))


def test_un_polynome_sans_racine_rationnelle() -> None:
    """`x^2 - 2` : aucun des candidats `+-1`, `+-2` ne convient."""
    assert racines_rationnelles((1, 0, -2)) == ()


def test_le_lien_entre_racines_et_coefficients_de_la_page_52() -> None:
    """`6x^5 - 23x^4 + 24x^3 + x^2 - 12x + 4` : `23/6`, `4`, `-1/6`, `-2`, `-2/3`."""
    racines = sp.symbols("alpha1:6")
    attendues = (R(23, 6), 4, -R(1, 6), -2, -R(2, 3))
    for ordre, attendue in enumerate(attendues, start=1):
        elementaire = sp.Add(
            *(
                sp.Mul(*groupe)
                for groupe in sp.utilities.iterables.subsets(racines, ordre)
            )
        )
        assert (
            valeur_symetrique_des_racines(QUINTIQUE, elementaire, racines) == attendue
        )


def test_les_relations_concordent_avec_les_fonctions_symetriques_existantes() -> None:
    """Lien : `e_k = (-1)^k a_(n-k) / a_n`, deja porte en flottants."""
    existantes = fonctions_symetriques_des_racines(tuple(map(float, QUINTIQUE)))
    assert existantes[1:] == pytest.approx((23 / 6, 4, -1 / 6, -2, -2 / 3))


def test_une_expression_non_symetrique_n_a_pas_de_valeur() -> None:
    """`alpha_1 - alpha_2` depend de l'ordre des racines."""
    x1, x2, x3 = sp.symbols("x1:4")
    with pytest.raises(ValueError, match="expression non symetrique"):
        valeur_symetrique_des_racines((1, 0, 3, -2), x1 - x2, (x1, x2, x3))
    with pytest.raises(ValueError, match="autant de racines que le degre"):
        valeur_symetrique_des_racines((1, 0, 3, -2), x1 + x2, (x1, x2))


def test_la_reponse_de_l_exercice_1() -> None:
    """a) `x^2 + x - 2`, reste `-1` ; b) `3x^2 + 2`, reste `-21x^2 + 5x - 9`."""
    assert division(X**3 - 3 * X + 1, X - 1) == (X**2 + X - 2, -1)
    quotient, reste = division(3 * X**5 - 4 * X**3 + X + 5, X**3 - 2 * X + 7)
    assert quotient == 3 * X**2 + 2
    assert reste == -21 * X**2 + 5 * X - 9


def test_la_reponse_de_l_exercice_2() -> None:
    """a) `+-1`, `+-2` ; b) `7/3`, `2 +- racine(2)` ; c) `-2`, `3`, `-1/2`, `1/3`."""
    assert set(racines_du_bicarre_dans_c(1, -5, 4)) == {1, -1, 2, -2}
    assert racines_rationnelles((3, -19, 34, -14)) == (R(7, 3),)
    quotient, reste = division(3 * X**3 - 19 * X**2 + 34 * X - 14, 3 * X - 7)
    assert (quotient, reste) == (X**2 - 4 * X + 2, 0)
    assert set(sp.solve(quotient, X)) == {2 + sp.sqrt(2), 2 - sp.sqrt(2)}
    assert racines_rationnelles((6, -5, -38, -5, 6)) == (-2, -R(1, 2), R(1, 3), 3)


@pytest.mark.parametrize(
    ("polynome", "facteurs"),
    [
        (X**4 + 4 * X**2 + 3, (X**2 + 3, X**2 + 1)),
        (X**4 + X**3 - 11 * X**2 + X - 12, (X - 3, X + 4, X**2 + 1)),
        (
            6 * X**4 - 13 * X**3 + 13 * X - 6,
            (X - 1, X + 1, 3 * X - 2, 2 * X - 3),
        ),
        (X**4 + X**2 + 1, (X**2 + X + 1, X**2 - X + 1)),
    ],
    ids=["3a", "3b", "3c", "3d"],
)
def test_la_reponse_de_l_exercice_3(
    polynome: sp.Expr, facteurs: tuple[sp.Expr, ...]
) -> None:
    """Les facteurs officiels, rendus unitaires, sont ceux de la factorisation."""
    constante, obtenus = factoriser_dans_r(polynome)
    unitaires = {sp.Poly(f, X).monic().as_expr() for f in facteurs}
    assert set(obtenus) == unitaires
    assert len(obtenus) == len(facteurs)
    assert sp.expand(constante * sp.Mul(*obtenus) - polynome) == 0


def test_une_factorisation_reelle_qui_sort_des_rationnels() -> None:
    """`x^4 + 1 = (x^2 + racine(2) x + 1)(x^2 - racine(2) x + 1)`."""
    constante, facteurs = factoriser_dans_r(X**4 + 1)
    assert constante == 1
    attendus = {X**2 + sp.sqrt(2) * X + 1, X**2 - sp.sqrt(2) * X + 1}
    assert {sp.expand(f) for f in facteurs} == attendus


def test_une_racine_multiple_donne_un_facteur_repete() -> None:
    """`2 (x - 1)^2 (x^2 + 1)`."""
    constante, facteurs = factoriser_dans_r(sp.expand(2 * (X - 1) ** 2 * (X**2 + 1)))
    assert constante == 2
    assert sorted(facteurs, key=str) == sorted((X - 1, X - 1, X**2 + 1), key=str)


def test_une_paire_conjuguee_double_donne_deux_fois_son_facteur() -> None:
    """`(x^2 + 1)^2` : exercice 4, ou `i` et `-i` sont racines doubles."""
    assert factoriser_dans_r(sp.expand((X**2 + 1) ** 2)) == (1, (X**2 + 1, X**2 + 1))


def test_le_signe_du_terme_du_premier_degre_d_un_facteur_carre() -> None:
    """`x^3 - 1 = (x - 1)(x^2 + x + 1)` : la page 51 s'en sert pour regrouper."""
    constante, facteurs = factoriser_dans_r(X**3 - 1)
    assert constante == 1
    assert set(facteurs) == {X - 1, X**2 + X + 1}


def test_des_racines_sans_forme_close_sont_refusees() -> None:
    """`x^5 - x + 1` : pas de racines par radicaux, donc pas de factorisation rendue."""
    with pytest.raises(ValueError, match="racines introuvables"):
        factoriser_dans_r(X**5 - X + 1)


def test_la_reponse_de_l_exercice_4() -> None:
    """`i` racine double de `x^6 + x^5 + 2x^3 - 3x^2 + x - 2` : restent 1 et -2."""
    polynome = X**6 + X**5 + 2 * X**3 - 3 * X**2 + X - 2
    assert ordre_de_la_racine(polynome, I) == 2
    assert ordre_de_la_racine(polynome, -I) == 2
    quotient, reste = division(polynome, sp.expand((X**2 + 1) ** 2))
    assert (quotient, reste) == (X**2 + X - 2, 0)
    assert set(sp.solve(quotient, X)) == {1, -2}
    assert ordre_de_la_racine(polynome, sp.Integer(1)) == 1
    assert ordre_de_la_racine(polynome, sp.Integer(3)) == 0


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_la_reponse_de_l_exercice_5(n: int) -> None:
    """`a z^n + b z + 1` divisible par `(z - 1)^2` : `a = 1/(n-1)`, `b = -n/(n-1)`."""
    a, b = sp.symbols("a b")
    polynome = a * X**n + b * X + 1
    conditions = [polynome.subs(X, 1), sp.diff(polynome, X).subs(X, 1)]
    solution = sp.solve(conditions, [a, b], dict=True)
    assert solution == [{a: R(1, n - 1), b: -R(n, n - 1)}]
    assert ordre_de_la_racine(polynome.subs(solution[0]), sp.Integer(1)) >= 2
    assert division(polynome.subs(solution[0]), (X - 1) ** 2)[1] == 0


def test_le_polynome_nul_n_a_pas_d_ordre_de_racine() -> None:
    with pytest.raises(ValueError, match="polynome nul"):
        ordre_de_la_racine(sp.Integer(0), sp.Integer(1))


def test_la_reponse_de_l_exercice_6() -> None:
    """Racines de `x^3 + 3x - 2` : a) `-6` ; b) `9` ; c) `3/2`, sans les calculer."""
    racines = x1, x2, x3 = sp.symbols("x1:4")
    cubique = (1, 0, 3, -2)
    carres = x1**2 + x2**2 + x3**2
    assert valeur_symetrique_des_racines(cubique, carres, racines) == -6
    produits = x1**2 * x2**2 + x1**2 * x3**2 + x2**2 * x3**2
    assert valeur_symetrique_des_racines(cubique, produits, racines) == 9
    inverses = 1 / x1 + 1 / x2 + 1 / x3
    assert valeur_symetrique_des_racines(cubique, inverses, racines) == R(3, 2)


def test_la_reponse_de_l_exercice_6_par_les_racines_elles_memes() -> None:
    """Second chemin : les trois racines numeriques de `x^3 + 3x - 2`."""
    racines = [complex(r) for r in sp.Poly(X**3 + 3 * X - 2, X).nroots()]
    assert sum(r**2 for r in racines) == pytest.approx(-6)
    assert sum(1 / r for r in racines) == pytest.approx(1.5)
