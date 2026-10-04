"""Syllabus de mathematiques appliquees, chapitre 3 : la transformee de Laplace.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 50 a 63 et 71 a 75. Ce que le syllabus ajoute aux planches : les
signaux retardes des exemples 3.4, les deux regles de Heaviside (3.5), les
equations resolues du 3.6, le lemme et le theoreme de la valeur initiale
(3.10), les coefficients non constants et les polynomes de Laguerre (3.11).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.laplace.coefficients_variables import (
    de_t_fois_la_derivee,
    de_t_fois_la_derivee_seconde,
    du_produit_par_t,
)
from polytech.maths.domain.laplace.heaviside import (
    fractions_simples,
    par_fractions_simples,
    par_la_serie,
)
from polytech.maths.domain.laplace.laguerre import (
    image_de_laguerre,
    polynome_de_laguerre,
)
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import du_retard
from polytech.maths.domain.laplace.theoremes import (
    limite_a_l_infini,
    limite_en_zero,
    poles_genants,
    valeur_finale,
    valeur_initiale,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
exp = sp.exp


def _par_l_integrale(morceaux: list[tuple[sp.Expr, object, object]]) -> sp.Expr:
    """La transformee d'un signal par morceaux, par la definition (3.1)."""
    return sum(
        (sp.integrate(f * exp(-P * T), (T, a, b)) for f, a, b in morceaux),
        sp.Integer(0),
    )


def test_l_echelon_retarde_de_l_exemple_3_4() -> None:
    """`L(tau_2 H) = e^(-2p) / p`."""
    retarde = du_retard(1 / P, 2)
    assert retarde == exp(-2 * P) / P
    assert sp.simplify(retarde - _par_l_integrale([(sp.Integer(1), 2, sp.oo)])) == 0


def test_le_creneau_d_aire_unite_de_l_exemple_3_4() -> None:
    """`n H - n tau_(1/n) H` a pour image `(n/p)(1 - e^(-p/n))`."""
    n = sp.Symbol("n", positive=True)
    image = n / P - du_retard(n / P, 1 / n)
    assert sp.simplify(image - n / P * (1 - exp(-P / n))) == 0
    assert sp.simplify(image - _par_l_integrale([(n, 0, 1 / n)])) == 0


def test_le_triangle_de_l_exemple_3_4() -> None:
    """`t - 2 tau_1 t + tau_2 t` a pour image `(1 - e^(-p))^2 / p^2`."""
    rampe = 1 / P**2
    image = rampe - 2 * du_retard(rampe, 1) + du_retard(rampe, 2)
    assert sp.simplify(image - (1 - exp(-P)) ** 2 / P**2) == 0
    directe = _par_l_integrale([(T, 0, 1), (2 - T, 1, 2)])
    assert sp.simplify(image - directe) == 0


def test_la_premiere_regle_de_heaviside_retrouve_le_sinus() -> None:
    """3.5.1 : `w/(p^2 + w^2) = somme (-1)^n w^(2n+1) / p^(2n+2)` donne `sin(wt)`."""
    w = sp.Symbol("omega", positive=True)
    original = par_la_serie(w / (P**2 + w**2), 8)
    attendu = sp.series(sp.sin(w * T), T, 0, 8).removeO()
    assert sp.expand(original - attendu) == 0
    assert original.coeff(T, 3) == -(w**3) / 6


def test_la_premiere_regle_de_heaviside_sur_une_exponentielle() -> None:
    """`1/(p - 2) = 1/p + 2/p^2 + 4/p^3 + ...` : le debut de `e^(2t)`."""
    original = par_la_serie(1 / (P - 2), 4)
    assert sp.expand(original - (1 + 2 * T + 2 * T**2 + R(4, 3) * T**3)) == 0


def test_la_deuxieme_regle_de_heaviside() -> None:
    """3.5.2 : `(3p + 7)/(p^2 - 2p - 3) = 4/(p - 3) - 1/(p + 1)`."""
    image = (3 * P + 7) / (P**2 - 2 * P - 3)
    assert set(fractions_simples(image)) == {4 / (P - 3), -1 / (P + 1)}
    assert sp.simplify(par_fractions_simples(image) - (4 * exp(3 * T) - exp(-T))) == 0


def test_la_deuxieme_regle_demande_un_denominateur_de_plus_haut_degre() -> None:
    """« tels que le degre de Q soit superieur au degre de P »."""
    with pytest.raises(ValueError, match="degre du numerateur"):
        par_fractions_simples((P**2 + 1) / (P**2 - 1))


def test_la_premiere_equation_du_3_6() -> None:
    """`y''' + 3y'' + 2y' = 0`, `y(0) = 1`, derivees nulles : `Y = 1/p`, `y = 1`."""
    probleme = ProblemeLineaire((1, 3, 2, 0), sp.Integer(0), (1, 0, 0))
    assert probleme.image() == 1 / P
    assert probleme.solution() == 1


def test_la_deuxieme_equation_du_3_6() -> None:
    """`y''' + 3y'' + 2y' = e^t`, au repos : `-1/2 + e^t/6 + e^(-t)/2 - e^(-2t)/6`."""
    probleme = ProblemeLineaire((1, 3, 2, 0), exp(T), (0, 0, 0))
    attendu = -R(1, 2) + exp(T) / 6 + exp(-T) / 2 - exp(-2 * T) / 6
    assert sp.simplify(probleme.image() - 1 / (P * (P - 1) * (P + 1) * (P + 2))) == 0
    assert sp.simplify(probleme.solution() - attendu) == 0
    assert probleme.est_solution(attendu)


def test_le_probleme_resolu_sous_matlab_au_3_12() -> None:
    """`y'' + 4y' + 3y = 0`, `y(0) = 3`, `y'(0) = 1` : `5e^(-t) - 2e^(-3t)`."""
    probleme = ProblemeLineaire((1, 4, 3), sp.Integer(0), (3, 1))
    assert sp.simplify(probleme.image() - (3 * P + 13) / (P**2 + 4 * P + 3)) == 0
    attendu = 5 * exp(-T) - 2 * exp(-3 * T)
    assert sp.simplify(par_fractions_simples(probleme.image()) - attendu) == 0
    assert probleme.est_solution(attendu)


def test_le_lemme_du_3_10() -> None:
    """Si `f` est de type exponentiel, `L(f)(p)` tend vers 0 quand `Re(p)` croit."""
    for original in (sp.Integer(1), T**3, exp(2 * T), sp.cos(T)):
        assert limite_a_l_infini(Transformee.de(original).image) == 0


def test_le_theoreme_de_la_valeur_initiale() -> None:
    """Exemple : `f = e^(-at)`, `F = 1/(p + a)`, `lim p F(p) = 1 = f(+0)`."""
    a = sp.Symbol("a", positive=True)
    assert valeur_initiale(1 / (P + a)) == 1
    for original in (sp.cos(T), 3 + T, exp(-T) * sp.sin(T), 5 * exp(2 * T)):
        image = Transformee.de(original).image
        assert valeur_initiale(image) == original.subs(T, 0)


def test_la_valeur_initiale_n_existe_pas_pour_une_impulsion() -> None:
    """`F = 1` est l'image de l'impulsion de Dirac : `p F(p)` diverge."""
    with pytest.raises(ValueError, match="pas de limite finie"):
        valeur_initiale(sp.Integer(1))


def test_le_theoreme_de_la_valeur_finale_sur_l_exemple_du_syllabus() -> None:
    """`f = e^(-at)` : `lim p/(p + a) = 0` quand `p` tend vers 0."""
    a = sp.Symbol("a", positive=True)
    assert valeur_finale(1 / (P + a)) == 0


def test_le_contre_exemple_de_la_remarque_3_2() -> None:
    """`F = 1/((p + 2)(p - 3))` : `lim p F = 0`, mais `f` diverge."""
    image = 1 / ((P + 2) * (P - 3))
    assert limite_en_zero(image) == 0
    assert poles_genants(image) == (3,)
    original = par_fractions_simples(image)
    assert sp.simplify(original - (exp(3 * T) - exp(-2 * T)) / 5) == 0
    assert sp.limit(original, T, sp.oo) == sp.oo
    with pytest.raises(ValueError, match="theoreme inapplicable"):
        valeur_finale(image)


@pytest.mark.parametrize("original", [exp(-T), sp.sin(2 * T), T**2 * exp(-3 * T)])
def test_multiplier_par_t_derive_l_image(original: sp.Expr) -> None:
    """Propriete 3.11.1 : `L[t y] = -Y'(p)` et `L[t^n y] = (-1)^n Y^(n)(p)`."""
    image = Transformee.de(original).image
    for n in (1, 2):
        directe = Transformee.de(T**n * original).image
        assert sp.simplify(du_produit_par_t(image, n) - directe) == 0


@pytest.mark.parametrize("original", [exp(-T), sp.cos(2 * T), 1 + T**2])
def test_les_images_de_t_y_prime_et_de_t_y_seconde(original: sp.Expr) -> None:
    """Propriete 3.11.2 : `-Y - p Y'` et `-2pY - p^2 Y' + y(0)`."""
    image = Transformee.de(original).image
    prime = Transformee.de(T * sp.diff(original, T)).image
    seconde = Transformee.de(T * sp.diff(original, T, 2)).image
    assert sp.simplify(de_t_fois_la_derivee(image) - prime) == 0
    initiale = original.subs(T, 0)
    assert sp.simplify(de_t_fois_la_derivee_seconde(image, initiale) - seconde) == 0


def test_les_quatre_premiers_polynomes_de_laguerre() -> None:
    """(3.6) : `1`, `1 - x`, `1 - 2x + x^2/2`, `1 - 3x + 3x^2/2 - x^3/6`."""
    attendus = (
        sp.Integer(1),
        1 - T,
        1 - 2 * T + T**2 / 2,
        1 - 3 * T + R(3, 2) * T**2 - T**3 / 6,
    )
    for n, attendu in enumerate(attendus):
        assert sp.expand(polynome_de_laguerre(n) - attendu) == 0


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
def test_l_image_d_un_polynome_de_laguerre(n: int) -> None:
    """`Y = (p - 1)^n / p^(n+1)`, la solution de l'equation separable en `Y(p)`."""
    image = Transformee.de(polynome_de_laguerre(n)).image
    assert sp.simplify(image - image_de_laguerre(n)) == 0
    quotient = sp.diff(image_de_laguerre(n), P) / image_de_laguerre(n)
    assert sp.simplify(quotient - (n + 1 - P) / ((P - 1) * P)) == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_la_recurrence_des_polynomes_de_laguerre(n: int) -> None:
    """`(n + 1) L_(n+1) = (2n + 1 - x) L_n - n L_(n-1)`."""
    gauche = (n + 1) * polynome_de_laguerre(n + 1)
    droite = (2 * n + 1 - T) * polynome_de_laguerre(n) - n * polynome_de_laguerre(n - 1)
    assert sp.expand(gauche - droite) == 0


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_les_polynomes_de_laguerre_verifient_l_equation_de_laguerre(n: int) -> None:
    """`t y'' + (1 - t) y' + n y = 0`.

    Le syllabus imprime `(1 - t) y(t)` page 74 ; sa propre transformee, deux
    lignes plus bas, est celle de `(1 - t) y'(t)`.
    """
    y = polynome_de_laguerre(n)
    juste = T * sp.diff(y, T, 2) + (1 - T) * sp.diff(y, T) + n * y
    assert sp.expand(juste) == 0
    if n:
        imprimee = T * sp.diff(y, T, 2) + (1 - T) * y + n * y
        assert sp.expand(imprimee) != 0


def test_un_degre_negatif_n_est_pas_un_polynome_de_laguerre() -> None:
    with pytest.raises(ValueError, match="degre negatif"):
        polynome_de_laguerre(-1)
