"""Les articles d'equations differentielles de la fiche de « Mathematique ».

Linearite du premier ordre, coefficients constants, transformee de
Laplace : chaque formule, piege, cas chiffre et point de figure de
`cursus/fiches/Mathematique.toml` est recalcule par
`polytech.maths.domain.{differentielles,laplace}`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.ordre_n import solution_de_cauchy
from polytech.maths.domain.differentielles.particulieres import pour_un_polynome
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as X
from polytech.maths.domain.differentielles.prolongement import intervalles_de_definition
from polytech.maths.domain.differentielles.second_ordre import EquationDuSecondOrdre
from polytech.maths.domain.laplace.heaviside import fractions_simples
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import de_la_derivee_n
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import CONSTANTE as C
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T


def test_la_formule_du_premier_ordre_est_celle_de_la_bibliotheque() -> None:
    p, q = sp.cos(X), sp.exp(-sp.sin(X)) * X
    noyau = sp.exp(-sp.integrate(p, X))
    formule = noyau * (C + sp.integrate(q * sp.exp(sp.integrate(p, X)), X))
    assert egales(EquationLineaire(p, q).solution_generale(), formule)


def test_janvier_2021_3_les_solutions_de_y_prime_moins_2ty_egal_t() -> None:
    generale = EquationLineaire(-2 * X, X).solution_generale()
    assert egales(generale, C * sp.exp(X**2) - sp.Rational(1, 2))
    for constante in (1, 0, -1):
        y = generale.subs(C, constante)
        assert egales(sp.diff(y, X) - 2 * X * y, X)


def test_janvier_2021_3_tan_et_cot_ne_sont_definies_que_par_intervalles() -> None:
    demi = sp.pi / 2
    etude = sp.Interval(-2, 4)
    tangente = intervalles_de_definition(sp.tan(X), etude)
    assert sp.Interval.open(-demi, demi) in tangente
    cotangente = intervalles_de_definition(sp.cot(X), etude)
    assert sp.Interval.open(0, sp.pi) in cotangente
    assert all(not i.contains(0) for i in cotangente)


def test_les_racines_caracteristiques_des_deux_examens_et_de_la_figure() -> None:
    assert set(EquationDuSecondOrdre(1, -4, 3).racines_caracteristiques()) == {1, 3}
    assert set(EquationDuSecondOrdre(1, -5, -14).racines_caracteristiques()) == {7, -2}
    homogene = EquationDuSecondOrdre(1, -5, -14).solution_homogene()
    assert egales(
        homogene, sp.Symbol("c_1") * sp.exp(7 * X) + sp.Symbol("c_2") * sp.exp(-2 * X)
    )


def test_janvier_2022_3_la_particuliere_et_la_solution_de_cauchy() -> None:
    operateur = OperateurDifferentiel(D**2 - 4 * D + 3)
    assert egales(
        pour_un_polynome(operateur, T**2), T**2 / 3 + 8 * T / 9 + sp.Rational(26, 27)
    )
    attendue = (
        T**2 / 3 + 8 * T / 9 + sp.Rational(26, 27) - sp.exp(T) + sp.exp(3 * T) / 27
    )
    assert egales(solution_de_cauchy(operateur, T**2, (0, 0)), attendue)


def test_janvier_2025_3_la_particuliere_et_la_solution_de_cauchy() -> None:
    operateur = OperateurDifferentiel(D**2 - 5 * D - 14)
    second = -14 * T**2 - 10 * T - 26
    assert egales(pour_un_polynome(operateur, second), T**2 + 2)
    attendue = T**2 + 2 + sp.exp(7 * T) - 3 * sp.exp(-2 * T)
    assert egales(solution_de_cauchy(operateur, second, (0, 13)), attendue)


def test_le_degre_monte_quand_zero_est_racine_caracteristique() -> None:
    assert (
        sp.degree(pour_un_polynome(OperateurDifferentiel(D**2 - 4 * D + 3), T**2), T)
        == 2
    )
    assert (
        sp.degree(pour_un_polynome(OperateurDifferentiel(D**2 - 4 * D), T**2), T) == 3
    )


def test_la_resonance_donne_un_facteur_t() -> None:
    equation = EquationDuSecondOrdre(1, 0, 1)
    assert equation.est_en_resonance(1)
    particuliere = equation.solution_particuliere(sp.cos(X))
    assert egales(particuliere, X * sp.sin(X) / 2)


def test_laplace_de_la_derivee_premiere_et_seconde() -> None:
    f = sp.exp(2 * T) * sp.cos(3 * T) + T**2
    image = Transformee.de(f).image
    assert egales(
        Transformee.de(sp.diff(f, T)).image, de_la_derivee_n(image, [f.subs(T, 0)])
    )
    initiales = [f.subs(T, 0), sp.diff(f, T).subs(T, 0)]
    assert egales(
        Transformee.de(sp.diff(f, T, 2)).image, de_la_derivee_n(image, initiales)
    )


def test_janvier_2025_3_l_image_garde_le_treize_des_conditions_initiales() -> None:
    probleme = ProblemeLineaire((1, -5, -14), -14 * T**2 - 10 * T - 26, (0, 13))
    second = 13 - 28 / P**3 - 10 / P**2 - 26 / P
    assert egales(probleme.image() * (P**2 - 5 * P - 14), second)
    termes = fractions_simples(probleme.image())
    assert egales(sum(termes), 2 / P**3 + 2 / P + 1 / (P - 7) - 3 / (P + 2))


def test_janvier_2022_3_conditions_nulles_et_pole_triple() -> None:
    probleme = ProblemeLineaire((1, -4, 3), T**2, (0, 0))
    assert egales(probleme.image() * (P**2 - 4 * P + 3), 2 / P**3)
    assert probleme.solution_verifiee()
    denominateur = sp.Poly(sp.denom(sp.together(probleme.image())), P)
    assert sp.roots(denominateur)[sp.Integer(0)] == 3
