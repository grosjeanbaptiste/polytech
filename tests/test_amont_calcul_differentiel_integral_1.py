"""Mathematiques I : le syllabus de calcul differentiel et integral.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels
`cursus/notions/CalculDifferentielIntegral1.toml` rattache un titre portent
ce que le syllabus enonce -- en recopiant ses enonces et ses exemples.

Source : `CalculDifferentielIntegral1/SYLLABUS/`
`Syllabus_Calcul_differentiel_et_integral_1.pdf`. Les pages sont celles du
PDF ; la page imprimee est entre parentheses.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.fourier.parite import est_impaire, est_paire
from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.impropres import IntegraleImpropre
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.integration.proprietes import integrale_orientee
from polytech.maths.domain.integration.riemann import limite_des_sommes
from polytech.maths.domain.integration.singulieres import IntegraleSinguliere
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.series.binomiale import developpement_binomial
from polytech.maths.domain.series.mac_laurin import (
    coefficients_de_mac_laurin,
    reste_de_lagrange,
)
from polytech.maths.domain.symboles import VARIABLE

X = VARIABLE
DEMI = sp.Rational(1, 2)


def test_la_parite_de_la_definition_3_18() -> None:
    """Page 64 (62) : `f(x) = f(-x)` ou `f(x) = -f(-x)`, ou ni l'un ni l'autre."""
    assert est_paire(X**2)
    assert est_paire(sp.cos(X))
    assert est_impaire(X**3)
    assert est_impaire(sp.sin(X))
    assert not est_paire(X + 1)
    assert not est_impaire(X + 1)


def test_le_developpement_de_la_racine_de_la_figure_5_7() -> None:
    """Page 147 (145), exemples 5.24 : `1 + x/2 - x2/8 + x3/16`, et le reste."""
    attendu = 1 + X / 2 - X**2 / 8 + X**3 / 16
    assert developpement_binomial(DEMI, 3) == attendu
    assert coefficients_de_mac_laurin(sp.sqrt(1 + X), 4) == (
        1,
        DEMI,
        -DEMI / 4,
        DEMI / 8,
    )
    theta = sp.Symbol("theta")
    assert reste_de_lagrange(sp.exp(X), 2, theta) == sp.exp(theta * X) * X**3 / 6


def test_une_primitive_se_verifie_en_la_derivant() -> None:
    """Pages 166 et 167 (164 et 165) : `F' = f`, a une constante pres."""
    primitive = Primitive.de(X**2)
    assert primitive.verifiee
    assert sp.diff(primitive.avec_constante(), X) == X**2


def test_la_primitivation_par_parties_des_exemples_6_12() -> None:
    """Page 171 (169) : `int e^(ax) P = e^(ax)/a (P - P'/a + P''/a2)`, ici `a = 2`."""
    polynome = X**2
    geste = IntegrationParParties(u=polynome, derivee_de_v=sp.exp(2 * X))
    attendue = sp.exp(2 * X) / 2 * (polynome - 2 * X / 2 + 2 / sp.Integer(4))
    assert sp.simplify(geste.resultat() - attendue) == 0


def test_le_changement_de_variable_de_la_proposition_6_13() -> None:
    """Page 172 (170) : `int f(phi(t)) phi'(t) dt = int f(xi) d xi`, `xi = phi(t)`."""
    geste = Substitution(integrande=2 * X * sp.cos(X**2), expression=X**2)
    assert geste.differentielle() == 2 * X
    assert sp.simplify(geste.resultat() - sp.sin(X**2)) == 0


def test_la_decomposition_en_fractions_simples_de_la_section_6_3() -> None:
    """Pages 173 a 177 (171 a 175) : division, racines simples, facteur quadratique."""
    impropre = FractionRationnelle(X**3, (X - 1) * (X - 2))
    assert not impropre.est_propre()
    assert impropre.partie_entiere() == X + 3
    simple = FractionRationnelle(sp.Integer(1), (X - 1) * (X - 2))
    assert set(simple.termes()) == {-1 / (X - 1), 1 / (X - 2)}
    quadratique = FractionRationnelle(sp.Integer(1), (X - 1) ** 2 + 4)
    assert sp.simplify(quadratique.primitive() - sp.atan((X - 1) / 2) / 2) == 0


def test_l_integrale_est_la_limite_des_sommes_et_la_difference_des_primitives() -> None:
    """Pages 192 et 202 (190 et 200) : Riemann, puis `F(b) - F(a)`."""
    assert limite_des_sommes(X**2, (0, 1)) == sp.Rational(1, 3)
    integrale = IntegraleDefinie(X**2, sp.Integer(0), sp.Integer(1))
    haut, bas = integrale.evaluations()
    assert integrale.valeur() == haut - bas == sp.Rational(1, 3)


def test_la_relation_de_chasles_de_la_proposition_7_20() -> None:
    """Page 200 (198) : `int_a^b = int_a^c + int_c^b`, et le signe selon l'ordre."""
    entiere = integrale_orientee(X**2, 0, 2)
    assert entiere == integrale_orientee(X**2, 0, 1) + integrale_orientee(X**2, 1, 2)
    assert integrale_orientee(X**2, 2, 0) == -entiere


@pytest.mark.parametrize(("alpha", "converge"), [(DEMI, True), (1, False), (2, False)])
def test_l_exemple_7_34_converge_si_alpha_est_inferieur_a_un(
    alpha: sp.Expr, *, converge: bool
) -> None:
    """Page 207 (205) : `int_0^1 dt / t^alpha` converge ssi `alpha < 1`."""
    integrale = IntegraleSinguliere(X**-alpha, sp.Integer(0), sp.Integer(1))
    assert integrale.converge() is converge


@pytest.mark.parametrize(("alpha", "converge"), [(DEMI, False), (1, False), (2, True)])
def test_a_l_infini_c_est_l_inverse(alpha: sp.Expr, *, converge: bool) -> None:
    """Page 208 (206) : `int_a^infini dt / t^alpha` converge si `alpha > 1`."""
    assert IntegraleImpropre(X**-alpha, sp.Integer(1)).converge() is converge
