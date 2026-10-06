"""Mathematiques II, section 9 : les exemples des planches, refaits par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Les EDO
du premier ordre de `polytech.maths` ont ete portees pour le cours du soir ;
ces tests verifient qu'elles portent aussi ce que la section 9 enonce, en
recopiant ses exemples.

Source : `CalculDifferentielIntegral2/SLIDES/
Slides_09_Equations_differentielles_ordinaires.pdf`, 49 pages numerotees.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.bernoulli import EquationDeBernoulli
from polytech.maths.domain.differentielles.caracterisation import (
    Caracterisation,
    caracteriser,
    est_solution,
)
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.facteur_integrant import rendre_exacte
from polytech.maths.domain.differentielles.graphique import isocline, tangente
from polytech.maths.domain.differentielles.homogenes import (
    Z,
    degre_d_homogeneite,
    equation_en_z,
)
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
)
from polytech.maths.domain.differentielles.prolongement import (
    intervalles_de_definition,
)
from polytech.maths.domain.differentielles.separables import solutions_constantes
from polytech.maths.domain.symboles import CONSTANTE

Y = FONCTION(VARIABLE)
D1, D2, D3 = (Y.diff(VARIABLE, ordre) for ordre in (1, 2, 3))


@pytest.mark.parametrize(
    ("equation", "attendue"),
    [
        (
            D2 + 5 * D1 + 4 * Y - 3,
            Caracterisation(2, lineaire=True, coefficients_constants=True),
        ),
        (
            D2 + 5 * D1 + 4 * Y**2,
            Caracterisation(2, lineaire=False, coefficients_constants=False),
        ),
        (
            VARIABLE**2 * D3 + 5 * sp.cos(VARIABLE) * D1 + 4 * Y - sp.exp(VARIABLE),
            Caracterisation(3, lineaire=True, coefficients_constants=False),
        ),
    ],
)
def test_les_trois_exemples_de_caracterisation_de_la_page_9(
    equation: sp.Expr, attendue: Caracterisation
) -> None:
    """Page 9 : ordre, linearite, coefficients constants ou variables."""
    assert caracteriser(equation) == attendue


def test_la_solution_generale_de_la_page_11() -> None:
    """Page 11 : `y = a cos x + b sin x` resout `y'' + y = 0`."""
    a, b = sp.symbols("a b", real=True)
    assert est_solution(D2 + Y, a * sp.cos(VARIABLE) + b * sp.sin(VARIABLE))


def test_le_potentiel_de_l_equation_exacte_de_la_page_19() -> None:
    """Page 19 : `(x2 + y) dx + (x - 2y) dy = 0`, `g = x3/3 + xy - y2`."""
    forme = FormeDifferentielle(VARIABLE**2 + INCONNUE, VARIABLE - 2 * INCONNUE)
    assert forme.est_exacte()
    attendu = VARIABLE**3 / 3 + VARIABLE * INCONNUE - INCONNUE**2
    assert sp.simplify(forme.potentiel() - attendu) == 0


def test_la_solution_nulle_de_la_page_22_n_est_pas_perdue() -> None:
    """Page 22 : pour `y' = -2xy`, « y = 0 est solution », et `y = K e^(-x2)`."""
    assert solutions_constantes(INCONNUE) == (0,)
    famille = CONSTANTE * sp.exp(-(VARIABLE**2))
    assert est_solution(D1 + 2 * VARIABLE * Y, famille)


def test_la_tangente_et_l_isocline_des_pages_24_a_26() -> None:
    """Pages 24 a 26 : `y' = x - y2`, `D_M` et `Gamma_p : f(x, y) = p`."""
    pente = VARIABLE - INCONNUE**2
    assert tangente(pente, (2, 1)) == sp.Eq(INCONNUE - 1, VARIABLE - 2)
    demi = sp.Rational(3, 2)
    assert isocline(pente, demi) == sp.Eq(VARIABLE - INCONNUE**2, demi)


def test_l_equation_des_pages_23_et_32_devient_exacte() -> None:
    """Pages 23 et 32 : `dy/dx + 2y = 3`, par la formule (1) et par le facteur."""
    generale = EquationLineaire(sp.Integer(2), sp.Integer(3)).solution_generale()
    assert est_solution(D1 + 2 * Y - 3, generale)
    forme = FormeDifferentielle(2 * INCONNUE - 3, sp.Integer(1))
    assert not forme.est_exacte()
    assert rendre_exacte(forme).est_exacte()


def test_l_equation_homogene_de_la_page_34() -> None:
    """Pages 33 et 34 : `y' = 2xy / (x2 - y2)` est de degre 0, en `z = y/x`."""
    second_membre = 2 * VARIABLE * INCONNUE / (VARIABLE**2 - INCONNUE**2)
    assert degre_d_homogeneite(second_membre) == 0
    en_z, _ = equation_en_z(second_membre)
    assert sp.simplify(en_z - 2 * Z / (1 - Z**2)) == 0


def test_l_equation_en_z_de_bernoulli_est_la_relation_2_de_la_page_39() -> None:
    """Page 39, (2) : `z' + (1 - a) P z = (1 - a) Q`."""
    en_z = EquationDeBernoulli(
        VARIABLE, sp.sin(VARIABLE), sp.Integer(3)
    ).equation_en_z()
    assert en_z == EquationLineaire(-2 * VARIABLE, -2 * sp.sin(VARIABLE))


def test_la_singularite_de_l_equation_lineaire_de_la_page_38() -> None:
    """Page 38 : `y = y(0) / (1 - x)` n'existe pas en `x = 1`."""
    avant, apres = intervalles_de_definition(1 / (1 - VARIABLE))
    assert (avant.sup, apres.inf) == (1, 1)
