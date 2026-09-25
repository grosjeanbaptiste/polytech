"""Rendu LaTeX des champs abstraits, dans la notation du cours."""

import sympy as sp

from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.domain.cinematique.champ_generique import (
    CHAMP_GENERIQUE,
    U_GENERIQUE,
    V_GENERIQUE,
)
from polytech.fluides.domain.cinematique.symboles import X, Y


def test_un_champ_abstrait_s_ecrit_sans_sa_liste_d_arguments() -> None:
    assert expression_latex(U_GENERIQUE) == "u"


def test_une_derivee_partielle_s_ecrit_comme_dans_le_cours() -> None:
    assert expression_latex(sp.Derivative(U_GENERIQUE, X)) == (
        r"\frac{\partial u}{\partial x}"
    )


def test_la_divergence_generique_est_la_definition_lisible() -> None:
    assert expression_latex(CHAMP_GENERIQUE.divergence()) == (
        r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}"
    )


def test_le_rendu_d_un_champ_concret_reste_inchange() -> None:
    """L'imprimeur ne doit pas perturber les expressions ordinaires."""
    assert expression_latex(X**2) == "x^{2}"


def test_un_facteur_rationnel_devant_une_somme_reste_en_facteur() -> None:
    """Le cours ecrit 1/2 (dv/dx - du/dy), pas la forme distribuee."""
    expression = sp.factor_terms(
        (sp.Derivative(V_GENERIQUE, X) - sp.Derivative(U_GENERIQUE, Y)) / 2,
    )
    rendu = expression_latex(expression)
    assert rendu.startswith(r"\frac{1}{2}\left(")
    assert rendu.endswith(r"\right)")
