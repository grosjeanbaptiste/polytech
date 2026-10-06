"""Mathematiques II, section 10 : les exemples des planches, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. L'algebre
des operateurs `K(D)` de `polytech.maths` a ete portee pour le cours du
soir ; ces tests verifient qu'elle porte aussi ce que la section 10 enonce.

Source : `CalculDifferentielIntegral2/SLIDES/
Slides_10_Equations_differentielles_ordinaires_d_ordre_n.pdf`, 39 pages.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.particulieres import (
    pour_un_polynome,
    pour_un_polynome_module,
)
from polytech.maths.domain.symboles import TEMPS

T = TEMPS
OSCILLATEUR = OperateurDifferentiel(D**2 + 1)


def test_composer_deux_operateurs_c_est_multiplier_leurs_polynomes() -> None:
    """Page 6 : `(D + 1)(D2 + D + 1) y = (D3 + 2 D2 + 2 D + 1) y`."""
    produit = OperateurDifferentiel(D + 1).composer(OperateurDifferentiel(D**2 + D + 1))
    assert produit == OperateurDifferentiel(D**3 + 2 * D**2 + 2 * D + 1)
    y = sp.Function("y")(T)
    attendu = y.diff(T, 3) + 2 * y.diff(T, 2) + 2 * y.diff(T) + y
    assert sp.simplify(produit.appliquer(y) - attendu) == 0


@pytest.mark.parametrize(
    ("polynome", "racines", "base"),
    [
        (D**2 - 3 * D + 2, {1: 1, 2: 1}, {sp.exp(T), sp.exp(2 * T)}),
        (
            D**3 - 3 * D + 2,
            {1: 2, -2: 1},
            {sp.exp(T), T * sp.exp(T), sp.exp(-2 * T)},
        ),
        (D**2 + 1, {sp.I: 1, -sp.I: 1}, {sp.cos(T), sp.sin(T)}),
    ],
    ids=["page 20, exemple 1", "page 20, exemple 2", "page 21, exemple 3"],
)
def test_les_exemples_de_sgeh_des_pages_20_et_21(
    polynome: sp.Expr, racines: dict[object, int], base: set[sp.Expr]
) -> None:
    """Pages 20 et 21 : racines, multiplicites, et la base qu'elles donnent."""
    operateur = OperateurDifferentiel(polynome)
    assert operateur.racines() == racines
    assert set(operateur.base_des_solutions()) == base


def test_les_racines_complexes_de_la_page_21() -> None:
    """Page 21, exemple 1 : `e^(-t/2) cos(racine(3) t / 2)` et son sinus."""
    demi, pulsation = sp.Rational(1, 2), sp.sqrt(3) / 2
    enveloppe = sp.exp(-demi * T)
    attendue = {enveloppe * sp.cos(pulsation * T), enveloppe * sp.sin(pulsation * T)}
    assert set(OperateurDifferentiel(D**2 + D + 1).base_des_solutions()) == attendue


def test_la_division_de_la_page_26() -> None:
    """Page 26 : `1 / (1 + D2) = 1 - D2 + D4`, applique a `1 + t + t4`."""
    assert OSCILLATEUR.inverse_tronque(4) == 1 - D**2 + D**4
    second_membre = 1 + T + T**4
    particuliere = pour_un_polynome(OSCILLATEUR, second_membre)
    attendue = OperateurDifferentiel(1 - D**2 + D**4).appliquer(second_membre)
    assert sp.expand(particuliere - attendue) == 0
    assert sp.expand(OSCILLATEUR.appliquer(particuliere) - second_membre) == 0


def test_la_solution_particuliere_des_pages_28_a_30() -> None:
    """Page 30 : pour `(D4 - D2) y = 1 + t + t2`, `y = -3 t2/2 - t3/6 - t4/12`."""
    particuliere = pour_un_polynome(OperateurDifferentiel(D**4 - D**2), 1 + T + T**2)
    attendue = -sp.Rational(3, 2) * T**2 - T**3 / 6 - T**4 / 12
    assert sp.expand(particuliere - attendue) == 0


def test_la_solution_particuliere_de_la_page_32() -> None:
    """Page 32 : pour `y'' + y = t e^(-t)`, `y = e^(-t) (t + 1) / 2`."""
    particuliere = pour_un_polynome_module(OSCILLATEUR, T, -1)
    assert sp.simplify(particuliere - sp.exp(-T) * (T + 1) / 2) == 0


def test_la_constante_des_pages_33_et_34() -> None:
    """Pages 33 et 34 : `K(D) y = alpha` donne `alpha / K(0)` ; ici `3 / 1`."""
    assert pour_un_polynome(OSCILLATEUR, sp.Integer(3)) == 3 / OSCILLATEUR.en(0)


def test_la_solution_generale_de_la_page_10_est_la_somme_des_deux() -> None:
    """Page 10 : pour `y'' + y = t`, `c1 cos t + c2 sin t + t`."""
    particuliere = pour_un_polynome(OSCILLATEUR, T)
    assert particuliere == T
    c1, c2 = sp.symbols("c1 c2", real=True)
    homogene = sum(
        c * y for c, y in zip((c1, c2), OSCILLATEUR.base_des_solutions(), strict=True)
    )
    assert sp.simplify(OSCILLATEUR.appliquer(homogene + particuliere) - T) == 0
