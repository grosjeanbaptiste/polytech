"""Mathematiques I, chapitre 2 d'algebre : l'anneau des polynomes.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire1.toml`
rattache un titre portent ce que la planche enonce -- en recopiant ses
exemples, page citee.

Source : `AlgebreLineaire1/SLIDES/Slides_02_Structures_algebriques.pdf`
(« ch. 2 »). Les pages sont celles du PDF.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import polynomes
from polytech.maths.domain.algebre.polynomes_exacts import division, ordre_de_la_racine

X = sp.Symbol("x")


def test_la_division_de_la_page_86() -> None:
    """Ch. 2, page 86 : `x4 + 3x3 - 3 = (x2 + 2x - 2)(x2 + x) + (2x - 3)`."""
    assert division(X**4 + 3 * X**3 - 3, X**2 + 2 * X - 2, X) == (X**2 + X, 2 * X - 3)
    quotient, reste = polynomes.division_euclidienne((1, 3, 0, 0, -3), (1, 2, -2))
    assert quotient == pytest.approx((1, 1, 0))
    assert reste[-2:] == pytest.approx((2, -3))
    assert polynomes.degre((1, 3, 0, 0, -3)) == 2 + polynomes.degre(tuple(quotient))


def test_les_racines_et_leurs_ordres_de_la_page_91() -> None:
    """Ch. 2, page 91 : `3x4 - 6x3 + 6x2 - 6x + 3`, 1 double, `i` et `-i` simples."""
    polynome = 3 * X**4 - 6 * X**3 + 6 * X**2 - 6 * X + 3
    assert ordre_de_la_racine(polynome, sp.Integer(1), X) == 2
    assert ordre_de_la_racine(polynome, sp.I, X) == 1
    assert ordre_de_la_racine(polynome, -sp.I, X) == 1
    assert polynomes.est_racine_multiple((3, -6, 6, -6, 3), 1.0)


def test_les_relations_entre_coefficients_et_racines_de_la_page_96() -> None:
    """Ch. 2, page 96 : `2x2 - 10x + 12`, racines 2 et 3, somme 5, produit 6."""
    symetriques = polynomes.fonctions_symetriques_des_racines((2, -10, 12))
    assert symetriques == pytest.approx((1, 5, 6))
