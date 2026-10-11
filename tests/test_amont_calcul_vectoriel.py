"""Mecanique rationnelle I, atelier de calcul vectoriel, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles de `polytech.maths` auxquels
`cursus/notions/MecaniqueRationnelle1.toml` rattache un titre portent ce que
la planche enonce.

Source : `MecaniqueRationnelle1/SLIDES/Slides_Atelier_Calcul_vectoriel.pdf`.
Les pages sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.algebre import metrique
from polytech.maths.domain.algebre.espaces import combinaison_lineaire, sont_colineaires

CANONIQUE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_la_base_cartesienne_est_orthonormee() -> None:
    """Page 18 (16) : orthogonaux deux a deux, et de norme 1."""
    assert metrique.est_orthonormale(CANONIQUE)
    assert not metrique.est_orthonormale(((2.0, 0.0, 0.0), *CANONIQUE[1:]))


def test_les_regles_de_calcul_par_composantes() -> None:
    """Page 20 (18) : `a + b` et `k a`, composante par composante."""
    a, b = sp.Matrix([1, 2, 3]), sp.Matrix([4, -5, 6])
    assert combinaison_lineaire((1, 1), (a, b)) == sp.Matrix([5, -3, 9])
    assert combinaison_lineaire((-2,), (a,)) == sp.Matrix([-2, -4, -6])


def test_deux_vecteurs_paralleles_ont_la_meme_direction() -> None:
    """Page 9 (9) : `a // c`, quel que soit le sens."""
    a = sp.Matrix([1, 2, 3])
    assert sont_colineaires(a, -2 * a)
    assert not sont_colineaires(a, sp.Matrix([1, 2, 4]))


def test_la_projection_et_les_composantes_tangentielle_et_normale() -> None:
    """Pages 25 et 26 (23 et 24) : `a_t = a . u`, `a = a_t u + a_n`."""
    a, u = (3.0, 4.0, 0.0), (1.0, 0.0, 0.0)
    assert metrique.produit_scalaire(a, u) == pytest.approx(3)
    tangentielle = metrique.projection_orthogonale(a, (u,))
    assert tangentielle == pytest.approx((3, 0, 0))
    normale = tuple(x - y for x, y in zip(a, tangentielle, strict=True))
    assert metrique.produit_scalaire(normale, u) == pytest.approx(0)


def test_le_travail_de_l_exercice_4_vaut_24_joules() -> None:
    """Page 30 (27) : 10 N a 37 degres sur 3 m, `W = F . d = 24 J`."""
    angle = math.radians(37)
    force = (10 * math.cos(angle), 10 * math.sin(angle), 0.0)
    travail = metrique.produit_scalaire(force, (3.0, 0.0, 0.0))
    assert travail == pytest.approx(24, abs=0.05)
    assert travail == pytest.approx(metrique.norme(force) * 3 * math.cos(angle))
