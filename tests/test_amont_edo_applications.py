"""Mathematiques II, sections 9 et 10 : les applications des planches.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les modeles de `polytech.maths` portent ceux des planches, en
recopiant leurs resultats.

Sources : `CalculDifferentielIntegral2/SLIDES/
Slides_09_Equations_differentielles_ordinaires.pdf`, pages 41 a 49, et
`Slides_10_Equations_differentielles_ordinaires_d_ordre_n.pdf`, pages 36 a 39.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.modeles import malthus, prix_de_marche
from polytech.maths.domain.differentielles.oscillateurs import masse_amortisseur_ressort
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE
from polytech.maths.domain.symboles import CONSTANTE


def test_les_mulots_de_la_page_43() -> None:
    """Page 43 : `y' = y/2 - 450`, `y = 900 + lambda e^(t/2)`."""
    generale = malthus(sp.Rational(1, 2), 450).solution_generale()
    assert sp.simplify(generale - 900 - CONSTANTE * sp.exp(VARIABLE / 2)) == 0


def test_le_prix_de_marche_de_la_page_49() -> None:
    """Pages 48 et 49 : le prix tend vers `(alpha + gamma) / (beta + delta)`."""
    alpha, beta, gamma, delta, k = sp.symbols("alpha beta gamma delta k", positive=True)
    modele = prix_de_marche(alpha, beta, gamma, delta, k)
    equilibre = (alpha + gamma) / (beta + delta)
    assert modele.prix_d_equilibre() == equilibre
    attendue = equilibre + CONSTANTE * sp.exp(-k * (beta + delta) * VARIABLE)
    assert sp.simplify(modele.equation().solution_generale() - attendue) == 0


def test_la_pulsation_du_systeme_masse_ressort_de_la_page_37() -> None:
    """Section 10, page 37 : `m r2 + k = 0` a pour racines `+- j racine(k / m)`."""
    masse, raideur = sp.symbols("m k", positive=True)
    racines = masse_amortisseur_ressort(masse, 0, raideur).racines()
    omega = sp.sqrt(raideur / masse)
    assert {sp.simplify(r) for r in racines} == {sp.I * omega, -sp.I * omega}


def test_les_racines_du_systeme_amorti_n_ont_pas_le_j_de_la_page_39() -> None:
    """Section 10, page 39 : la planche ecrit `(-c +- j racine(c2 - 4km)) / 2m`.

    Ecart au cours, lu sur la page rendue : le `j` est de trop tant que le
    radical porte `c2 - 4km`. Pour `m = 1`, `c = 3`, `k = 2`, le discriminant
    vaut 1 et les racines sont reelles, `-1` et `-2` ; la formule de la
    planche donnerait `-3/2 +- j/2`, qui n'annule pas `r2 + 3r + 2`.
    """
    operateur = masse_amortisseur_ressort(1, 3, 2)
    assert operateur.racines() == {-1: 1, -2: 1}
    assert operateur.discriminant() == 1
    de_la_planche = sp.Rational(-3, 2) + sp.I / 2
    assert operateur.en(de_la_planche) != 0
