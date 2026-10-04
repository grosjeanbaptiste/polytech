"""Chapitre II.1 d'optimisation : la forme equivalente et le tableau simplexe.

Oracles : les trois tableaux de l'illustration II.4.4 a) du cours
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`, page 57),
recopies case par case, et les definitions de la page 48 --
`a_i0 = B^-1 d`, `A = B^-1 R`, `a_0 = c_B B^-1 R - c_R`,
`a_00 = c_B B^-1 d` -- refaites ici a la main sur la base `{1, 2, 4}`.
Les indices sont ceux du cours, a partir de 1.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.tableau import (
    TableauSimplexe,
    nombre_maximal_de_bases,
    tableau_de_la_base,
)

R = sp.Rational


def _illustration() -> FormeStandard:
    """`max 6 x1 + 5 x2` sous forme standard : couts de `z`, non de `-z`."""
    return FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


def _tableau(*base: int) -> TableauSimplexe:
    return tableau_de_la_base(_illustration(), base)


def test_le_nombre_maximal_de_bases_est_un_coefficient_binomial() -> None:
    """`C(n, m) = n! / ((n - m)! m!)` : dix pour `n = 5`, `m = 3`."""
    assert nombre_maximal_de_bases(5, 3) == math.comb(5, 3) == 10


def test_le_premier_tableau_est_celui_du_cours() -> None:
    """Base des ecarts `{3, 4, 5}` : le probleme est deja sous forme equivalente."""
    tableau = _tableau(3, 4, 5)
    assert tableau.hors_base == (1, 2)
    assert tableau.coefficients == sp.Matrix(
        [[0, -6, -5], [8, 1, 1], [6, -2, 3], [2, 1, -1]]
    )


def test_le_deuxieme_tableau_est_celui_du_cours() -> None:
    """`x1` est entre, `x5` est sorti : base `{3, 4, 1}`, hors base `{5, 2}`."""
    tableau = tableau_de_la_base(_illustration(), (3, 4, 1), hors_base=(5, 2))
    assert tableau.coefficients == sp.Matrix(
        [[12, 6, -11], [6, -1, 2], [10, 2, 1], [2, 1, -1]]
    )


def test_le_tableau_optimal_est_celui_du_cours() -> None:
    tableau = tableau_de_la_base(_illustration(), (2, 4, 1), hors_base=(5, 3))
    assert tableau.coefficients == sp.Matrix(
        [
            [45, R(1, 2), R(11, 2)],
            [3, -R(1, 2), R(1, 2)],
            [7, R(5, 2), -R(1, 2)],
            [5, R(1, 2), R(1, 2)],
        ]
    )


def test_la_premiere_colonne_donne_la_valeur_et_la_solution() -> None:
    """`a_00` est la valeur de `z`, `a_i0` celle de la variable de base `x_i`."""
    tableau = tableau_de_la_base(_illustration(), (2, 4, 1), hors_base=(5, 3))
    assert tableau.valeur == 45
    assert list(tableau.solution(5)) == [5, 3, 0, 7, 0]


def test_le_tableau_suit_les_definitions_de_la_page_48() -> None:
    """Sur la base `{1, 2, 4}`, tout se recalcule a partir de `B^-1`."""
    forme = _illustration()
    b = forme.matrice[:, [0, 1, 3]]
    r = forme.matrice[:, [2, 4]]
    c_b, c_r = forme.couts[:, [0, 1, 3]], forme.couts[:, [2, 4]]
    tableau = _tableau(1, 2, 4)
    assert tableau.coefficients[1:, 0] == b.inv() * forme.seconds_membres
    assert tableau.coefficients[1:, 1:] == b.inv() * r
    assert tableau.coefficients[0, 1:] == c_b * b.inv() * r - c_r
    assert tableau.coefficients[0, 0] == (c_b * b.inv() * forme.seconds_membres)[0, 0]


def test_une_colonne_du_tableau_exprime_un_vecteur_hors_base() -> None:
    """Remarque 1, page 49 : `t_j = somme(a_ij t_i)` sur les vecteurs de base."""
    forme = _illustration()
    tableau = _tableau(1, 2, 4)
    for rang, j in enumerate(tableau.hors_base, start=1):
        combinaison = sp.zeros(3, 1)
        for ligne, i in enumerate(tableau.base, start=1):
            combinaison += tableau.coefficients[ligne, rang] * forme.matrice[:, i - 1]
        assert combinaison == forme.matrice[:, j - 1]


def test_le_tableau_complet_porte_l_identite_sur_la_base() -> None:
    """Remarque 2, page 49 : `m + 1` lignes, `n + 1` colonnes."""
    complet = _tableau(1, 2, 4).complet(5)
    assert complet.shape == (4, 6)
    assert complet[1:, [1, 2, 4]] == sp.eye(3)
    assert list(complet[0, [1, 2, 4]]) == [0, 0, 0]


def test_le_tableau_complet_contient_l_inverse_de_la_base() -> None:
    """Remarque 3 : sous les colonnes de la matrice unite de `T`, on lit `B^-1`."""
    forme = _illustration()
    inverse = forme.matrice[:, [0, 1, 3]].inv()
    assert _tableau(1, 2, 4).complet(5)[1:, [3, 4, 5]] == inverse


def test_une_base_non_reguliere_est_refusee() -> None:
    singuliere = FormeStandard(
        couts=sp.Matrix([[2, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 2, 0, 0, 1]]),
        seconds_membres=sp.Matrix([4, 3, 8]),
    )
    with pytest.raises(ValueError, match="n'est pas une base"):
        tableau_de_la_base(singuliere, (1, 3, 5))


def test_une_base_de_mauvaise_taille_est_refusee() -> None:
    with pytest.raises(ValueError, match="indices de base pour"):
        tableau_de_la_base(_illustration(), (1, 2))
