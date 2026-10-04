"""Chapitre II.5 d'optimisation : comparer des couts ou figure la constante `M`.

`M` est « une constante symbolique aussi grande que l'on veut » (page
59) : `a + b M` se compare donc d'abord par `b`, puis par `a`. L'oracle
est le premier tableau de l'illustration II.5.4 a)
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`, page 63), que
le cours ecrit sur deux lignes : les termes en `M`, puis les autres.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import est_optimal, variable_entrante
from polytech.optimisation.domain.grand_m import GRAND_M, deux_lignes, ordre
from polytech.optimisation.domain.tableau import TableauSimplexe

M = GRAND_M
MAX = Optimisation.MAX


def test_un_nombre_n_a_pas_de_terme_en_m() -> None:
    assert ordre(sp.Integer(5)) == (0, 5)


def test_un_terme_en_m_l_emporte_sur_toute_constante() -> None:
    """`M - 1000 > 999` des que `M` est assez grand."""
    assert ordre(M - 1000) > ordre(sp.Integer(999))
    assert ordre(-M + 1000) < ordre(sp.Integer(-999))


def test_a_coefficient_de_m_egal_la_constante_departage() -> None:
    assert ordre(M + 6) > ordre(M + 5)


def test_une_expression_non_lineaire_en_m_est_refusee() -> None:
    with pytest.raises(ValueError, match="pas lineaire en M"):
        ordre(M**2)


def _premier_tableau() -> TableauSimplexe:
    """Page 63 : base `(v, x4, x5)`, ou `v` porte l'indice 6 ; hors base `(1, 2, 3)`."""
    coefficients = sp.Matrix(
        [[-8 * M, -M - 6, -M - 5, M], [8, 1, 1, -1], [6, -2, 3, 0], [2, 1, -1, 0]]
    )
    return TableauSimplexe((6, 4, 5), (1, 2, 3), coefficients)


def test_la_ligne_de_z_se_lit_sur_deux_lignes_comme_dans_le_cours() -> None:
    """Page 63 : `-8M | -M  -M  M` au-dessus de `0 | -6  -5  0`."""
    en_m, sans_m = deux_lignes(_premier_tableau())
    assert en_m == [-8, -1, -1, 1]
    assert sans_m == [0, -6, -5, 0]


def test_x1_entre_dans_le_premier_tableau_de_la_page_63() -> None:
    """En maximisation, `-M - 6` est le plus negatif : `x1` rentre dans la base."""
    assert not est_optimal(_premier_tableau(), MAX)
    assert variable_entrante(_premier_tableau(), MAX) == 1
