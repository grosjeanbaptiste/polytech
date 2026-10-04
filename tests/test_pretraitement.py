"""Chapitre XIV.3.5 c) d'optimisation : le pretraitement d'un programme entier.

Oracles (`Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 311-312), les quatre
« exemples elementaires » : `x1 + x2 + x3 <= 9` est redondante sous
`1 <= x_i <= 3` ; `2 x1 - x2 + 5 x3 <= 7` donne `x1 <= 2` et `x3 = 1`,
puis se simplifie en `2 x1 - x2 <= 2` ; `5 x1 - x2 + 6 x3 <= 7` est
impossible, son premier membre valant au moins 8.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Sens
from polytech.optimisation.domain.pretraitement import (
    est_impossible,
    est_redondante,
    etendue_du_premier_membre,
    fixer_les_variables,
    resserrer_les_bornes,
)

R = sp.Rational
INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
BORNES = ((1, 3), (1, 3), (1, 3))


def test_le_premier_membre_est_minimal_aux_bornes_opposees_aux_signes() -> None:
    """Page 312 : coefficient positif a la borne inferieure, negatif a la superieure."""
    contrainte = Contrainte((5, -1, 6), INF, 7)
    assert etendue_du_premier_membre(contrainte, BORNES) == (5 - 3 + 6, 15 - 1 + 18)


def test_une_contrainte_que_les_bornes_impliquent_est_redondante() -> None:
    """Page 311 : `x1 + x2 + x3 <= 9` avec `1 <= x_i <= 3`."""
    assert est_redondante(Contrainte((1, 1, 1), INF, 9), BORNES)
    assert not est_redondante(Contrainte((1, 1, 1), INF, 8), BORNES)


def test_une_contrainte_que_les_bornes_contredisent_est_impossible() -> None:
    """Page 312 : `5 x1 - x2 + 6 x3 <= 7`, premier membre au moins `5 - 3 + 6 = 8`."""
    assert est_impossible(Contrainte((5, -1, 6), INF, 7), BORNES)
    assert not est_impossible(Contrainte((5, -1, 6), INF, 8), BORNES)


def test_les_bornes_se_resserrent_variable_par_variable() -> None:
    """Page 311 : `x1 <= (7 + 3 - 5)/2`, donc 2 ; `x3 <= (7 - 2 + 3)/5`, donc 1."""
    contrainte = Contrainte((2, -1, 5), INF, 7)
    assert resserrer_les_bornes(contrainte, BORNES) == ((1, 2), (1, 3), (1, 1))


def test_un_coefficient_negatif_resserre_la_borne_inferieure() -> None:
    """`-2 x1 + x2 <= -3`, `0 <= x_i <= 3` : `x1 >= (3 + 0)/2`, donc 2 ; `x2 <= 3`."""
    contrainte = Contrainte((-2, 1), INF, -3)
    assert resserrer_les_bornes(contrainte, ((0, 3), (0, 3))) == ((2, 3), (0, 3))


def test_une_variable_fixee_disparait_de_la_contrainte() -> None:
    """Page 312 : avec `x3 = 1`, la contrainte se simplifie en `2 x1 - x2 <= 2`."""
    contrainte = Contrainte((2, -1, 5), INF, 7)
    simplifiee = fixer_les_variables(contrainte, ((1, 2), (1, 3), (1, 1)))
    assert simplifiee == Contrainte((2, -1, 0), INF, 2)


def test_une_contrainte_superieure_se_traite_par_son_opposee() -> None:
    """`x1 + x2 >= 2` avec `1 <= x_i` : redondante, comme `-x1 - x2 <= -2`."""
    assert est_redondante(Contrainte((1, 1), SUP, 2), ((1, 3), (1, 3)))
    assert est_impossible(Contrainte((1, 1), SUP, 7), ((1, 3), (1, 3)))
    assert resserrer_les_bornes(Contrainte((1, 1), SUP, 5), ((1, 3), (1, 3))) == (
        (2, 3),
        (2, 3),
    )


def test_une_egalite_n_est_pas_traitee_comme_une_inegalite() -> None:
    with pytest.raises(ValueError, match="une egalite ne se pretraite pas ainsi"):
        est_redondante(Contrainte((1, 1), EGA, 2), ((1, 3), (1, 3)))


def test_des_bornes_en_nombre_different_des_variables_sont_refusees() -> None:
    with pytest.raises(ValueError, match="bornes pour"):
        est_redondante(Contrainte((1, 1), INF, 2), ((1, 3),))


def test_le_resserrement_ne_perd_aucun_point_entier() -> None:
    """Le pretraitement « ne modifie pas l'ensemble des solutions admissibles »."""
    contrainte = Contrainte((2, -1, 5), INF, 7)
    resserrees = resserrer_les_bornes(contrainte, BORNES)

    def points(bornes: tuple[tuple[int, int], ...]) -> set[tuple[int, int, int]]:
        return {
            (a, b, c)
            for a in range(bornes[0][0], bornes[0][1] + 1)
            for b in range(bornes[1][0], bornes[1][1] + 1)
            for c in range(bornes[2][0], bornes[2][1] + 1)
            if contrainte.est_satisfaite((a, b, c))
        }

    assert points(resserrees) == points(BORNES)
    assert len(points(BORNES)) > 0


def test_une_contrainte_lache_ne_desserre_jamais_une_borne() -> None:
    """`x1 + x2 <= 100` : la borne calculee, 99, ne remplace pas la borne 3."""
    assert resserrer_les_bornes(Contrainte((1, 1), INF, 100), ((1, 3), (1, 3))) == (
        (1, 3),
        (1, 3),
    )
