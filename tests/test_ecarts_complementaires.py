"""Chapitre IV.2.3 et le lemme de Farkas : les couples de contraintes duales.

Oracle (`Optimisation/SLIDES/Slides_04_Definitions_proprietes_et_`
`interpretation_de_la_dualite.pdf`, page 99) : pour `max 6 x1 + 5 x2`
sur le polygone de la figure I.6, l'optimum primal `(5, 3)` impose
`u2 = 0`, `u1 - 2 u2 + u3 = 6`, `u1 + 3 u2 - u3 = 5`, d'ou le dual
optimal `(11/2, 0, 1/2)` -- et trois contraintes serrees sur cinq couples.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.ecarts_complementaires import (
    CoupleDeContraintes,
    couples_de_contraintes,
    duale_par_les_ecarts,
    verifie_les_ecarts_complementaires,
)
from polytech.optimisation.domain.farkas import Alternative, alternative_de_farkas
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import Solution

R = sp.Rational
INF = Sens.INFERIEUR
DUALE = (R(11, 2), 0, R(1, 2))


def _primal() -> ProgrammeGeneral:
    """Page 99 : `max 6 x1 + 5 x2` sur le polygone de la figure I.6."""
    lignes = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))
    contraintes = tuple(Contrainte(a, INF, d) for a, d in lignes)
    return ProgrammeGeneral(Optimisation.MAX, (6, 5), contraintes)


def test_l_optimum_et_son_dual_verifient_les_ecarts_complementaires() -> None:
    """`u (T x - d) = 0` et `(c - u T) x = 0`."""
    assert verifie_les_ecarts_complementaires(_primal(), x=(5, 3), u=DUALE)


def test_un_point_admissible_non_optimal_ne_les_verifie_pas() -> None:
    """Le sommet `(2, 0)` est admissible, mais `z = 12` n'est pas l'optimum."""
    assert not verifie_les_ecarts_complementaires(_primal(), x=(2, 0), u=DUALE)


def test_une_duale_non_optimale_ne_les_verifie_pas() -> None:
    """`u = (6, 0, 0)` est admissible pour le dual, de valeur 48 > 45."""
    assert not verifie_les_ecarts_complementaires(_primal(), x=(5, 3), u=(6, 0, 0))


def test_les_cinq_couples_de_la_page_99() -> None:
    """Deux variables primales positives, puis les trois contraintes reelles."""
    couples = couples_de_contraintes(_primal(), x=(5, 3), u=DUALE)
    assert couples == (
        CoupleDeContraintes("x_1 >= 0", 5, "contrainte duale 1", 0),
        CoupleDeContraintes("x_2 >= 0", 3, "contrainte duale 2", 0),
        CoupleDeContraintes("contrainte primale 1", 0, "u_1 >= 0", R(11, 2)),
        CoupleDeContraintes("contrainte primale 2", 7, "u_2 >= 0", 0),
        CoupleDeContraintes("contrainte primale 3", 0, "u_3 >= 0", R(1, 2)),
    )


def test_a_l_optimum_chaque_couple_a_au_moins_une_contrainte_serree() -> None:
    """L'enonce encadre de la page 97."""
    couples = couples_de_contraintes(_primal(), x=(5, 3), u=DUALE)
    assert all(couple.a_une_contrainte_serree for couple in couples)
    assert all(couple.a_exactement_une_contrainte_serree for couple in couples)


def test_hors_de_l_optimum_un_couple_n_a_aucune_contrainte_serree() -> None:
    """En `(2, 0)` : `x1 + x2 = 2 < 8`, et pourtant `u1 = 6 > 0`."""
    couples = couples_de_contraintes(_primal(), x=(2, 0), u=(6, 0, 0))
    assert not couples[2].a_une_contrainte_serree
    assert not couples[2].a_exactement_une_contrainte_serree


def test_un_couple_deux_fois_serre_n_est_pas_exactement_serre() -> None:
    """La degenerescence : la forme forte du theoreme ne vaut plus telle quelle."""
    couple = CoupleDeContraintes("x_1 >= 0", 0, "contrainte duale 1", 0)
    assert couple.a_une_contrainte_serree
    assert not couple.a_exactement_une_contrainte_serree


def test_la_duale_optimale_se_deduit_de_la_primale() -> None:
    """Page 99 : le systeme de trois equations a trois inconnues."""
    assert duale_par_les_ecarts(_primal(), x=(5, 3)) == DUALE


def test_une_primale_degeneree_ne_determine_pas_la_duale() -> None:
    """Deux contraintes serrees pour une variable positive : une equation, deux u.

    `max x1 + x2`, `x1 + x2 <= 1`, `x1 <= 1`, a l'optimum `(1, 0)`.
    """
    primal = ProgrammeGeneral(
        Optimisation.MAX,
        (1, 1),
        (Contrainte((1, 1), INF, 1), Contrainte((1, 0), INF, 1)),
    )
    with pytest.raises(ValueError, match="ne determinent pas la solution duale"):
        duale_par_les_ecarts(primal, x=(1, 0))


def test_farkas_un_systeme_qui_a_une_solution_non_negative() -> None:
    """`x1 + x2 = 2`, `x1 - x2 = 0` : `x = (1, 1)`."""
    alternative = alternative_de_farkas(sp.Matrix([[1, 1], [1, -1]]), sp.Matrix([2, 0]))
    assert alternative.cote is Alternative.SOLUTION_NON_NEGATIVE
    assert alternative.temoin == (1, 1)


def test_farkas_sinon_un_certificat_d_impossibilite() -> None:
    """`x1 + x2 = -1` n'a pas de solution `x >= 0` ; alors `u T <= 0`, `u d > 0`."""
    matrice, second = sp.Matrix([[1, 1]]), sp.Matrix([-1])
    alternative = alternative_de_farkas(matrice, second)
    assert alternative.cote is Alternative.CERTIFICAT
    u = sp.Matrix([alternative.temoin])
    assert all(v <= 0 for v in u * matrice)
    assert (u * second)[0, 0] > 0


def test_farkas_le_certificat_d_un_systeme_a_deux_equations() -> None:
    """`x1 = 1`, `x1 = 2` : incompatible ; `u = (-1, 1)` convient, `u d = 1`."""
    matrice, second = sp.Matrix([[1], [1]]), sp.Matrix([1, 2])
    alternative = alternative_de_farkas(matrice, second)
    u = sp.Matrix([alternative.temoin])
    assert alternative.cote is Alternative.CERTIFICAT
    assert all(v <= 0 for v in u * matrice)
    assert (u * second)[0, 0] == 1


def test_farkas_un_solveur_qui_ne_rend_ni_solution_ni_certificat_est_une_erreur(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le lemme exclut ce cas ; s'il survient, c'est le solveur qui est fautif."""
    monkeypatch.setattr(
        "polytech.optimisation.domain.farkas.resoudre",
        lambda _programme: Solution(Issue.IMPOSSIBLE),
    )
    with pytest.raises(RuntimeError, match="ni solution ni certificat"):
        alternative_de_farkas(sp.Matrix([[1, 1]]), sp.Matrix([-1]))
