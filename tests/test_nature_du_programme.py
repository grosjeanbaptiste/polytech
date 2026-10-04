"""Chapitre I.4.1 et I.4.2 d'optimisation : ce qu'est `D`, ce qu'est l'optimum.

Oracles : les six illustrations du cours
(`Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
figures I.6 a I.11, pages 38 a 41). Les memes trois droites, dont on
retourne les inegalites, donnent un polytope, un polyedre non borne et un
ensemble vide ; trois fonctions economiques donnent une solution unique,
une face de solutions, et aucune solution finie.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import FormeStandard, forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.nature import (
    NatureDeLOptimum,
    NatureDuDomaine,
    nature_de_l_optimum,
    nature_du_domaine,
    rayons_extremes,
)

INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR


def _programme(
    objectif: tuple[int, int], sens: tuple[Sens, Sens, Sens]
) -> FormeStandard:
    """`x1 + x2 ? 8`, `-2 x1 + 3 x2 ? 6`, `x1 - x2 ? 2`, a maximiser."""
    lignes = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))
    contraintes = tuple(
        Contrainte(a, s, d) for (a, d), s in zip(lignes, sens, strict=True)
    )
    return forme_standard(ProgrammeGeneral(Optimisation.MAX, objectif, contraintes))


def test_figure_i6_le_domaine_est_un_polytope() -> None:
    assert (
        nature_du_domaine(_programme((6, 5), (INF, INF, INF)))
        is NatureDuDomaine.POLYTOPE
    )


def test_figure_i7_le_domaine_est_non_borne() -> None:
    assert (
        nature_du_domaine(_programme((6, 5), (SUP, SUP, SUP)))
        is NatureDuDomaine.NON_BORNE
    )


def test_figure_i8_le_domaine_est_vide() -> None:
    assert (
        nature_du_domaine(_programme((6, 5), (INF, SUP, SUP))) is NatureDuDomaine.VIDE
    )


def test_figure_i9_la_solution_optimale_est_unique() -> None:
    assert (
        nature_de_l_optimum(_programme((6, 5), (INF, INF, INF)))
        is NatureDeLOptimum.UNIQUE
    )


def test_figure_i10_une_face_entiere_est_optimale() -> None:
    """`z = 5 x1 + 5 x2` est parallele a la contrainte (1)."""
    nature = nature_de_l_optimum(_programme((5, 5), (INF, INF, INF)))
    assert nature is NatureDeLOptimum.INFINITE


def test_figure_i11_il_n_y_a_pas_de_solution_optimale_finie() -> None:
    nature = nature_de_l_optimum(_programme((6, 5), (SUP, INF, SUP)))
    assert nature is NatureDeLOptimum.NON_BORNEE


def test_un_domaine_vide_n_a_pas_de_solution() -> None:
    nature = nature_de_l_optimum(_programme((6, 5), (INF, SUP, SUP)))
    assert nature is NatureDeLOptimum.SANS_SOLUTION


def test_un_domaine_non_borne_peut_avoir_un_optimum_fini() -> None:
    """Minimiser `x1 + x2` sur la figure I.7 : le rayon monte, le minimum existe."""
    nature = nature_de_l_optimum(_programme((-1, -1), (SUP, SUP, SUP)))
    assert nature is NatureDeLOptimum.UNIQUE


def test_un_rayon_extreme_est_une_direction_du_domaine() -> None:
    """`T d = 0`, `d >= 0`, `d != 0` : le long de `d`, on reste dans `D`."""
    forme = _programme((6, 5), (SUP, SUP, SUP))
    rayons = rayons_extremes(forme)
    assert rayons
    for d in rayons:
        assert forme.matrice * d == sp.zeros(forme.matrice.rows, 1)
        assert all(v >= 0 for v in d)
        assert any(v > 0 for v in d)


def test_un_polytope_n_a_aucun_rayon() -> None:
    assert rayons_extremes(_programme((6, 5), (INF, INF, INF))) == ()


@pytest.mark.parametrize("sens", [(INF, INF, INF), (SUP, SUP, SUP)])
def test_un_domaine_non_vide_a_au_moins_un_sommet(
    sens: tuple[Sens, Sens, Sens],
) -> None:
    assert nature_du_domaine(_programme((6, 5), sens)) is not NatureDuDomaine.VIDE
