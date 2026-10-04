"""Resoudre un programme lineaire sous sa forme generale, de bout en bout.

La chaine du cours, d'un seul geste : variables non negatives,
minimisation, forme standard (chapitre I), phase I puis phase II
(chapitre II), et retour aux variables et au sens d'origine. Oracles :
l'illustration de la page 25 du chapitre I, resolue ici a la main, et
les figures deja utilisees aux chapitres I et II.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
MAX, MIN = Optimisation.MAX, Optimisation.MIN
LIGNES = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))


def _figure(
    sens: tuple[Sens, Sens, Sens], optimisation: Optimisation
) -> ProgrammeGeneral:
    contraintes = tuple(
        Contrainte(a, s, d) for (a, d), s in zip(LIGNES, sens, strict=True)
    )
    return ProgrammeGeneral(optimisation, (6, 5), contraintes)


def test_une_maximisation_rend_sa_valeur_dans_le_sens_d_origine() -> None:
    """Figure I.9 : `max 6 x1 + 5 x2 = 45` en `(5, 3)`, et non `-45`."""
    solution = resoudre(_figure((INF, INF, INF), MAX))
    assert solution.issue is Issue.OPTIMALE
    assert solution.valeur == 45
    assert solution.point == (5, 3)


def test_une_variable_libre_peut_etre_negative_a_l_optimum() -> None:
    """Page 25 : `x2 = 5 - x1`, `z = 5 x1 - 20`, `x1 <= 7` ; donc `(7, -2)`."""
    programme = ProgrammeGeneral(
        optimisation=MAX,
        objectif=(1, -4),
        contraintes=(
            Contrainte((1, 0), INF, 7),
            Contrainte((3, -2), SUP, 4),
            Contrainte((1, 1), EGA, 5),
        ),
        libres=frozenset({1}),
    )
    solution = resoudre(programme)
    assert solution.point == (7, -2)
    assert solution.valeur == 15


def test_une_minimisation_sur_un_domaine_non_borne() -> None:
    """Figure I.7 : `min 6 x1 + 5 x2` au sommet `(12, 10)`."""
    solution = resoudre(_figure((SUP, SUP, SUP), MIN))
    assert solution.point == (12, 10)
    assert solution.valeur == 6 * 12 + 5 * 10


def test_un_probleme_impossible_n_a_ni_valeur_ni_point() -> None:
    """Figure I.8 : `D` est vide."""
    solution = resoudre(_figure((INF, SUP, SUP), MAX))
    assert solution.issue is Issue.IMPOSSIBLE
    assert solution.valeur is None
    assert solution.point is None


def test_un_probleme_non_borne_n_a_ni_valeur_ni_point() -> None:
    """Figure I.11."""
    solution = resoudre(_figure((SUP, INF, SUP), MAX))
    assert solution.issue is Issue.NON_BORNEE
    assert solution.valeur is None


def test_la_solution_trouvee_verifie_les_contraintes_d_origine() -> None:
    programme = _figure((SUP, INF, INF), MAX)
    solution = resoudre(programme)
    assert solution.point is not None
    assert all(c.est_satisfaite(solution.point) for c in programme.contraintes)
    assert programme.valeur(solution.point) == solution.valeur == sp.Integer(122)
