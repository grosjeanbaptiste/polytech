"""Les liens du chapitre II avec le chapitre I, et avec le simplexe deja en place.

Le chapitre II s'appuie sur le chapitre I : une iteration passe d'un
sommet a un sommet **adjacent**, l'optimum est la meilleure solution de
base admissible, un probleme impossible est un domaine vide. Ces tests
font repondre aux memes questions deux parties de la bibliotheque qui ne
partagent aucun calcul -- le tableau et ses pivots d'un cote, la
geometrie et l'enumeration des bases de l'autre.

Le dernier relie le simplexe du cours (tableau condense, minimisation ou
maximisation) a celui que la bibliotheque portait deja (`max c x`,
`A x <= b`, tableau etendu) : deux ecritures, un seul optimum.
"""

from __future__ import annotations

import itertools

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue, phase_deux
from polytech.optimisation.domain.bases import solutions_de_base
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.forme_standard import ProgrammeLineaire
from polytech.optimisation.domain.formes import FormeStandard, forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.nature import (
    NatureDeLOptimum,
    NatureDuDomaine,
    nature_de_l_optimum,
    nature_du_domaine,
)
from polytech.optimisation.domain.phase_un import methode_des_variables_artificielles
from polytech.optimisation.domain.polyedres import DemiEspace, Polyedre
from polytech.optimisation.domain.simplexe import Simplexe
from polytech.optimisation.domain.tableau import tableau_de_la_base

INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR
MAX = Optimisation.MAX
LIGNES = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))


def _figure_i6() -> FormeStandard:
    """`max 6 x1 + 5 x2` sur le polygone de la figure I.6, couts de `z`."""
    return FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


def _polygone() -> Polyedre:
    reelles = [DemiEspace(a, d) for a, d in LIGNES]
    return Polyedre((*reelles, DemiEspace((-1, 0), 0), DemiEspace((0, -1), 0)))


def test_chaque_iteration_passe_a_un_sommet_adjacent() -> None:
    """II.2.4 : le simplexe longe une arete du polyedre du chapitre I."""
    depart = tableau_de_la_base(_figure_i6(), (3, 4, 5))
    sommets = [list(t.solution(5))[:2] for t in phase_deux(depart, MAX).tableaux]
    assert sommets == [[0, 0], [2, 0], [5, 3]]
    for avant, apres in itertools.pairwise(sommets):
        assert _polygone().sont_adjacents(avant, apres)


def test_l_optimum_du_simplexe_est_la_meilleure_solution_de_base_admissible() -> None:
    """Le simplexe evite l'enumeration ; il doit retrouver ce qu'elle trouve."""
    forme = _figure_i6()
    admissibles = [s for s in solutions_de_base(forme) if s.admissible]
    meilleure = max((forme.couts * s.point)[0, 0] for s in admissibles)
    optimum = phase_deux(tableau_de_la_base(forme, (3, 4, 5)), MAX).tableaux[-1].valeur
    assert optimum == meilleure == 45


@pytest.mark.parametrize(
    ("sens", "issue", "domaine"),
    [
        ((INF, INF, INF), Issue.OPTIMALE, NatureDuDomaine.POLYTOPE),
        ((SUP, INF, INF), Issue.OPTIMALE, NatureDuDomaine.POLYTOPE),
        ((INF, SUP, SUP), Issue.IMPOSSIBLE, NatureDuDomaine.VIDE),
        ((SUP, INF, SUP), Issue.NON_BORNEE, NatureDuDomaine.NON_BORNE),
    ],
)
def test_l_issue_du_simplexe_est_celle_que_la_geometrie_annonce(
    sens: tuple[Sens, Sens, Sens],
    issue: Issue,
    domaine: NatureDuDomaine,
) -> None:
    """Figures I.6, II.4, I.8 et I.11 : impossible si `D` est vide, etc."""
    contraintes = tuple(
        Contrainte(a, s, d) for (a, d), s in zip(LIGNES, sens, strict=True)
    )
    programme = ProgrammeGeneral(MAX, (6, 5), contraintes)
    forme = forme_standard(programme)
    assert nature_du_domaine(forme) is domaine
    en_z = FormeStandard(-forme.couts, forme.matrice, forme.seconds_membres)
    assert methode_des_variables_artificielles(en_z, MAX).issue is issue
    non_bornee = nature_de_l_optimum(forme) is NatureDeLOptimum.NON_BORNEE
    assert non_bornee is (issue is Issue.NON_BORNEE)


def test_le_simplexe_du_cours_et_celui_deja_en_place_donnent_le_meme_optimum() -> None:
    """Tableau condense du cours contre tableau etendu : `(5, 3)`, `z = 45`."""
    ancien = Simplexe(
        ProgrammeLineaire(
            objectif=sp.Matrix([6, 5]),
            contraintes=sp.Matrix([[1, 1], [-2, 3], [1, -1]]),
            seconds_membres=sp.Matrix([8, 6, 2]),
        )
    )
    point, valeur = ancien.optimum()
    final = phase_deux(tableau_de_la_base(_figure_i6(), (3, 4, 5)), MAX).tableaux[-1]
    assert valeur == final.valeur == 45
    assert list(point) == list(final.solution(5))[:2] == [5, 3]
