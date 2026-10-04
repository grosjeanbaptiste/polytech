"""Chapitre IV.1 et IV.2 d'optimisation : le probleme dual et ses proprietes.

Oracles (`Optimisation/SLIDES/Slides_04_Definitions_proprietes_et_`
`interpretation_de_la_dualite.pdf`) : l'illustration des pages 92-93 --
un primal sous forme generale et son dual, recopies coefficient par
coefficient -- et les trois couples de la page 95, qui illustrent les
trois situations du theoreme fondamental.
"""

from __future__ import annotations

import pytest

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.dual import (
    Situation,
    dual_de,
    ecart_de_dualite,
    situation_du_couple,
    sous_la_convention,
)
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import Solution, resoudre

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
MAX, MIN = Optimisation.MAX, Optimisation.MIN


def _illustration() -> ProgrammeGeneral:
    """Page 92 : `max x1 - 4 x2`, deja ecrit sous la convention (`<=`)."""
    return ProgrammeGeneral(
        optimisation=MAX,
        objectif=(1, -4),
        contraintes=(
            Contrainte((1, 0), INF, 7),
            Contrainte((-3, 2), INF, -4),
            Contrainte((1, 1), EGA, 5),
        ),
        libres=frozenset({1}),
    )


def _couple(optimisation: Optimisation, sens: Sens) -> ProgrammeGeneral:
    """Page 95 : `2 x1 + 2 x2`, `-x1 + x2 ? 1`, `x1 - x2 ? 1`, `x >= 0`."""
    contraintes = (Contrainte((-1, 1), sens, 1), Contrainte((1, -1), sens, 1))
    return ProgrammeGeneral(optimisation, (2, 2), contraintes)


def test_le_dual_de_l_illustration_est_celui_de_la_page_93() -> None:
    """`min 7 u1 - 4 u2 + 5 u3`, `u1 - 3 u2 + u3 >= 1`, `2 u2 + u3 = -4`, `u3` libre."""
    dual = dual_de(_illustration())
    assert dual.optimisation is MIN
    assert dual.objectif == (7, -4, 5)
    assert dual.contraintes == (
        Contrainte((1, -3, 1), SUP, 1),
        Contrainte((0, 2, 1), EGA, -4),
    )
    assert dual.libres == frozenset({2})


def test_la_convention_retourne_les_inegalites_mal_orientees() -> None:
    """Page 91 : en minimisation, les inegalites s'ecrivent `>=`."""
    primal = _couple(MIN, INF)
    assert sous_la_convention(primal).contraintes == (
        Contrainte((1, -1), SUP, -1),
        Contrainte((-1, 1), SUP, -1),
    )


def test_la_dualite_est_involutive() -> None:
    """« Le probleme dual du dual est le probleme primal initial » (page 92)."""
    assert dual_de(dual_de(_illustration())) == _illustration()


def test_le_dual_du_premier_couple_est_celui_de_la_page_95() -> None:
    """`max -u1 - u2`, `u1 - u2 <= 2`, `-u1 + u2 <= 2`, `u >= 0`."""
    dual = dual_de(_couple(MIN, INF))
    assert dual.optimisation is MAX
    assert dual.objectif == (-1, -1)
    assert set(dual.contraintes) == {
        Contrainte((1, -1), INF, 2),
        Contrainte((-1, 1), INF, 2),
    }


def test_la_dualite_faible_borne_chaque_probleme_par_l_autre() -> None:
    """Propriete 1 : `c x >= u d` pour `x` et `u` admissibles (primal minimum)."""
    primal = _couple(MIN, INF)
    assert ecart_de_dualite(primal, x=(1, 1), u=(1, 0)) == 4 - (-1)
    assert ecart_de_dualite(primal, x=(0, 0), u=(0, 0)) == 0


def test_la_dualite_forte_egale_les_deux_optimums() -> None:
    """Propriete 2 : `z = w` ; ici 15 pour l'illustration des pages 92-93."""
    primal = _illustration()
    assert resoudre(primal).valeur == resoudre(dual_de(primal)).valeur == 15


@pytest.mark.parametrize(
    ("optimisation", "sens", "situation", "issues"),
    [
        (MIN, INF, Situation.DEUX_OPTIMUMS_FINIS, (Issue.OPTIMALE, Issue.OPTIMALE)),
        (
            MAX,
            INF,
            Situation.NON_BORNE_ET_IMPOSSIBLE,
            (Issue.NON_BORNEE, Issue.IMPOSSIBLE),
        ),
        (MAX, SUP, Situation.DEUX_IMPOSSIBLES, (Issue.IMPOSSIBLE, Issue.IMPOSSIBLE)),
    ],
)
def test_les_trois_situations_du_theoreme_fondamental(
    optimisation: Optimisation,
    sens: Sens,
    situation: Situation,
    issues: tuple[Issue, Issue],
) -> None:
    """Page 95 : les trois couples illustrant le corollaire 5."""
    primal = _couple(optimisation, sens)
    assert situation_du_couple(primal) is situation
    assert (resoudre(primal).issue, resoudre(dual_de(primal)).issue) == issues


def test_le_premier_couple_a_ses_deux_optimums_a_l_origine() -> None:
    """Page 95 : `x = (0, 0)` et `u = (0, 0)`."""
    primal = _couple(MIN, INF)
    assert resoudre(primal).point == (0, 0)
    assert resoudre(dual_de(primal)).point == (0, 0)


def test_un_couple_contraire_au_theoreme_est_une_erreur_et_non_une_situation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deux problemes non bornes : le theoreme l'exclut, un solveur fautif non."""
    monkeypatch.setattr(
        "polytech.optimisation.domain.dual.resoudre",
        lambda _programme: Solution(Issue.NON_BORNEE),
    )
    with pytest.raises(RuntimeError, match="contraire au theoreme fondamental"):
        situation_du_couple(_couple(MIN, INF))


def test_la_dualite_faible_d_un_primal_de_maximisation() -> None:
    """Corollaire 1 : `w >= c x`. Page 99 : `x = (2, 0)` vaut 12, `u = (6, 0, 0)` 48."""
    lignes = (((1, 1), 8), ((-2, 3), 6), ((1, -1), 2))
    contraintes = tuple(Contrainte(a, INF, d) for a, d in lignes)
    primal = ProgrammeGeneral(MAX, (6, 5), contraintes)
    assert ecart_de_dualite(primal, x=(2, 0), u=(6, 0, 0)) == 48 - 12


def test_la_situation_symetrique_primal_impossible_et_dual_non_borne() -> None:
    """Corollaire 5, cas 2.b : le deuxieme couple de la page 95, lu a l'envers."""
    primal = dual_de(_couple(MAX, INF))
    assert situation_du_couple(primal) is Situation.IMPOSSIBLE_ET_NON_BORNE
