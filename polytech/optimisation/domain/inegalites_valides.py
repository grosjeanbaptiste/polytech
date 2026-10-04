"""Chapitre XIV.1.3 et XIV.3.5 d) : les inegalites valides, et les coupes.

Plusieurs polyedres decrivent le meme ensemble `S` de points entiers ; le
plus restrictif donne la meilleure relaxation lineaire. On resserre une
formulation en lui ajoutant des **inegalites valides** : des contraintes
que toutes les solutions entieres admissibles verifient, et qui ne
peuvent donc en eliminer aucune.

Une **coupe** est une inegalite valide choisie pour ne pas etre verifiee
par la solution optimale du probleme relaxe : elle retranche ce point --
et un morceau du polyedre -- sans toucher a `S`. C'est le principe des
methodes de coupure (Gomory), que le cours cite sans les developper, et
du « branch and cut » qui les integre a la separation et evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.entiers import points_entiers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.contraintes import Contrainte
    from polytech.optimisation.domain.formulation import ProgrammeGeneral


def est_une_inegalite_valide(
    contrainte: Contrainte,
    programme: ProgrammeGeneral,
    boite: Sequence[tuple[int, int]],
) -> bool:
    """Vrai si tous les points entiers du programme, dans la boite, la verifient."""
    return all(contrainte.est_satisfaite(p) for p in points_entiers(programme, boite))


def est_une_coupe(
    contrainte: Contrainte,
    programme: ProgrammeGeneral,
    boite: Sequence[tuple[int, int]],
    optimum_continu: Sequence[object],
) -> bool:
    """Vrai si l'inegalite est valide et ecarte la solution du probleme relaxe."""
    valide = est_une_inegalite_valide(contrainte, programme, boite)
    return valide and not contrainte.est_satisfaite(optimum_continu)
