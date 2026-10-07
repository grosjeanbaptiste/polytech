"""Modelisation, probleme VIII : choisir des examens pour depister cinq maladies.

Un jeu d'examens permet le diagnostic s'il distingue toute paire de
maladies : pour chaque paire, au moins un examen retenu doit repondre
differemment aux deux. C'est un probleme de couverture.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

RESULTATS: Final = (
    (1, 0, 1, 1, 0),
    (0, 1, 1, 1, 0),
    (1, 1, 0, 0, 1),
    (0, 0, 1, 1, 1),
    (1, 1, 0, 1, 0),
)
"""Planche 44 : une ligne par examen, une colonne par maladie."""
COUTS: Final = (6, 8, 2, 7, 6)
MALADIES: Final = len(RESULTATS[0])
PAIRES: Final = tuple(itertools.combinations(range(1, MALADIES + 1), 2))
Choix = tuple[int, ...]


def examens_qui_distinguent(premiere: int, seconde: int) -> tuple[int, ...]:
    """Les examens (de 1) qui ne repondent pas pareil a ces deux maladies."""
    return tuple(
        rang
        for rang, ligne in enumerate(RESULTATS, start=1)
        if ligne[premiere - 1] != ligne[seconde - 1]
    )


def programme() -> ProgrammeGeneral:
    """Le modele de la planche 45 : une contrainte par paire, puis `x_i <= 1`."""
    contraintes = []
    for paire in PAIRES:
        utiles = examens_qui_distinguent(*paire)
        ligne = tuple(1 if i in utiles else 0 for i in range(1, len(COUTS) + 1))
        contraintes.append(Contrainte(ligne, Sens.SUPERIEUR, 1))
    bornes = [
        Contrainte(tuple(int(j == i) for j in range(len(COUTS))), Sens.INFERIEUR, 1)
        for i in range(len(COUTS))
    ]
    return ProgrammeGeneral(Optimisation.MIN, COUTS, (*contraintes, *bornes))


def suffit(choix: Choix) -> bool:
    """Vrai si les examens retenus distinguent toute paire de maladies."""
    retenus = {rang for rang, pris in enumerate(choix, start=1) if pris}
    return all(retenus & set(examens_qui_distinguent(*paire)) for paire in PAIRES)


def choix_suffisants() -> tuple[tuple[int, Choix], ...]:
    """Tous les choix qui permettent le diagnostic, du moins cher au plus cher."""
    tous = itertools.product((0, 1), repeat=len(COUTS))
    couts = [
        (sum(c for c, pris in zip(COUTS, x, strict=True) if pris), x) for x in tous
    ]
    return tuple(sorted(paire for paire in couts if suffit(paire[1])))


def meilleur_choix() -> tuple[int, Choix]:
    """Le choix le moins cher, par enumeration des 32 possibles."""
    return choix_suffisants()[0]


def par_separation() -> tuple[sp.Expr | None, Choix | None]:
    """Le meme optimum, par « Branch and Bound » sur le modele."""
    entieres = frozenset(range(len(COUTS)))
    exploration = resoudre_en_variables_entieres(ProgrammeEntier(programme(), entieres))
    solution = exploration.solution
    return exploration.valeur, None if solution is None else tuple(solution)
