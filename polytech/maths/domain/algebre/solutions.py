"""L'ensemble des solutions d'un systeme lineaire `A x = b` quelconque.

La condition de compatibilite (planche 19) le decrit sans le resoudre :

* `rang(A) != rang(A | b)` : le systeme est **impossible** ;
* `rang(A) = rang(A | b) = n` : une solution, et une seule ;
* sinon il est **indetermine**, avec `n - rang(A)` parametres -- une
  droite, un plan, ... « simplement », « doublement » indetermine.

Un systeme homogene est toujours compatible : `x = 0` est solution.

La **methode de Cramer generalisee** (planche 16) le resout : chercher une
sous-matrice `M` d'ordre `rang(A)` et de determinant non nul ; negliger
les equations qui n'y interviennent pas ; faire passer au second membre
les inconnues hors de `M` ; resoudre le systeme carre par Cramer ; enfin
**verifier** que la solution satisfait tout le systeme.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.methodes import par_cramer
from polytech.maths.domain.algebre.systemes import SystemeLineaire

PARAMETRES: Final = sp.symbols("alpha beta gamma delta", real=True)
"""Les parametres d'une solution indeterminee, dans l'ordre ou le cours les prend."""


class Nature(Enum):
    """Ce que peut etre l'ensemble des solutions."""

    IMPOSSIBLE = "aucune solution"
    UNIQUE = "une solution et une seule"
    INDETERMINE = "une infinite de solutions"


@dataclass(frozen=True, slots=True)
class ReductionDeCramer:
    """La sous-matrice retenue, et la solution parametree qu'elle donne."""

    lignes: tuple[int, ...]
    colonnes: tuple[int, ...]
    sous_matrice: sp.Matrix
    solution: tuple[sp.Expr, ...]


def _est_compatible(systeme: SystemeLineaire) -> bool:
    augmentee = systeme.matrice.row_join(systeme.second_membre)
    return bool(systeme.rang() == augmentee.rank())


def nature_des_solutions(systeme: SystemeLineaire) -> Nature:
    """Impossible, unique ou indetermine, par la condition de compatibilite."""
    if not _est_compatible(systeme):
        return Nature.IMPOSSIBLE
    plein = systeme.rang() == systeme.matrice.cols
    return Nature.UNIQUE if plein else Nature.INDETERMINE


def nombre_de_parametres(systeme: SystemeLineaire) -> int:
    """`n - rang(A)` pour un systeme compatible ; zero s'il est impossible."""
    if not _est_compatible(systeme):
        return 0
    return int(systeme.matrice.cols) - systeme.rang()


def solution_generale(systeme: SystemeLineaire) -> tuple[sp.Expr, ...]:
    """La solution, les inconnues libres remplacees par `alpha`, `beta`, ..."""
    if not _est_compatible(systeme):
        msg = "systeme impossible : rang(A) different de rang(A | b)"
        raise ValueError(msg)
    inconnues = sp.symbols(f"x1:{systeme.matrice.cols + 1}")
    (solution,) = sp.linsolve((systeme.matrice, systeme.second_membre), inconnues)
    libres = [x for x in inconnues if any(x in s.free_symbols for s in solution)]
    renommage = dict(zip(libres, PARAMETRES, strict=False))
    return tuple(sp.simplify(s.subs(renommage)) for s in solution)


def _sous_matrice_reguliere(
    matrice: sp.Matrix,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordre = int(matrice.rank())
    for lignes in combinations(range(matrice.rows), ordre):
        for colonnes in combinations(range(matrice.cols), ordre):
            if matrice[list(lignes), list(colonnes)].det() != 0:
                return lignes, colonnes
    msg = "matrice nulle : aucune sous-matrice reguliere"
    raise ValueError(msg)


def cramer_generalise(systeme: SystemeLineaire) -> ReductionDeCramer:
    """Les cinq etapes de la planche 16."""
    lignes, colonnes = _sous_matrice_reguliere(systeme.matrice)
    sous_matrice = systeme.matrice[list(lignes), list(colonnes)]
    hors = [j for j in range(systeme.matrice.cols) if j not in colonnes]
    valeurs = dict(zip(hors, PARAMETRES, strict=False))
    second = sp.Matrix(
        [
            systeme.second_membre[i, 0]
            - sum(systeme.matrice[i, j] * valeurs[j] for j in hors)
            for i in lignes
        ]
    )
    principales = par_cramer(SystemeLineaire(sous_matrice, second))
    valeurs |= dict(zip(colonnes, principales, strict=True))
    solution = tuple(sp.simplify(valeurs[j]) for j in range(systeme.matrice.cols))
    ecart = sp.simplify(systeme.matrice * sp.Matrix(solution) - systeme.second_membre)
    if ecart != sp.zeros(systeme.matrice.rows, 1):
        msg = "les solutions obtenues ne satisfont pas le systeme : il est impossible"
        raise ValueError(msg)
    return ReductionDeCramer(lignes, colonnes, sous_matrice, solution)
