"""Chapitre XIV.2.2 : l'evaluation d'un sous-ensemble, et les tests qui le sondent.

Une **fonction d'evaluation** associe a chaque sous-ensemble `S_i` une
valeur `v_i` qui minore `F` sur `S_i` (regle de minoration) et lui est
egale sur un sous-ensemble terminal (regle de coincidence). On l'obtient
par **relaxation** : elargir `S_i` en `R_i` et resoudre sur `R_i`.

Connaissant la meilleure solution admissible trouvee, de valeur `F^`, un
sous-ensemble est **sonde** si :

* (alpha) `R_i` est vide -- a fortiori `S_i` ;
* (beta) la solution optimale du probleme relaxe appartient a `S_i` :
  c'est la solution optimale du noeud ;
* (gamma) `F^ <= v_i` : rien dans `S_i` ne fait strictement mieux.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp


class Sondage(Enum):
    """Les trois conditions de sondage d'un sous-ensemble (page 301)."""

    IMPOSSIBILITE = "alpha : le probleme relaxe n'a aucune solution"
    OPTIMALITE = "beta : la solution du probleme relaxe est admissible"
    DOMINANCE = "gamma : la meilleure solution connue fait aussi bien"


@dataclass(frozen=True, slots=True)
class Evaluation[S]:
    """`v_i`, la solution du probleme relaxe, et si elle appartient a `S`."""

    valeur: sp.Expr | None
    solution: S | None
    admissible: bool


@dataclass(frozen=True, slots=True)
class Noeud[P]:
    """Un sous-probleme examine : sa place, son evaluation, son sort."""

    probleme: P
    chemin: tuple[int, ...]
    evaluation: sp.Expr | None
    sondage: Sondage | None = None


@dataclass(frozen=True, slots=True)
class Exploration[P, S]:
    """La meilleure solution, sa valeur, et les noeuds dans l'ordre d'examen."""

    solution: S | None
    valeur: sp.Expr | None
    noeuds: tuple[Noeud[P], ...]


def sonder[S](evaluation: Evaluation[S], meilleure: sp.Expr | None) -> Sondage | None:
    """Lequel des trois tests sonde le noeud, ou `None` s'il faut le separer."""
    if evaluation.valeur is None:
        return Sondage.IMPOSSIBILITE
    if evaluation.admissible:
        return Sondage.OPTIMALITE
    if meilleure is not None and meilleure <= evaluation.valeur:
        return Sondage.DOMINANCE
    return None
