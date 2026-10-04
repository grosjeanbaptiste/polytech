"""Chapitre I.3.4 : polyedres convexes, sommets, aretes, rayons.

Un **demi-espace ferme** est `{x | alpha x <= b}` ; l'**hyperplan**
`alpha x = b` en est la frontiere, et `alpha` lui est orthogonal, dirige
vers l'exterieur. Un **polyedre convexe** est l'intersection d'un nombre
fini de demi-espaces fermes ; borne et non vide, c'est un **polytope**.

Dans `R^n`, un point du polyedre est **interieur** si toutes les
inegalites y sont strictes, **frontiere** si l'une au moins y est une
egalite, et c'est un **sommet** si les hyperplans qui y passent sont de
rang `n` : il ne s'ecrit alors comme combinaison convexe d'aucune autre
paire de points. Deux sommets sont **adjacents** s'ils partagent `n - 1`
hyperplans independants -- une arete. Un polyedre non borne contient un
**rayon** `x + lambda d` ; si tous ses hyperplans passent par l'origine,
c'est un **cone polyedrique**.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import exacts

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class DemiEspace:
    """`{x | alpha x <= b}` : sa normale `alpha`, sortante, et sa borne `b`."""

    normale: tuple[sp.Expr, ...]
    borne: sp.Expr

    def __post_init__(self) -> None:
        """Rend les nombres exacts."""
        object.__setattr__(self, "normale", exacts(self.normale))
        object.__setattr__(self, "borne", sp.nsimplify(self.borne, rational=True))

    def ecart(self, point: Sequence[object]) -> sp.Expr:
        """`alpha x - b` : negatif dedans, nul sur l'hyperplan."""
        produit = (a * sp.sympify(x) for a, x in zip(self.normale, point, strict=True))
        return sp.Add(*produit) - self.borne

    def contient(self, point: Sequence[object]) -> bool:
        """Vrai si le point est dans le demi-espace ferme."""
        return bool(self.ecart(point) <= 0)

    def est_sur_l_hyperplan(self, point: Sequence[object]) -> bool:
        """Vrai si le point est sur la frontiere `alpha x = b`."""
        return bool(self.ecart(point) == 0)


@dataclass(frozen=True, slots=True)
class Polyedre:
    """L'intersection d'un nombre fini de demi-espaces fermes de `R^n`."""

    demi_espaces: tuple[DemiEspace, ...]

    def __post_init__(self) -> None:
        """Refuse des demi-espaces qui ne vivent pas dans le meme espace."""
        dimensions = {len(d.normale) for d in self.demi_espaces}
        if len(dimensions) > 1:
            msg = f"demi-espaces de dimensions differentes : {sorted(dimensions)}"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        """`n`, la dimension de l'espace des decisions."""
        return len(self.demi_espaces[0].normale)

    def contient(self, point: Sequence[object]) -> bool:
        """Vrai si le point verifie toutes les inegalites."""
        return all(d.contient(point) for d in self.demi_espaces)

    def actifs(self, point: Sequence[object]) -> tuple[DemiEspace, ...]:
        """Les demi-espaces dont l'hyperplan passe par le point."""
        return tuple(d for d in self.demi_espaces if d.est_sur_l_hyperplan(point))

    def est_interieur(self, point: Sequence[object]) -> bool:
        """Toutes les inegalites sont strictes."""
        return self.contient(point) and not self.actifs(point)

    def est_frontiere(self, point: Sequence[object]) -> bool:
        """Le point est dans le polyedre et touche au moins un hyperplan."""
        return self.contient(point) and bool(self.actifs(point))

    def est_sommet(self, point: Sequence[object]) -> bool:
        """Les hyperplans qui passent par le point sont de rang `n`."""
        return self.contient(point) and _rang(self.actifs(point)) == self.dimension

    def sommets(self) -> tuple[sp.Matrix, ...]:
        """Les intersections de `n` hyperplans independants qui restent dans `P`."""
        trouves: list[sp.Matrix] = []
        for choix in combinations(self.demi_espaces, self.dimension):
            matrice = sp.Matrix([list(d.normale) for d in choix])
            if matrice.det() == 0:
                continue
            point = matrice.solve(sp.Matrix([d.borne for d in choix]))
            if self.contient(list(point)) and point not in trouves:
                trouves.append(point)
        return tuple(trouves)

    def sont_adjacents(
        self, premier: Sequence[object], second: Sequence[object]
    ) -> bool:
        """Une arete les relie : `n - 1` hyperplans communs independants."""
        communs = [d for d in self.actifs(premier) if d in self.actifs(second)]
        deux_sommets = self.est_sommet(premier) and self.est_sommet(second)
        return deux_sommets and _rang(communs) == self.dimension - 1

    def est_un_cone(self) -> bool:
        """Tous les hyperplans passent par l'origine."""
        return all(d.borne == 0 for d in self.demi_espaces)


def _rang(demi_espaces: Sequence[DemiEspace]) -> int:
    if not demi_espaces:
        return 0
    return int(sp.Matrix([list(d.normale) for d in demi_espaces]).rank())
