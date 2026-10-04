"""Les EDO autonomes : points critiques, stabilite, isoclines.

Une EDO est autonome quand son second membre ne depend pas de `x`. Le
fascicule en traite une en detail -- la logistique
`y' = lambda y (1 - y/K)` -- et pose sur elle les trois questions de ce
module : ses points critiques, l'allure de ses solutions, et l'equation
de ses lignes isoclines.

La stabilite se lit sur le signe de `f'` au point critique, et non sur une
esquisse : `f'(y*) < 0` attire, `f'(y*) > 0` repousse.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

VARIABLE = sp.Symbol("x", real=True)
INCONNUE = sp.Symbol("y", real=True)


ORDRE_MAXIMAL = 8
"""Jusqu'ou chercher la premiere derivee non nulle de `f` en un point critique."""


class Stabilite(Enum):
    """Les types de points critiques de la planche 34."""

    STABLE = "stable (attractif)"
    INSTABLE = "instable (repulsif)"
    SEMI_STABLE = "semi-stable"


@dataclass(frozen=True, slots=True)
class EquationAutonome:
    """`y' = f(y)`, dont le second membre ignore `x`."""

    second_membre: sp.Expr

    def __post_init__(self) -> None:
        """Refuse une equation non autonome : elle n'a pas de portrait de phase."""
        if self.second_membre.has(VARIABLE):
            msg = f"equation non autonome : {self.second_membre} depend de x"
            raise ValueError(msg)

    def points_critiques(self) -> tuple[sp.Expr, ...]:
        """Les valeurs qui annulent `y'`, dans l'ordre."""
        racines = sp.solve(sp.Eq(self.second_membre, 0), INCONNUE)
        return tuple(sorted(racines, key=sp.default_sort_key))

    def est_stable(self, point: sp.Expr) -> bool:
        """Vrai si `f'` y est negative : les solutions voisines y reviennent."""
        pente = sp.diff(self.second_membre, INCONNUE).subs(INCONNUE, point)
        return bool(sp.simplify(pente).is_negative)

    def sens_de_variation(self, valeur: object) -> int:
        """Le signe de `f(y)` : +1, -1 ou 0 -- une solution autonome est monotone."""
        return int(sp.sign(self.second_membre.subs(INCONNUE, sp.sympify(valeur))))

    def nature(self, point: object) -> Stabilite:
        """Stable, instable ou semi-stable, par la premiere derivee non nulle de `f`."""
        critique = sp.sympify(point)
        if sp.simplify(self.second_membre.subs(INCONNUE, critique)) != 0:
            msg = f"{critique} n'est pas un point critique : f({critique}) non nul"
            raise ValueError(msg)
        derivee = self.second_membre
        for ordre in range(1, ORDRE_MAXIMAL + 1):
            derivee = sp.diff(derivee, INCONNUE)
            valeur = sp.simplify(derivee.subs(INCONNUE, critique))
            if valeur == 0:
                continue
            if ordre % 2 == 0:
                return Stabilite.SEMI_STABLE
            return Stabilite.STABLE if valeur.is_negative else Stabilite.INSTABLE
        msg = f"f est plate en {critique} jusqu'a l'ordre {ORDRE_MAXIMAL}"
        raise ValueError(msg)

    def isocline(self, pente: sp.Expr) -> sp.Eq:
        """L'equation `f(y) = p` de la ligne ou toutes les pentes valent `p`."""
        return sp.Eq(self.second_membre, pente)
