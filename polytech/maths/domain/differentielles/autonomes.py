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

import sympy as sp

VARIABLE = sp.Symbol("x", real=True)
INCONNUE = sp.Symbol("y", real=True)


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

    def isocline(self, pente: sp.Expr) -> sp.Eq:
        """L'equation `f(y) = p` de la ligne ou toutes les pentes valent `p`."""
        return sp.Eq(self.second_membre, pente)
