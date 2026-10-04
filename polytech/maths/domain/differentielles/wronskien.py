"""La matrice de Wronsky d'un systeme lineaire `y' = P(t) y`.

Les solutions du systeme homogene forment un espace vectoriel de
dimension `n`. Une matrice `W(t)` dont les colonnes sont `n` solutions
independantes les donne toutes : `y = W(t) C`. Son determinant ne s'annule
nulle part ou s'annule partout, et se deduit de sa valeur en un point :

    det W(t) = det W(t_0) exp(int_{t_0}^{t} trace P(tau) dtau).

Le noyau `R(t_1, t_2) = W(t_1) W^-1(t_2)` porte la solution du probleme de
Cauchy, second membre compris :

    y(t) = R(t, t_0) y_0 + int_{t_0}^{t} R(t, tau) x(tau) dtau.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Wronskien:
    """`W(t)`, une solution par colonne."""

    matrice: sp.Matrix

    @classmethod
    def des_solutions(cls, solutions: Sequence[sp.Expr]) -> Wronskien:
        """Celui d'une equation d'ordre `n` : les solutions et leurs derivees."""
        ordre = len(solutions)
        return cls(
            sp.Matrix(ordre, ordre, lambda k, i: sp.diff(solutions[i], TEMPS, k))
        )

    def determinant(self) -> sp.Expr:
        """`det W(t)`."""
        return sp.simplify(self.matrice.det())

    def resout(self, systeme: sp.Matrix) -> bool:
        """Vrai si chaque colonne verifie `y' = P(t) y`."""
        ecart = sp.simplify(sp.diff(self.matrice, TEMPS) - systeme * self.matrice)
        return bool(ecart == sp.zeros(*self.matrice.shape))

    def est_fondamental(self) -> bool:
        """Vrai si les colonnes sont lineairement independantes."""
        return bool(self.determinant() != 0)

    def noyau(self, premier: object, second: object) -> sp.Matrix:
        """`R(t_1, t_2) = W(t_1) W^-1(t_2)`."""
        inverse = self.matrice.subs(TEMPS, second).inv()
        return sp.simplify(self.matrice.subs(TEMPS, premier) * inverse)

    def solution_de_cauchy(
        self, t0: object, depart: sp.Matrix, force: sp.Matrix | None = None
    ) -> sp.Matrix:
        """`R(t, t_0) y_0`, plus `int R(t, tau) x(tau) dtau` s'il y a une force."""
        libre = self.noyau(TEMPS, t0) * depart
        if force is None:
            return sp.simplify(libre)
        muette = sp.Symbol("tau", positive=True)
        integrande = self.noyau(TEMPS, muette) * force.subs(TEMPS, muette)
        forcee = integrande.applyfunc(lambda e: sp.integrate(e, (muette, t0, TEMPS)))
        return sp.simplify(libre + forcee)


def determinant_par_la_trace(
    systeme: sp.Matrix, *, t0: object, initial: object
) -> sp.Expr:
    """`det W(t_0) exp(int_{t_0}^{t} trace P)`."""
    muette = sp.Symbol("tau", positive=True)
    trace = systeme.trace().subs(TEMPS, muette)
    return sp.simplify(initial * sp.exp(sp.integrate(trace, (muette, t0, TEMPS))))
