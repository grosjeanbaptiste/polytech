"""Le probleme de Cauchy : une equation differentielle et une condition.

Etant donne `(x0, y0)`, trouver une solution de `y' = f(x, y)` telle que
`y(x0) = y0` (planche 13). Il equivaut a une **equation integrale**
(planche 14) :

    y(x) = y0 + int_{x0}^{x} f(u, y(u)) du,

dont la solution est le point fixe d'un operateur -- d'ou les
approximations successives, qui partent de la constante `y0`.

Deux questions avant de resoudre (planche 15) : l'**existence** et
l'**unicite**. Le seau troue, `h' = gamma sqrt(h)`, a une infinite de
solutions pour la meme condition ; la datation au carbone 14,
`x' = -beta x`, en a une seule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
)

MUETTE: Final = sp.Symbol("u", real=True)
ECART: Final = sp.Symbol("s", positive=True)


def _est_nul_sur(reste: sp.Expr, intervalle: sp.Set) -> bool:
    """Vrai si `reste` s'annule a l'interieur de l'intervalle, parametre par `s > 0`."""
    bas, haut = intervalle.inf, intervalle.sup
    if bas == -sp.oo and haut == sp.oo:
        point = VARIABLE
    elif bas == -sp.oo:
        point = haut - ECART
    elif haut == sp.oo:
        point = bas + ECART
    else:
        point = bas + (haut - bas) / (1 + ECART)
    return bool(sp.simplify(reste.subs(VARIABLE, point)) == 0)


@dataclass(frozen=True, slots=True)
class ProblemeDeCauchy:
    """`y' = f(x, y)` et `y(x0) = y0`, le second membre ecrit en `x` et `y`."""

    second_membre: sp.Expr
    x0: sp.Expr
    y0: sp.Expr

    def _second_membre_le_long_de(self, candidate: sp.Expr) -> sp.Expr:
        return self.second_membre.subs(INCONNUE, candidate)

    def est_solution(self, candidate: sp.Expr) -> bool:
        """Vrai si la fonction verifie la condition et, par morceaux, l'equation."""
        if sp.simplify(candidate.subs(VARIABLE, self.x0) - self.y0) != 0:
            return False
        morceaux = (
            candidate.as_expr_set_pairs()
            if isinstance(candidate, sp.Piecewise)
            else [(candidate, sp.S.Reals)]
        )
        return all(
            _est_nul_sur(
                sp.diff(expression, VARIABLE)
                - self._second_membre_le_long_de(expression),
                domaine,
            )
            for expression, domaine in morceaux
        )

    def iterees_de_picard(self, nombre: int) -> tuple[sp.Expr, ...]:
        """`y_0 = y0`, puis `y_{k+1}(x) = y0 + int_{x0}^{x} f(u, y_k(u)) du`."""
        iterees = [sp.sympify(self.y0)]
        for _ in range(nombre):
            integrande = self._second_membre_le_long_de(iterees[-1])
            integrale = sp.integrate(
                integrande.subs(VARIABLE, MUETTE), (MUETTE, self.x0, VARIABLE)
            )
            iterees.append(sp.expand(self.y0 + integrale))
        return tuple(iterees)

    def verifie_l_equation_integrale(self, candidate: sp.Expr) -> bool:
        """Vrai si `y(x) = y0 + int_{x0}^{x} f(u, y(u)) du`."""
        integrande = self._second_membre_le_long_de(candidate).subs(VARIABLE, MUETTE)
        integrale = sp.integrate(integrande, (MUETTE, self.x0, VARIABLE))
        return bool(sp.simplify(candidate - self.y0 - integrale) == 0)

    def solution(self) -> sp.Expr:
        """La solution que rend la resolution formelle, condition comprise."""
        inconnue = FONCTION(VARIABLE)
        equation = sp.Eq(
            inconnue.diff(VARIABLE), self.second_membre.subs(INCONNUE, inconnue)
        )
        resolue = sp.dsolve(
            equation, inconnue, ics={inconnue.subs(VARIABLE, self.x0): self.y0}
        )
        return resolue.rhs
