"""Les trois proprietes qui font qu'un systeme est justiciable du cours.

Lineaire, invariant, causal. Le cours ne les enonce pas pour le plaisir :
c'est leur conjonction qui autorise la convolution, donc la transmittance,
donc tout le reste. Un systeme qui en manque une sort du cadre, et le
corrige doit le dire avant de calculer quoi que ce soit.

Les proprietes sont testees sur des signaux symboliques, et non sur des
echantillons : deux signaux quelconques `x1` et `x2`, et un retard
quelconque. C'est une verification, pas un sondage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Systeme:
    """Un systeme donne par l'operateur qui transforme son entree."""

    operateur: Callable[[sp.Expr], sp.Expr]

    def est_lineaire(self) -> bool:
        """`S(a x1 + b x2) = a S(x1) + b S(x2)` pour des signaux quelconques."""
        premier, second = sp.Function("x_1"), sp.Function("x_2")
        a, b = sp.symbols("a b")
        combinee = a * premier(TEMPS) + b * second(TEMPS)
        ecart = self.operateur(combinee) - (
            a * self.operateur(premier(TEMPS)) + b * self.operateur(second(TEMPS))
        )
        return bool(sp.simplify(sp.expand(ecart)) == 0)

    def est_invariant(self) -> bool:
        """Retarder l'entree doit retarder la sortie d'autant."""
        entree = sp.Function("x")
        retard = sp.Symbol("tau", positive=True)
        retardee = self.operateur(entree(TEMPS - retard))
        attendue = self.operateur(entree(TEMPS)).subs(TEMPS, TEMPS - retard)
        return bool(sp.simplify(retardee - attendue) == 0)


def convolution(premier: sp.Expr, second: sp.Expr) -> sp.Expr:
    """Le produit de convolution de deux signaux causaux.

    Sur des signaux causaux l'integrale court de zero a `t`, et non de
    moins l'infini a plus l'infini : c'est la causalite qui raccourcit les
    bornes, et l'ecrire ici evite de la reinventer a chaque exercice.
    """
    variable = sp.Symbol("theta", nonnegative=True)
    integrande = premier.subs(TEMPS, variable) * second.subs(TEMPS, TEMPS - variable)
    return sp.simplify(sp.integrate(integrande, (variable, 0, TEMPS)))
