"""L'integrale definie, par Newton-Leibniz et ses deux evaluations.

Ce que le corrige doit montrer est `F(b) - F(a)` avec `F(b)` et `F(a)`
ecrits, et non leur difference deja faite. D'ou `evaluations()`, qui rend
le couple avant toute soustraction.

L'orientation des bornes est verifiee a la construction. Inverser
silencieusement `a` et `b` changerait le signe du resultat sans que rien
le signale -- et le cours definit l'integrale sur `[a, b]` avec `a < b`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.symboles import VARIABLE


@dataclass(frozen=True, slots=True)
class IntegraleDefinie:
    """Un integrande et ses deux bornes, dans le bon ordre."""

    integrande: sp.Expr
    borne_basse: sp.Expr
    borne_haute: sp.Expr

    def __post_init__(self) -> None:
        """Refuse des bornes inversees : le signe changerait en silence."""
        ecart = sp.simplify(self.borne_haute - self.borne_basse)
        if ecart.is_number and ecart <= 0:
            msg = (
                f"borne basse superieure ou egale a la borne haute : "
                f"{self.borne_basse} >= {self.borne_haute}"
            )
            raise ValueError(msg)

    def primitive(self) -> sp.Expr:
        """La primitive `F`, celle que le corrige affiche avant d'evaluer."""
        return Primitive.de(self.integrande).expression

    def evaluations(self) -> tuple[sp.Expr, sp.Expr]:
        """Le couple `(F(b), F(a))`, avant toute soustraction."""
        primitive = self.primitive()
        return (
            sp.simplify(primitive.subs(VARIABLE, self.borne_haute)),
            sp.simplify(primitive.subs(VARIABLE, self.borne_basse)),
        )

    def valeur(self) -> sp.Expr:
        """`F(b) - F(a)`."""
        en_haut, en_bas = self.evaluations()
        return sp.simplify(en_haut - en_bas)
