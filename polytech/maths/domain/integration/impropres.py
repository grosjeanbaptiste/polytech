"""L'integrale impropre, definie par la limite que le cours ecrit.

`int_a^oo f = lim_{b -> oo} int_a^b f`. La limite est exposee, parce que
c'est elle qui definit l'objet : rendre directement une valeur escamoterait
la seule chose que le chapitre enseigne.

Refuser la valeur d'une integrale divergente n'est pas de la pedanterie.
Sympy rend `oo`, qui est un objet parfaitement compilable : le document
afficherait un infini comme s'il etait un resultat, et aucun test ne
tomberait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.symboles import VARIABLE

BORNE_MOBILE: Final = sp.Symbol("b", positive=True)
"""La borne que l'on fait tendre vers l'infini."""


@dataclass(frozen=True, slots=True)
class IntegraleImpropre:
    """Une integrale de `a` a l'infini, et la limite qui la definit."""

    integrande: sp.Expr
    borne_basse: sp.Expr

    def borne_mobile(self) -> sp.Symbol:
        """Le symbole que le corrige fait tendre vers l'infini."""
        return BORNE_MOBILE

    def primitive_bornee(self) -> sp.Expr:
        """`F(b) - F(a)`, avant le passage a la limite."""
        primitive = Primitive.de(self.integrande).expression
        return sp.simplify(
            primitive.subs(VARIABLE, BORNE_MOBILE)
            - primitive.subs(VARIABLE, self.borne_basse),
        )

    def limite(self) -> sp.Expr:
        """La limite de `F(b) - F(a)` quand `b` tend vers l'infini."""
        return sp.limit(self.primitive_bornee(), BORNE_MOBILE, sp.oo)

    def converge(self) -> bool:
        """Vrai si cette limite est finie."""
        return bool(self.limite().is_finite)

    def valeur(self) -> sp.Expr:
        """La valeur de l'integrale ; leve si elle diverge."""
        limite = self.limite()
        if not limite.is_finite:
            msg = f"l'integrale diverge : la limite vaut {limite}"
            raise ValueError(msg)
        return limite
