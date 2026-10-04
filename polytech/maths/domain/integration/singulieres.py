"""L'integrale impropre d'une fonction non definie en une borne (planche 40).

`ln x` n'est pas definie en `x = 0` : on integre de `delta` a 1, puis on
fait tendre `delta` vers 0.

    int_0^1 ln x dx = lim [x ln x - x]_delta^1 = -1.

C'est l'autre espece d'integrale impropre ; `IntegraleImpropre` porte
celle dont une borne est infinie. La borne mobile approche la singularite
du cote de l'intervalle d'integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.symboles import VARIABLE

BORNE_MOBILE: Final = sp.Symbol("delta", real=True)
"""La borne que l'on fait tendre vers la singularite."""


@dataclass(frozen=True, slots=True)
class IntegraleSinguliere:
    """Un integrande, la borne ou il n'est pas defini, et l'autre borne."""

    integrande: sp.Expr
    singularite: sp.Expr
    autre_borne: sp.Expr

    def borne_mobile(self) -> sp.Symbol:
        """Le symbole que le corrige fait tendre vers la singularite."""
        return BORNE_MOBILE

    def _singularite_en_bas(self) -> bool:
        return bool(sp.sympify(self.singularite) < sp.sympify(self.autre_borne))

    def primitive_bornee(self) -> sp.Expr:
        """`F(b) - F(a)`, la borne singuliere remplacee par la borne mobile."""
        primitive = Primitive.de(self.integrande).expression
        fixe = primitive.subs(VARIABLE, self.autre_borne)
        mobile = primitive.subs(VARIABLE, BORNE_MOBILE)
        difference = fixe - mobile if self._singularite_en_bas() else mobile - fixe
        return sp.simplify(difference)

    def limite(self) -> sp.Expr:
        """La limite quand la borne mobile rejoint la singularite, de l'interieur."""
        cote = "+" if self._singularite_en_bas() else "-"
        return sp.limit(self.primitive_bornee(), BORNE_MOBILE, self.singularite, cote)

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
