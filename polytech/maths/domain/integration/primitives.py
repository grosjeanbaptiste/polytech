"""Une primitive, et le seul controle qui ne depende d'aucune table.

Deriver la reponse. Une table de primitives peut etre fausse, une
convention de signe peut avoir glisse, un facteur peut manquer : la
derivation tranche sans rien supposer. La bibliotheque la fait donc
systematiquement, et le porte dans un champ que les tests lisent.

La garde qui refuse une `Integral` non evaluee vaut d'etre nommee. Sans
elle, le document composerait « int(...) dx » a la place d'une valeur, et
aucun test ne tomberait -- c'est bien du LaTeX, c'est juste la mauvaise
reponse.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import CONSTANTE, VARIABLE


@dataclass(frozen=True, slots=True)
class Primitive:
    """Une primitive calculee, et son integrande d'origine."""

    integrande: sp.Expr
    expression: sp.Expr

    @classmethod
    def de(cls, integrande: sp.Expr, variable: sp.Symbol = VARIABLE) -> Primitive:
        """Calcule la primitive ; leve si elle n'est pas elementaire."""
        calculee = sp.integrate(integrande, variable)
        if calculee.has(sp.Integral):
            msg = f"primitive non elementaire : {sp.srepr(integrande)[:80]}"
            raise ValueError(msg)
        return cls(integrande=integrande, expression=sp.simplify(calculee))

    @property
    def verifiee(self) -> bool:
        """Vrai si deriver la primitive redonne bien l'integrande."""
        ecart = sp.simplify(sp.diff(self.expression, VARIABLE) - self.integrande)
        return bool(ecart == 0)

    def avec_constante(self) -> sp.Expr:
        """La primitive suivie du `+ C` que le cours ecrit explicitement."""
        return self.expression + CONSTANTE
