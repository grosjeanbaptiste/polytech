"""Le changement de variable, avec la variable nouvelle et sa differentielle.

Le fascicule du cours ne donne jamais autre chose que cela : « changement
de variable u = sin x », « u = e^x », « t = tan(x/2) ». Le geste est donc
l'expression substituee, et ce que le lecteur doit voir est le triplet
`u = g(x)`, `du = g'(x) dx`, et l'integrale reecrite en `u`.

La garde qui refuse une substitution laissant subsister `x` n'est pas
formelle : une integrale en `u` ou traine encore un `x` est inutilisable,
et le corrige l'afficherait sans que rien ne signale la faute.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

NOUVELLE = sp.Symbol("u", real=True)
"""La variable d'arrivee. Le cours l'ecrit `u`, parfois `t`."""


@dataclass(frozen=True, slots=True)
class Substitution:
    """L'integrande, et l'expression que l'on baptise variable nouvelle."""

    integrande: sp.Expr
    expression: sp.Expr

    def nouvelle_variable(self) -> sp.Symbol:
        """Le symbole d'arrivee, celui que le corrige nomme."""
        return NOUVELLE

    def differentielle(self) -> sp.Expr:
        """`g'(x)`, le facteur que l'integrande doit contenir."""
        return sp.diff(self.expression, VARIABLE)

    def integrale_transformee(self) -> sp.Expr:
        """Reecrit l'integrande en `u` ; leve si `x` y subsiste."""
        quotient = sp.simplify(self.integrande / self.differentielle())
        reecrit = sp.simplify(quotient.subs(self.expression, NOUVELLE))
        if reecrit.has(VARIABLE):
            msg = (
                f"le changement de variable ne fait pas disparaitre "
                f"{VARIABLE.name} : {reecrit}"
            )
            raise ValueError(msg)
        return reecrit

    def resultat(self) -> sp.Expr:
        """La primitive en `x`, obtenue en integrant puis en revenant."""
        primitive = sp.integrate(self.integrale_transformee(), NOUVELLE)
        return sp.simplify(primitive.subs(NOUVELLE, self.expression))
