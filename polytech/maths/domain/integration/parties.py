"""L'integration par parties, exposee comme un geste et non comme un calcul.

Ce que le lecteur doit voir est la ligne `u v - integrale de v du`, avec
`u` et `v` nommes. `etape()` la rend telle quelle, l'integrale restante non
evaluee : c'est l'etat intermediaire du raisonnement.

`resultat()`, lui, conclut : il rend la primitive que sympy calcule, sans
montrer d'etape. La reponse 3 du fascicule, `(x^3 - 3x^2 + 6x - 6) e^x`,
demande trois parties successives : c'est `parties_repetees.jusqu_au_bout`
qui les itere et les expose une a une.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


@dataclass(frozen=True, slots=True)
class IntegrationParParties:
    """Le choix de `u` et de `dv`, dont tout le reste decoule."""

    u: sp.Expr
    derivee_de_v: sp.Expr

    def v(self) -> sp.Expr:
        """La primitive de `dv`, c'est-a-dire `v` lui-meme."""
        return sp.integrate(self.derivee_de_v, VARIABLE)

    def derivee_de_u(self) -> sp.Expr:
        """`du`, qui remplace `u` dans l'integrale restante."""
        return sp.diff(self.u, VARIABLE)

    def integrale_restante(self) -> sp.Integral:
        """L'integrale de `v du`, laissee non evaluee a dessein."""
        return sp.Integral(self.v() * self.derivee_de_u(), VARIABLE)

    def etape(self) -> sp.Expr:
        """`u v - integrale de v du` : l'etat intermediaire a montrer."""
        return self.u * self.v() - self.integrale_restante()

    def resultat(self) -> sp.Expr:
        """La primitive, calculee d'un bloc (les etapes : `jusqu_au_bout`)."""
        return sp.simplify(sp.integrate(self.u * self.derivee_de_v, VARIABLE))
