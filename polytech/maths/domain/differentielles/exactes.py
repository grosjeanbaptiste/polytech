"""Les EDO exactes, sous la forme differentielle que le cours leur donne.

Le fascicule l'ecrit : « Elle admet alors comme solution implicite
g(x, y) = c ». Une equation exacte ne se resout donc pas en `y(x)` mais en
une fonction potentielle dont les lignes de niveau sont les solutions --
et c'est ce potentiel que le corrige doit exhiber.

Une equation exacte ne se reconnait pas a son second membre `y' = f(x, y)` :
la decomposition `f = -M/N` n'est pas unique. Elle se donne donc par le
couple `(M, N)`, comme le cours la pose, et le critere de Schwarz
`dM/dy = dN/dx` la qualifie.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

VARIABLE = sp.Symbol("x", real=True)
INCONNUE = sp.Symbol("y", real=True)


@dataclass(frozen=True, slots=True)
class FormeDifferentielle:
    """`M(x, y) dx + N(x, y) dy = 0`."""

    facteur_en_x: sp.Expr
    facteur_en_y: sp.Expr

    def ecart_de_schwarz(self) -> sp.Expr:
        """`dM/dy - dN/dx` : nul si et seulement si la forme est exacte."""
        return sp.simplify(
            sp.diff(self.facteur_en_x, INCONNUE) - sp.diff(self.facteur_en_y, VARIABLE),
        )

    def est_exacte(self) -> bool:
        """Vrai si le critere de Schwarz est satisfait."""
        return bool(self.ecart_de_schwarz() == 0)

    def potentiel(self) -> sp.Expr:
        """La fonction `g` telle que `dg = M dx + N dy` ; leve si inexacte."""
        if not self.est_exacte():
            msg = (
                f"forme differentielle non exacte : dM/dy - dN/dx vaut "
                f"{self.ecart_de_schwarz()}"
            )
            raise ValueError(msg)
        partiel = sp.integrate(self.facteur_en_x, VARIABLE)
        reste = sp.simplify(self.facteur_en_y - sp.diff(partiel, INCONNUE))
        return sp.simplify(partiel + sp.integrate(reste, INCONNUE))

    def solution_implicite(self, constante: sp.Expr) -> sp.Eq:
        """`g(x, y) = c`, la forme sous laquelle le cours donne la solution."""
        return sp.Eq(self.potentiel(), constante)
