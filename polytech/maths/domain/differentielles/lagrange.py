"""L'equation de Lagrange, `y = x psi(y') + phi(y')`.

Avec `z = y'` et une derivation par rapport a `x`,
`z - psi(z) = (x psi'(z) + phi'(z)) dz/dx` : en prenant `x` pour fonction
de `z`, c'est une equation **lineaire**,

    dx/dz - x psi'(z) / (z - psi(z)) = phi'(z) / (z - psi(z)).

Chaque zero `z_0` de `z - psi(z)` donne en outre une droite
`y = x psi(z_0) + phi(z_0)`. Elle est **singuliere** si aucune valeur de
la constante ne la fait sortir de la solution generale.

Clairaut est le cas `psi(z) = z` : `z - psi(z)` y est nul partout.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.differentielles.clairaut import PENTE
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE


@dataclass(frozen=True, slots=True)
class EquationDeLagrange:
    """`psi` et `phi`, ecrites en la pente `z`."""

    psi: sp.Expr
    phi: sp.Expr

    def __post_init__(self) -> None:
        """Refuse `psi(z) = z`."""
        if sp.simplify(PENTE - self.psi) == 0:
            msg = "psi(z) = z : c'est une equation de Clairaut, sans equation en x"
            raise ValueError(msg)

    def equation_en_x(self) -> tuple[sp.Expr, sp.Expr]:
        """`(P, Q)` de `dx/dz + P x = Q`."""
        ecart = PENTE - self.psi
        return (
            sp.simplify(-sp.diff(self.psi, PENTE) / ecart),
            sp.simplify(sp.diff(self.phi, PENTE) / ecart),
        )

    def x_de_z(self, constante: object) -> sp.Expr:
        """`x = e^(-int P) (C + int Q e^(int P) dz)`."""
        p, q = self.equation_en_x()
        facteur = sp.simplify(sp.exp(sp.integrate(p, PENTE)))
        primitive = sp.integrate(sp.simplify(q * facteur), PENTE)
        return sp.simplify((sp.sympify(constante) + primitive) / facteur)

    def pentes_constantes(self) -> tuple[sp.Expr, ...]:
        """Les zeros de `z - psi(z)`."""
        racines = sp.solve(sp.Eq(PENTE - self.psi, 0), PENTE)
        return tuple(sorted(racines, key=sp.default_sort_key))

    def solutions_lineaires(self) -> tuple[sp.Expr, ...]:
        """`y = x psi(z_0) + phi(z_0)`, pour chaque pente constante."""
        return tuple(
            sp.expand(VARIABLE * self.psi.subs(PENTE, z0) + self.phi.subs(PENTE, z0))
            for z0 in self.pentes_constantes()
        )


def est_singuliere(solution: sp.Expr, generale: sp.Expr, constante: sp.Symbol) -> bool:
    """Vrai si aucune valeur **constante** ne tire `solution` de la generale."""
    candidates = sp.solve(sp.Eq(generale, solution), constante)
    libres = constante.free_symbols | {constante}
    return all(valeur.free_symbols - libres for valeur in candidates)
