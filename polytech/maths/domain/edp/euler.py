"""L'equation d'Euler `a u_xx + 2h u_xy + b u_yy = 0`, a coefficients constants.

Le changement de variables `xi = p x + q y`, `eta = r x + s y` la
transforme en

    (a p^2 + 2h p q + b q^2) u_xi_xi + 2 (a p r + h (q r + p s) + b q s) u_xi_eta
        + (a r^2 + 2h r s + b s^2) u_eta_eta = 0.

Avec `p = r = 1` et pour `q`, `s` les deux racines de
`a + 2h lambda + b lambda^2 = 0`, il ne reste que le terme croise :
`u = F(x + lambda_1 y) + G(x + lambda_2 y)`. Les racines sont reelles pour
le type hyperbolique, complexes conjuguees pour l'elliptique. Si la racine
est double (type parabolique), `u = F(x + lambda y) + y G(x + lambda y)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.edp.classification import Type, type_a_deux_variables
from polytech.maths.domain.edp.operateurs import ORDONNEE
from polytech.maths.domain.symboles import VARIABLE


@dataclass(frozen=True, slots=True)
class EquationDEuler:
    """Les trois coefficients constants `a`, `h`, `b`."""

    a: sp.Expr
    h: sp.Expr
    b: sp.Expr

    def __post_init__(self) -> None:
        """Refuse `b = 0` : l'equation des pentes ne serait plus du second degre."""
        if sp.sympify(self.b) == 0:
            msg = "b nul : a + 2h lambda + b lambda^2 n'a plus deux racines"
            raise ValueError(msg)

    def membre(self, fonction: sp.Expr) -> sp.Expr:
        """`a u_xx + 2h u_xy + b u_yy`."""
        return (
            self.a * sp.diff(fonction, VARIABLE, 2)
            + 2 * self.h * sp.diff(fonction, VARIABLE, ORDONNEE)
            + self.b * sp.diff(fonction, ORDONNEE, 2)
        )

    def type(self) -> Type:
        """Elliptique, parabolique ou hyperbolique."""
        return type_a_deux_variables(self.a, self.h, self.b)

    def pentes(self) -> tuple[sp.Expr, ...]:
        """Les deux racines de `a + 2h lambda + b lambda^2 = 0`, avec multiplicite."""
        inconnue = sp.Symbol("lambda")
        racines = sp.roots(
            self.a + 2 * self.h * inconnue + self.b * inconnue**2, inconnue
        )
        etalees = [
            r for r, multiplicite in racines.items() for _ in range(multiplicite)
        ]
        return tuple(sorted(etalees, key=sp.default_sort_key))

    def solution_generale(self, premiere: sp.Function, seconde: sp.Function) -> sp.Expr:
        """`F(x + l1 y) + G(x + l2 y)`, ou `F + y G` pour une racine double."""
        l1, l2 = self.pentes()
        if l1 == l2:
            argument = VARIABLE + l1 * ORDONNEE
            return premiere(argument) + ORDONNEE * seconde(argument)
        return premiere(VARIABLE + l1 * ORDONNEE) + seconde(VARIABLE + l2 * ORDONNEE)

    def coefficients_transformes(
        self, p: object, q: object, r: object, s: object
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        """Les coefficients de `u_xi_xi`, `u_xi_eta` et `u_eta_eta`."""
        a, h, b = self.a, self.h, self.b
        p_, q_, r_, s_ = (sp.sympify(valeur) for valeur in (p, q, r, s))
        return (
            a * p_**2 + 2 * h * p_ * q_ + b * q_**2,
            2 * (a * p_ * r_ + h * (q_ * r_ + p_ * s_) + b * q_ * s_),
            a * r_**2 + 2 * h * r_ * s_ + b * s_**2,
        )
