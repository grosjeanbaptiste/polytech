"""La covariance de deux variables aleatoires, et ce qu'elle mesure.

`Cov[X, Y] = E[(X - E[X]) (Y - E[Y])*]` mesure la variabilite conjointe.
Sa lecture la plus directe : parmi toutes les droites `beta_1 X + beta_0`,
celle qui approche le mieux `Y` en moyenne quadratique a pour pente
`Cov[X, Y] / Var[X]` et passe par le point des esperances.

La matrice de covariance etend la notion a un vecteur aleatoire, et se
transforme par congruence : si `Y = A X + b`, alors `K_YY = A K_XX A^H`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class LoiConjointe:
    """Une loi discrete de deux variables : des couples et leurs probabilites."""

    couples: tuple[tuple[sp.Expr, sp.Expr], ...]
    probabilites: tuple[sp.Expr, ...]

    def __post_init__(self) -> None:
        """Refuse des probabilites qui ne forment pas une loi."""
        if len(self.couples) != len(self.probabilites):
            msg = "autant de probabilites que de couples sont attendues"
            raise ValueError(msg)
        if any(p < 0 for p in self.probabilites):
            msg = f"probabilite negative : {self.probabilites}"
            raise ValueError(msg)
        if sp.nsimplify(sum(self.probabilites)) != 1:
            msg = f"les probabilites ne somment pas a un : {self.probabilites}"
            raise ValueError(msg)

    def _esperance(self, rang_x: int, rang_y: int) -> sp.Expr:
        """`E[X^i Y^j]`."""
        termes = (
            p * x**rang_x * y**rang_y
            for (x, y), p in zip(self.couples, self.probabilites, strict=True)
        )
        return sp.nsimplify(sum(termes, sp.Integer(0)))

    def esperances(self) -> tuple[sp.Expr, sp.Expr]:
        """`(E[X], E[Y])`."""
        return self._esperance(1, 0), self._esperance(0, 1)

    def variances(self) -> tuple[sp.Expr, sp.Expr]:
        """`(Var[X], Var[Y])`."""
        en_x, en_y = self.esperances()
        return self._esperance(2, 0) - en_x**2, self._esperance(0, 2) - en_y**2

    def covariance(self) -> sp.Expr:
        """`Cov[X, Y] = E[X Y] - E[X] E[Y]`, pour des variables reelles."""
        en_x, en_y = self.esperances()
        return self._esperance(1, 1) - en_x * en_y

    def meilleure_droite(self) -> tuple[sp.Expr, sp.Expr]:
        """`(beta_0', beta_1')` minimisant `E[(Y - (beta_1 X + beta_0))^2]`."""
        variance = self.variances()[0]
        if variance == 0:
            msg = "variance de X nulle : aucune droite n'est meilleure qu'une autre"
            raise ValueError(msg)
        pente = self.covariance() / variance
        en_x, en_y = self.esperances()
        return en_y - pente * en_x, pente

    def matrice_de_covariance(self) -> sp.Matrix:
        """`K = [[Var X, Cov], [Cov, Var Y]]`."""
        en_x, en_y = self.variances()
        croisee = self.covariance()
        return sp.Matrix([[en_x, croisee], [croisee, en_y]])


def covariance_apres_transformation(
    transformation: sp.Matrix, covariance: sp.Matrix
) -> sp.Matrix:
    """`K_YY = A K_XX A^H`, pour `Y = A X + b`."""
    if transformation.cols != covariance.rows or not covariance.is_square:
        msg = "dimensions incompatibles entre la transformation et la covariance"
        raise ValueError(msg)
    return sp.simplify(transformation * covariance * transformation.H)
