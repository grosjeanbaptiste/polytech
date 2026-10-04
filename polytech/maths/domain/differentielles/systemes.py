"""Les systemes d'equations differentielles lineaires a coefficients constants.

`y' = A y + x(t)`, `y(0) = c`, la matrice `A` etant constante (exercices,
seance III). Trois lectures du meme systeme :

* les **modes propres** : chercher `y = xi e^(rt)` revient a `A xi = r xi`,
  et la solution generale combine les modes ; le **wronskien**, determinant
  d'une matrice de solutions, dit si elles sont independantes ;
* la **transformee de Laplace** : `p Y - c = A Y + X`, d'ou
  `Y = (pI - A)^-1 (c + X)` ;
* l'**exponentielle de matrice** : `y = e^(At) c + int_0^t e^(A(t - tau))
  x(tau) d tau`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import IMAGE, TEMPS

_TAU: Final = sp.Symbol("tau", nonnegative=True)


@dataclass(frozen=True, slots=True)
class SystemeDifferentiel:
    """`y' = A y`, donne par sa matrice constante."""

    matrice: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse une matrice non carree."""
        if self.matrice.rows != self.matrice.cols:
            forme = f"{self.matrice.rows} x {self.matrice.cols}"
            msg = f"matrice carree attendue pour y' = A y, {forme} recue"
            raise ValueError(msg)

    def modes_propres(self) -> tuple[tuple[sp.Expr, sp.Matrix], ...]:
        """Les couples `(r, xi)` tels que `xi e^(rt)` est solution."""
        return tuple(
            (valeur, vecteur)
            for valeur, _, vecteurs in self.matrice.eigenvects()
            for vecteur in vecteurs
        )

    def wronskien(self, solutions: sp.Matrix) -> sp.Expr:
        """Le determinant d'une matrice dont chaque colonne est une solution."""
        ecart = sp.simplify(sp.diff(solutions, TEMPS) - self.matrice * solutions)
        if ecart != sp.zeros(*solutions.shape):
            msg = "les colonnes ne sont pas des solutions de y' = A y"
            raise ValueError(msg)
        return sp.simplify(solutions.det())

    def exponentielle(self) -> sp.Matrix:
        """`e^(At)`, sous forme reelle."""
        brute = sp.exp(self.matrice * TEMPS)
        return brute.applyfunc(lambda e: sp.simplify(sp.re(sp.expand_complex(e))))

    def par_exponentielle(
        self, conditions: sp.Matrix, force: sp.Matrix | None = None
    ) -> sp.Matrix:
        """`e^(At) c + int_0^t e^(A(t - tau)) x(tau) d tau`."""
        exponentielle = self.exponentielle()
        libre = exponentielle * conditions
        if force is None:
            return sp.simplify(libre)
        noyau = exponentielle.subs(TEMPS, TEMPS - _TAU) * force.subs(TEMPS, _TAU)
        forcee = noyau.applyfunc(lambda e: sp.integrate(e, (_TAU, 0, TEMPS)))
        return sp.simplify(libre + forcee)

    def par_laplace(
        self, conditions: sp.Matrix, force: sp.Matrix | None = None
    ) -> sp.Matrix:
        """L'original de `(pI - A)^-1 (c + X(p))`."""
        membre = conditions
        if force is not None:
            membre = conditions + force.applyfunc(
                lambda e: sp.laplace_transform(e, TEMPS, IMAGE, noconds=True)
            )
        return (self.resolvante() * membre).applyfunc(_original)

    def resolvante(self) -> sp.Matrix:
        """`(pI - A)^-1`, l'image de `e^(At)`."""
        return sp.simplify((IMAGE * sp.eye(self.matrice.rows) - self.matrice).inv())


def _original(image: sp.Expr) -> sp.Expr:
    """La transformee inverse, sans l'echelon que le cours n'ecrit pas."""
    inverse = sp.inverse_laplace_transform(
        sp.simplify(image), IMAGE, TEMPS, noconds=True
    )
    reelle = sp.re(sp.expand_complex(inverse.subs(sp.Heaviside(TEMPS), 1)))
    return sp.simplify(reelle)
