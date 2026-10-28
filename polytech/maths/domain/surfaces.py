"""Flux et circulation : le theoreme de Stokes sur une surface `z = h(x, y)`.

Pour la surface `z = h(x, y)`, le vecteur `(-h_x, -h_y, 1)` est normal et
son produit par `dx dy` est `n dS` : sa norme est l'element d'aire
`racine(1 + h_x^2 + h_y^2)`. Le flux de `rot F` a travers la surface est
la circulation de `F` le long de son contour (Stokes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.analyse_vectorielle import rotationnel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.analyse_vectorielle import Vecteur

_R, _THETA = sp.symbols("r theta", nonnegative=True)


@dataclass(frozen=True, slots=True)
class SurfaceEnGraphe:
    """La surface `z = hauteur(x, y)`."""

    hauteur: sp.Expr
    x: sp.Symbol
    y: sp.Symbol

    def normale(self) -> Vecteur:
        """`(-h_x, -h_y, 1)`, non unitaire."""
        return (
            -sp.diff(self.hauteur, self.x),
            -sp.diff(self.hauteur, self.y),
            sp.Integer(1),
        )

    def element_d_aire(self) -> sp.Expr:
        """La norme de la normale : `dS = element dx dy`."""
        return sp.sqrt(sum(c**2 for c in self.normale()))

    def normale_unitaire(self) -> Vecteur:
        """La normale divisee par sa norme."""
        n = self.element_d_aire()
        a, b, c = (sp.simplify(k / n) for k in self.normale())
        return (a, b, c)

    def flux_du_rotationnel(
        self, champ: Sequence[sp.Expr], coordonnees: Sequence[sp.Symbol], rayon: float
    ) -> sp.Expr:
        """Flux de `rot F` sur la surface au-dessus du disque de ce rayon."""
        if rayon <= 0:
            msg = f"rayon non positif : {rayon}"
            raise ValueError(msg)
        tourbillon = rotationnel(champ, coordonnees)
        sur_la_surface = {coordonnees[2]: self.hauteur}
        integrande = sum(
            sp.sympify(a).subs(sur_la_surface) * b
            for a, b in zip(tourbillon, self.normale(), strict=True)
        )
        polaire = sp.simplify(
            integrande.subs({self.x: _R * sp.cos(_THETA), self.y: _R * sp.sin(_THETA)})
        )
        return sp.integrate(polaire * _R, (_R, 0, rayon), (_THETA, 0, 2 * sp.pi))

    def contour_du_disque(self, parametre: sp.Symbol, rayon: float) -> Vecteur:
        """`(r cos t, r sin t, h)` : le bord de la surface au-dessus du disque."""
        x, y = rayon * sp.cos(parametre), rayon * sp.sin(parametre)
        return (x, y, self.hauteur.subs({self.x: x, self.y: y}))


def circulation(
    champ: Sequence[sp.Expr],
    coordonnees: Sequence[sp.Symbol],
    chemin: Sequence[sp.Expr],
    parametre: sp.Symbol,
    bornes: tuple[sp.Expr, sp.Expr],
) -> sp.Expr:
    """`int F . dr` le long du chemin parametre : le second chemin de Stokes."""
    sur_le_chemin = dict(zip(coordonnees, chemin, strict=True))
    travail = sum(
        sp.sympify(f).subs(sur_le_chemin) * sp.diff(c, parametre)
        for f, c in zip(champ, chemin, strict=True)
    )
    return sp.integrate(travail, (parametre, *bornes))
