"""Un terme de la serie de la plaque, verifie contre ses equations (4.3) a (4.5).

Le terme est `exp(-delta^2 a tau / l^2) cos(delta x / l)`, `l` etant la
demi-epaisseur. (4.7) imprime l'exposant `-delta (a tau / l^2)` : le
terme `champ(exposant_imprime=True)` le reproduit pour montrer que sa
diffusion n'est pas nulle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.thermique.domain.conduction.plaque_variable import PlaqueEnEchelon

X, TAU, A = sp.symbols("x tau a", positive=True)


@dataclass(frozen=True, slots=True)
class TermeDeLaSerie:
    """Le terme de rang `rang` de la plaque, de demi-epaisseur donnee."""

    plaque: PlaqueEnEchelon
    rang: int
    demi_epaisseur: sp.Expr

    def champ(self, *, exposant_imprime: bool = False) -> sp.Expr:
        """Le terme, avec `delta^2` (correct) ou `delta` (imprime) a l'exposant."""
        delta = self.plaque.valeurs_propres(self.rang + 1)[self.rang]
        puissance = delta if exposant_imprime else delta**2
        decroissance = sp.exp(-puissance * A * TAU / self.demi_epaisseur**2)
        return decroissance * sp.cos(delta * X / self.demi_epaisseur)

    def residu_de_diffusion(self, champ: sp.Expr) -> sp.Expr:
        """`d champ / d tau - a d2 champ / dx2` (nul pour une solution)."""
        return sp.simplify(sp.diff(champ, TAU) - A * sp.diff(champ, X, 2))

    def pente_au_centre(self, champ: sp.Expr) -> sp.Expr:
        """`d champ / dx` en `x = 0` (nulle par symetrie)."""
        return sp.diff(champ, X).subs(X, 0)

    def residu_de_fourier(self, champ: sp.Expr, echange: sp.Expr) -> float:
        """`-lambda d champ / dx - K champ` en `x = l`, a `tau = 0`.

        `lambda = K l / Bi` : le nombre de Biot de la plaque fixe la conductivite.
        """
        conductivite = echange * self.demi_epaisseur / sp.nsimplify(self.plaque.biot)
        au_bord = -conductivite * sp.diff(champ, X) - echange * champ
        return float(au_bord.subs({X: self.demi_epaisseur, TAU: 0}))
