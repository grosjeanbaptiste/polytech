"""La boucle de mesure plongee dans le champ d'un fil rectiligne infini.

Le fil porte un courant variable ; le champ qu'il cree decroit en `1/x`.
Le flux a travers la boucle s'obtient donc en integrant ce `1/x` entre la
distance du fil et le bord oppose -- d'ou le logarithme, et d'ou la forme
`log((a+b)/a)` qui revient dans tout le corrige.

La loi de Faraday fait ensuite le reste : la boucle n'ayant aucune
resistance, la difference entre la tension de la source et celle que lit le
voltmetre est exactement la derivee du flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

ABSCISSE = sp.Symbol("x", positive=True)
"""La variable d'integration : la distance au fil."""


@dataclass(frozen=True, slots=True)
class BoucleDeMesure:
    """Une boucle rectangulaire, a `distance` du fil, de `largeur` et `hauteur`."""

    hauteur: sp.Expr
    distance: sp.Expr
    largeur: sp.Expr

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.distance == 0:
            msg = "boucle collee au fil : le flux diverge en son bord"
            raise ValueError(msg)
        if self.largeur == 0 or self.hauteur == 0:
            msg = f"boucle d'aire nulle : {self.largeur} x {self.hauteur}"
            raise ValueError(msg)

    def flux(
        self,
        permeabilite: sp.Expr,
        courant_crete: sp.Expr,
        pulsation: sp.Expr,
        instant: sp.Expr,
    ) -> sp.Expr:
        """Le flux magnetique a travers la boucle, integre entre `a` et `a+b`."""
        champ = (
            permeabilite
            * courant_crete
            * sp.sin(pulsation * instant)
            / (2 * sp.pi * ABSCISSE)
        )
        return self.hauteur * sp.integrate(
            champ,
            (ABSCISSE, self.distance, self.distance + self.largeur),
        )

    def tension_mesuree(
        self,
        tension_de_la_source: sp.Expr,
        permeabilite: sp.Expr,
        courant_crete: sp.Expr,
        pulsation: sp.Expr,
        instant: sp.Expr,
    ) -> sp.Expr:
        """`V_m = V + d(Phi)/dt`, les fils de la boucle etant sans resistance.

        Faraday donne `V - V_m = -d(Phi)/dt` le long du contour ; le signe
        suit l'orientation choisie sur le schema du corrige.
        """
        flux = self.flux(permeabilite, courant_crete, pulsation, instant)
        return tension_de_la_source + sp.diff(flux, instant)


@dataclass(frozen=True, slots=True)
class BoucleRectangulaire:
    """Une boucle plongee dans un champ uniforme et variable dans le temps.

    Le champ ne depend plus de la position : le flux est simplement le
    produit de l'aire par le champ, et la loi de Faraday se reduit a une
    derivee temporelle.
    """

    longueur: sp.Expr
    largeur: sp.Expr

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.longueur == 0 or self.largeur == 0:
            msg = f"boucle d'aire nulle : {self.longueur} x {self.largeur}"
            raise ValueError(msg)

    def flux(
        self,
        champ_crete: sp.Expr,
        pulsation: sp.Expr,
        instant: sp.Expr,
    ) -> sp.Expr:
        """`a b B_0 sin(omega t)` : l'aire multiplie le champ, rien de plus."""
        return self.longueur * self.largeur * champ_crete * sp.sin(pulsation * instant)

    def tension_induite(
        self,
        champ_crete: sp.Expr,
        pulsation: sp.Expr,
        instant: sp.Expr,
    ) -> sp.Expr:
        """`-d(Phi)/dt`, la tension que la loi de Faraday impose au contour."""
        return -sp.diff(self.flux(champ_crete, pulsation, instant), instant)
