"""Les equations de Prandtl, qui gouvernent la couche limite.

Planches du chapitre IX. Pres d'une paroi, la viscosite compte autant que
l'inertie, si grand que soit le nombre de Reynolds : egaler l'advection
`U^2 / L` a la diffusion `nu U / delta^2` donne `delta / L` en
`1 / racine(Re)`.

Adimensionnees avec cette echelle -- `y* = (y / L) racine(Re)`,
`v* = (v / U) racine(Re)` -- les equations de Navier-Stokes ne gardent,
quand `Re` tend vers l'infini, que

`u u_x + v u_y = -(1 / rho) dp/dx + nu u_yy`, `dp/dy = 0`, `u_x + v_y = 0`.

La pression traverse la couche limite sans changer : c'est celle de
l'ecoulement exterieur, que Bernoulli donne, `dp_e/dx = -rho U_e U_e'`.
A la paroi, ou `u = v = 0`, il reste `nu u_yy = (1 / rho) dp/dx` : le
gradient de pression fixe la courbure du profil, et un gradient adverse
le creuse jusqu'au decollement.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import X, Y

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def epaisseur_relative_de_couche_limite(reynolds: float) -> float:
    """`delta / L`, en ordre de grandeur : `1 / racine(Re)`."""
    if reynolds <= 0:
        msg = f"Reynolds nul ou negatif : {reynolds}"
        raise ValueError(msg)
    return float(reynolds**-0.5)


@dataclass(frozen=True, slots=True)
class Echelles:
    """La vitesse `U`, la longueur `L` du corps et la viscosite cinematique."""

    vitesse: Grandeur
    longueur: Grandeur
    viscosite: Grandeur

    def reynolds(self) -> Grandeur:
        """`Re = U L / nu`."""
        return self.vitesse * self.longueur / self.viscosite

    def _etoile(self, reduit: sp.Expr) -> sp.Expr:
        etire = Y * sp.sqrt(self.reynolds()) / self.longueur
        return reduit.subs({X: X / self.longueur, Y: etire}, simultaneous=True)

    def longitudinale(self, reduit: sp.Expr) -> sp.Expr:
        """`u = U u*(x / L, (y / L) racine(Re))`."""
        return self.vitesse * self._etoile(reduit)

    def transversale(self, reduit: sp.Expr) -> sp.Expr:
        """`v = (U / racine(Re)) v*(x / L, (y / L) racine(Re))`."""
        return self.vitesse * self._etoile(reduit) / sp.sqrt(self.reynolds())


def gradient_de_pression_exterieur(
    masse_volumique: Grandeur, exterieure: Grandeur
) -> sp.Expr:
    """`dp_e/dx = -rho U_e dU_e/dx`, par Bernoulli hors de la couche limite."""
    vitesse = sp.sympify(exterieure)
    return -masse_volumique * vitesse * sp.diff(vitesse, X)


def residu_de_prandtl(
    u: Grandeur, v: Grandeur, exterieure: Grandeur, viscosite: Grandeur
) -> sp.Expr:
    """`u u_x + v u_y - U_e U_e' - nu u_yy` : nul pour une couche limite."""
    le_long, libre = sp.sympify(u), sp.sympify(exterieure)
    inertie = le_long * sp.diff(le_long, X) + v * sp.diff(le_long, Y)
    pression = libre * sp.diff(libre, X)
    return sp.simplify(inertie - pression - viscosite * sp.diff(le_long, Y, 2))


def courbure_a_la_paroi(exterieure: Grandeur, viscosite: Grandeur) -> sp.Expr:
    """`u_yy` en `y = 0` : `-U_e U_e' / nu`."""
    libre = sp.sympify(exterieure)
    return -libre * sp.diff(libre, X) / viscosite


def nature_du_gradient_de_pression(gradient: float) -> str:
    """Favorable s'il accelere l'ecoulement, adverse s'il le freine."""
    if gradient < 0:
        return "favorable"
    return "adverse" if gradient > 0 else "nul"
