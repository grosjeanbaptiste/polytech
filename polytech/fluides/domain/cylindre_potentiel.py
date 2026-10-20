"""L'ecoulement potentiel autour d'un cylindre : un doublet dans un ecoulement uniforme.

Le cours ne derive pas ce champ. Cleynen, *Fluid Dynamics for Engineers*,
chapitre 11 (`sources/externes/CleynenFluidDynamics/11/chap11.tex`), le
construit par superposition : un ecoulement uniforme `psi = U r sin(theta)`
et un doublet `psi = -K sin(theta) / r`, avec `K = U R^2` pour que le
cercle `r = R` soit une ligne de courant. D'ou

`psi = U sin(theta) (r - R^2 / r)`.

Les vitesses en derivent, `v_r = (1/r) d psi / d theta` et
`v_theta = -d psi / d r`. L'ecoulement etant permanent, incompressible et
non visqueux, Bernoulli tient le long de la ligne de courant de paroi :
`p - p_inf = rho (U^2 - v^2) / 2`, donc `C_p = 1 - (v / U)^2`.

Les angles sont ceux de Cleynen : l'ecoulement va vers les `x` croissants,
`theta = 0` est le point d'arret **aval** et `theta = pi` le point d'arret
amont. `C_p` ne depend que de `sin^2(theta)` : il est le meme, que l'on
compte l'angle depuis l'amont ou depuis l'aval.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

ANGLE = sp.Symbol("theta", real=True)
"""L'angle polaire, compte depuis l'axe aval (`theta = 0` en aval)."""


def fonction_de_courant(
    rayon: Grandeur, angle: Grandeur, vitesse: Grandeur, rayon_du_cylindre: Grandeur
) -> sp.Expr:
    """`psi = U sin(theta) (r - R^2 / r)`."""
    r, theta, u, a = map(sp.sympify, (rayon, angle, vitesse, rayon_du_cylindre))
    return u * sp.sin(theta) * (r - a**2 / r)


def potentiel_des_vitesses(
    rayon: Grandeur, angle: Grandeur, vitesse: Grandeur, rayon_du_cylindre: Grandeur
) -> sp.Expr:
    """`phi = U cos(theta) (r + R^2 / r)` : uniforme plus doublet (Cleynen)."""
    r, theta, u, a = map(sp.sympify, (rayon, angle, vitesse, rayon_du_cylindre))
    return u * sp.cos(theta) * (r + a**2 / r)


def _derivee(
    variable: str, rayon: Grandeur, angle: Grandeur, u: Grandeur, a: Grandeur
) -> sp.Expr:
    """Derive `psi` en `r` ou en `theta`, puis evalue au point demande."""
    r, theta = sp.symbols("r theta")
    psi = fonction_de_courant(r, theta, u, a)
    derivee = sp.diff(psi, r if variable == "r" else theta)
    return derivee.subs({r: sp.sympify(rayon), theta: sp.sympify(angle)})


def vitesse_radiale(
    rayon: Grandeur, angle: Grandeur, vitesse: Grandeur, rayon_du_cylindre: Grandeur
) -> sp.Expr:
    """`v_r = (1 / r) d psi / d theta = U cos(theta) (1 - R^2 / r^2)`."""
    derivee = _derivee("theta", rayon, angle, vitesse, rayon_du_cylindre)
    return sp.simplify(derivee / sp.sympify(rayon))


def vitesse_tangentielle(
    rayon: Grandeur, angle: Grandeur, vitesse: Grandeur, rayon_du_cylindre: Grandeur
) -> sp.Expr:
    """`v_theta = -d psi / d r = -U sin(theta) (1 + R^2 / r^2)`."""
    return sp.simplify(-_derivee("r", rayon, angle, vitesse, rayon_du_cylindre))


def coefficient_de_pression(
    rayon: Grandeur, angle: Grandeur, rayon_du_cylindre: Grandeur
) -> sp.Expr:
    """`C_p = 1 - (v / U)^2` en tout point : il ne depend ni de `U` ni de `rho`."""
    un = sp.Integer(1)
    radiale = vitesse_radiale(rayon, angle, un, rayon_du_cylindre)
    tangentielle = vitesse_tangentielle(rayon, angle, un, rayon_du_cylindre)
    return sp.simplify(1 - radiale**2 - tangentielle**2)


def coefficient_de_pression_parietal(angle: Grandeur) -> sp.Expr:
    """`C_p = 1 - 4 sin^2(theta)` sur la paroi."""
    return sp.simplify(coefficient_de_pression(1, angle, 1))


def points_d_arret() -> tuple[sp.Expr, sp.Expr]:
    """Les angles ou la vitesse parietale s'annule : `sin(theta) = 0`.

    Dans l'ordre `(0, pi)` : l'aval, puis l'amont.
    """
    theta = sp.Symbol("theta", real=True)
    racines = sp.solveset(sp.sin(theta), theta, sp.Interval.Ropen(0, 2 * sp.pi))
    aval, amont = sorted(racines, key=float)
    return aval, amont


def efforts_de_pression_reduits(
    distribution: sp.Expr | None = None,
) -> tuple[sp.Expr, sp.Expr]:
    """Trainee et portance par unite de longueur, rapportees a `R rho U^2 / 2`.

    `F_x = -int p cos(theta) R d theta`, `F_y = -int p sin(theta) R d theta`
    (Cleynen) ; `p_inf` s'elimine sur un tour complet. La distribution est
    un `C_p` en `ANGLE`, celle du modele potentiel par defaut.
    """
    theta = ANGLE
    cp = (
        coefficient_de_pression_parietal(theta)
        if distribution is None
        else distribution
    )
    tour = (theta, 0, 2 * sp.pi)
    return (
        sp.simplify(-sp.integrate(cp * sp.cos(theta), tour)),
        sp.simplify(-sp.integrate(cp * sp.sin(theta), tour)),
    )
