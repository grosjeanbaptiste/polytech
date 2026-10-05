"""Les potentiels des etats variables, leur jauge et leur retard.

Le syllabus d'electricite generale (14.2 a 14.10 et 14.18). L'induction
reste `B = rot A` (14.2.3). Toute variation de `A` fait naitre un champ
induit statique `E_i = -dA/dt` (14.3.5), et le champ du aux charges
s'ecrit `E_s + E_i = -grad V - dA/dt` (14.5.3) : la loi de Faraday
`rot (E_s + E_i) = -dB/dt` (14.5.2) est alors satisfaite d'office.

Les potentiels ne sont pas determines : `A + grad(phi)` (14.8.1) et
`V - d(phi)/dt` (14.8.2) donnent le meme champ et la meme induction.
Maxwell leve l'arbitraire par `div A = 0` (14.10.4), Lorenz et Lorentz
par `div A + (1 / c_0^2) dV/dt = 0` (14.18.4) ; les potentiels de ces
derniers sont retardes du temps que met une perturbation a venir du
point potentiant, a la celerite `c_0`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    changement_de_jauge,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS

if TYPE_CHECKING:
    Vecteur = tuple[Any, Any, Any]


def champ_induit_statique(vecteur: Vecteur) -> Vecteur:
    """`E_i = -dA/dt` (14.3.5)."""
    x, y, z = (-sp.diff(sp.sympify(composante), TEMPS) for composante in vecteur)
    return (x, y, z)


def champ_des_potentiels(scalaire: Any, vecteur: Vecteur) -> Vecteur:  # noqa: ANN401 -- reel ou symbole
    """`E_s + E_i = -grad V - dA/dt` (14.5.3)."""
    x, y, z = (
        induit - sp.diff(sp.sympify(scalaire), axe)
        for induit, axe in zip(
            champ_induit_statique(vecteur), CARTESIENNES, strict=True
        )
    )
    return (x, y, z)


def changer_de_jauge(
    scalaire: sp.Expr, vecteur: Vecteur, fonction: sp.Expr
) -> tuple[sp.Expr, Vecteur]:
    """`(V - d(phi)/dt, A + grad(phi))` : (14.8.2) et (14.8.1)."""
    autre = changement_de_jauge(vecteur, fonction)
    return scalaire - sp.diff(fonction, TEMPS), autre


def residu_de_la_jauge_de_lorenz(
    scalaire: object, vecteur: Vecteur, celerite: sp.Expr
) -> sp.Expr:
    """`div A + (1 / c_0^2) dV/dt` (14.18.4) : nul en jauge de Lorenz."""
    derivee = sp.diff(sp.sympify(scalaire), TEMPS) / celerite**2
    champ = tuple(sp.sympify(composante) for composante in vecteur)
    return sp.simplify(divergence_cartesienne(champ) + derivee)


def potentiel_retarde(charge: sp.Expr, distance: sp.Expr, celerite: Any) -> sp.Expr:  # noqa: ANN401 -- reel ou symbole
    """`q(t - r / c_0) / (4 pi eps_0 r)` : (14.18.2a) pour une charge ponctuelle."""
    if celerite == 0:
        msg = "celerite nulle : la perturbation n'arriverait jamais"
        raise ValueError(msg)
    return COULOMB * charge.subs(TEMPS, TEMPS - distance / celerite) / distance
