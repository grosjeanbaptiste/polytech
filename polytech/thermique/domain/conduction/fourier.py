"""La loi de Fourier et l'equation de Fourier-Kirchhoff.

Syllabus de thermique, tome 1, premiere partie, chapitre 1. La
conductibilite pure se produit dans un milieu materiel au repos, que la
theorie traite comme un milieu continu.

La loi de Fourier est un postulat, d'accord avec le second principe : la
densite de flux est proportionnelle au gradient de temperature et
s'ecoule vers les temperatures decroissantes, `phi = -lambda grad t`. A
travers une surface de normale `n`, il en passe la projection.

Le bilan d'energie d'un volume, joint a cette loi, donne l'equation de
Fourier-Kirchhoff. A conductivite uniforme,

`dt/dtau = a laplacien(t) + q / (rho c)`, avec `a = lambda / (rho c)`,

`a` etant la diffusivite thermique et `q` la chaleur generee par unite de
volume. Le gradient et le laplacien sont ceux du cours de mathematiques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.analyse_vectorielle import gradient, laplacien

if TYPE_CHECKING:
    from collections.abc import Sequence

ESPACE: Final = sp.symbols("x y z", real=True)
INSTANT: Final = sp.Symbol("tau", nonnegative=True)

type Grandeur = sp.Expr | float
type Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]


def densite_de_flux(champ: sp.Expr, conductivite: Grandeur) -> Vecteur:
    """`phi = -lambda grad t`, en W/m^2."""
    x, y, z = gradient(champ, ESPACE)
    return (-conductivite * x, -conductivite * y, -conductivite * z)


def flux_a_travers(flux: Sequence[sp.Expr], normale: Sequence[Grandeur]) -> sp.Expr:
    """`phi . n` : la densite de flux a travers une surface de normale `n`."""
    paires = zip(flux, normale, strict=True)
    return sp.Add(*(composante * cosinus for composante, cosinus in paires))


def diffusivite(conductivite: float, masse_volumique: float, chaleur: float) -> float:
    """`a = lambda / (rho c)`, en m^2/s."""
    if masse_volumique * chaleur <= 0:
        msg = "capacite volumique nulle ou negative : rho c doit etre positif"
        raise ValueError(msg)
    return conductivite / (masse_volumique * chaleur)


def residu_de_fourier_kirchhoff(
    champ: sp.Expr, diffusivite_thermique: Grandeur, *, source: Grandeur = 0
) -> sp.Expr:
    """`dt/dtau - a laplacien(t) - q / (rho c)` : nul pour un champ licite."""
    diffusion = diffusivite_thermique * laplacien(champ, ESPACE)
    return sp.simplify(sp.diff(champ, INSTANT) - diffusion - source)


def conductivite_lineaire(reference: float, pente: float, temperature: float) -> float:
    """`lambda = lambda_0 (1 + b t)`."""
    return reference * (1 + pente * temperature)


def conductivite_moyenne(
    ordonnee: Grandeur, pente: Grandeur, chaude: Grandeur, froide: Grandeur
) -> Grandeur:
    """`lambda_m = a + b (t_1 + t_2) / 2`, pour `lambda = a + b t`."""
    return ordonnee + pente * (chaude + froide) / 2
