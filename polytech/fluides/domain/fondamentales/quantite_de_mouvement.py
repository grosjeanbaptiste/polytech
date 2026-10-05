"""La conservation de la quantite de mouvement : Cauchy, puis Navier-Stokes.

Chapitre IV.2 et IV.3 des planches. La variation de quantite de mouvement
d'un domaine materiel egale la somme des forces exterieures, de volume et
de surface. Sous forme locale, c'est l'equation de **Cauchy** :

`rho Dv/Dt = rho g + div(sigma)`.

Avec la loi de comportement d'un fluide newtonien et la continuite d'un
ecoulement incompressible, le terme visqueux se reduit au laplacien de la
vitesse : ce sont les equations de **Navier-Stokes**,

`rho [dv/dt + (v . grad) v] = -grad p + rho g + mu laplacien(v)`.

Les deux sont exposees comme des residus a annuler, composante par
composante.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import ESPACE, T
from polytech.maths.domain.analyse_vectorielle import gradient, laplacien

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

Vecteur = tuple[Grandeur, Grandeur, Grandeur]


def _acceleration(vitesse: Vecteur) -> list[sp.Expr]:
    """`Dv/Dt = dv/dt + (v . grad) v`."""
    composantes = [sp.sympify(c) for c in vitesse]
    return [
        sp.diff(c, T)
        + sp.Add(
            *(v * sp.diff(c, axe) for v, axe in zip(composantes, ESPACE, strict=True))
        )
        for c in composantes
    ]


def _trois(valeurs: list[sp.Expr]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    x, y, z = (sp.simplify(valeur) for valeur in valeurs)
    return (x, y, z)


def residus_de_cauchy(
    masse_volumique: Grandeur,
    vitesse: Vecteur,
    contraintes: sp.Matrix,
    pesanteur: Vecteur,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`rho Dv/Dt - rho g - div(sigma)`, ou `(div sigma)_i = d(sigma_ij)/dx_j`."""
    acceleration = _acceleration(vitesse)
    residus = []
    for i in range(3):
        surface = sp.Add(*(sp.diff(contraintes[i, j], ESPACE[j]) for j in range(3)))
        inertie = masse_volumique * (acceleration[i] - pesanteur[i])
        residus.append(inertie - surface)
    return _trois(residus)


def residus_de_navier_stokes(
    masse_volumique: Grandeur,
    viscosite: Grandeur,
    vitesse: Vecteur,
    pression: Grandeur,
    pesanteur: Vecteur,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`rho Dv/Dt + grad p - rho g - mu laplacien(v)`, en incompressible."""
    acceleration = _acceleration(vitesse)
    pentes = gradient(sp.sympify(pression), ESPACE)
    residus = []
    for i in range(3):
        visqueux = viscosite * laplacien(sp.sympify(vitesse[i]), ESPACE)
        inertie = masse_volumique * (acceleration[i] - pesanteur[i])
        residus.append(inertie + pentes[i] - visqueux)
    return _trois(residus)
