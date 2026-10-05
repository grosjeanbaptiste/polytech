"""La conservation de l'energie d'un fluide.

Chapitre IV.4 des planches. L'energie cinetique d'un domaine recoit la
puissance des forces de contact et celle des forces a distance ; elle en
cede une part a l'energie interne, par la puissance des efforts internes
`eps : sigma` -- la dissipation visqueuse, qui eleve la temperature. Seule
la part symetrique du gradient de vitesse y contribue, le tenseur des
contraintes etant symetrique.

L'energie interne suit `rho DU/Dt = -div q + r + eps : sigma`, avec la loi
de Fourier `q = -lambda grad T`. Pour un ecoulement incompressible, sans
source interne et a echauffement visqueux negligeable, elle se reduit a

`rho c_p DT/Dt = div(lambda grad T)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.espace import tenseur_des_deformations
from polytech.fluides.domain.cinematique.symboles import ESPACE, T
from polytech.maths.domain.analyse_vectorielle import divergence, gradient

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

Vecteur = tuple[Grandeur, Grandeur, Grandeur]


def puissance_des_efforts_internes(vitesse: Vecteur, contraintes: sp.Matrix) -> sp.Expr:
    """`eps : sigma`, en watts par metre cube."""
    x, y, z = (sp.sympify(composante) for composante in vitesse)
    deformations = tenseur_des_deformations((x, y, z))
    termes = (
        deformations[i, j] * contraintes[i, j] for i in range(3) for j in range(3)
    )
    return sp.simplify(sp.Add(*termes))


def residu_de_la_chaleur(
    masse_volumique: Grandeur,
    chaleur_massique: Grandeur,
    conductivite: Grandeur,
    temperature: sp.Expr,
    vitesse: Vecteur,
) -> sp.Expr:
    """`rho c_p DT/Dt - div(lambda grad T)` : nul si l'energie se conserve."""
    pentes = gradient(temperature, ESPACE)
    advection = sp.Add(*(v * pente for v, pente in zip(vitesse, pentes, strict=True)))
    materielle = sp.diff(temperature, T) + advection
    flux = tuple(conductivite * pente for pente in pentes)
    inertie = masse_volumique * chaleur_massique * materielle
    return sp.simplify(inertie - divergence(flux, ESPACE))
