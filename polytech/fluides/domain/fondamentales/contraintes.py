"""Le tenseur des contraintes d'un fluide newtonien.

Chapitre IV.2 et IV.3 des planches. `sigma_ij` est la contrainte dans la
direction `j` qui s'applique sur un plan perpendiculaire a la direction
`i`. Il se separe en contraintes de pression et contraintes de viscosite,
`sigma = -p I + tau` ; au repos, il ne reste que la pression.

Pour un ecoulement incompressible d'un fluide newtonien, `tau = 2 mu eps`,
`eps` etant le tenseur des deformations : c'est la loi de comportement qui
ferme les equations de conservation.

La force qu'un fluide exerce sur un element de surface de normale `n` est
`dF = sigma . n dA`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.espace import tenseur_des_deformations

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

Vecteur = tuple[Grandeur, Grandeur, Grandeur]


def contraintes_visqueuses(vitesse: Vecteur, viscosite: Grandeur) -> sp.Matrix:
    """`tau = 2 mu eps`."""
    x, y, z = (sp.sympify(composante) for composante in vitesse)
    return 2 * viscosite * tenseur_des_deformations((x, y, z))


def tenseur_des_contraintes(
    pression: Grandeur, vitesse: Vecteur, viscosite: Grandeur
) -> sp.Matrix:
    """`sigma = -p I + 2 mu eps`."""
    return -pression * sp.eye(3) + contraintes_visqueuses(vitesse, viscosite)


def force_de_surface(
    contraintes: sp.Matrix, normale: Vecteur, aire: Grandeur
) -> Vecteur:
    """`dF = sigma . n dA`."""
    x, y, z = contraintes * sp.Matrix(normale) * aire
    return (x, y, z)
