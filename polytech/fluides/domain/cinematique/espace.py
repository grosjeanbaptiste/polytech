"""La rotation et la deformation d'une particule, en trois dimensions.

Chapitre III.4 et III.5 des planches. La **vorticite** est le rotationnel
de la vitesse, `zeta = rot V`, et le taux de rotation instantane en vaut
la moitie : `omega_z = (dv/dx - du/dy) / 2`, et de meme par permutation.
Un ecoulement est rotationnel la ou elle n'est pas nulle.

Le **tenseur des deformations** est la part symetrique du gradient de
vitesse, `eps_ij = (du_i/dx_j + du_j/dx_i) / 2`. Ses composantes
diagonales sont les vitesses de dilatation dans chaque direction, et leur
somme la dilatation volumique `(1 / V) DV/Dt = div V` ; les autres sont
les vitesses de cisaillement.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import ESPACE

Champ = tuple[sp.Expr, sp.Expr, sp.Expr]


def _gradient(champ: Champ) -> sp.Matrix:
    return sp.Matrix(3, 3, lambda i, j: sp.diff(sp.sympify(champ[i]), ESPACE[j]))


def vorticite_dans_l_espace(champ: Champ) -> Champ:
    """`zeta = rot V`."""
    g = _gradient(champ)
    return (g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1])


def taux_de_rotation_dans_l_espace(champ: Champ) -> Champ:
    """`omega = (rot V) / 2`."""
    x, y, z = (composante / 2 for composante in vorticite_dans_l_espace(champ))
    return (x, y, z)


def tenseur_des_deformations(champ: Champ) -> sp.Matrix:
    """`eps_ij = (du_i/dx_j + du_j/dx_i) / 2`."""
    gradient = _gradient(champ)
    return (gradient + gradient.T) / 2


def dilatation_volumique(champ: Champ) -> sp.Expr:
    """`(1 / V) DV/Dt = eps_xx + eps_yy + eps_zz = div V`."""
    return tenseur_des_deformations(champ).trace()
