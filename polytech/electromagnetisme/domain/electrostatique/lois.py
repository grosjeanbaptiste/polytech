"""Les deux proprietes fondamentales du champ electrostatique.

Les planches les nomment ainsi : la premiere est le **flux**, qui donne
`div E = (eta + eta') / epsilon_0` ; la seconde est la **circulation**,
nulle sur tout contour ferme, qui donne `rot E = 0` et donc l'existence du
potentiel.

La divergence est exposee dans les **deux** systemes de coordonnees parce
que les exercices passent de l'un a l'autre -- « Statique 1.1 » part de la
forme cartesienne et la reecrit en cylindriques, et l'erreur la plus
courante y est d'oublier le facteur `1/rho`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.symboles import (
    ABSCISSE,
    AZIMUT,
    COTE,
    ORDONNEE,
    RAYON,
)


def divergence_cartesienne(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """`dEx/dx + dEy/dy + dEz/dz`."""
    selon_x, selon_y, selon_z = champ
    return sp.simplify(
        sp.diff(selon_x, ABSCISSE)
        + sp.diff(selon_y, ORDONNEE)
        + sp.diff(selon_z, COTE),
    )


def divergence_cylindrique(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """`(1/rho) d(rho E_rho)/drho + (1/rho) dE_theta/dtheta + dEz/dz`.

    Le facteur `1/rho` est ce qu'on oublie, et son oubli ne se voit pas :
    l'expression reste homogene et le resultat est simplement faux.
    """
    radiale, azimutale, axiale = champ
    return sp.simplify(
        sp.diff(RAYON * radiale, RAYON) / RAYON
        + sp.diff(azimutale, AZIMUT) / RAYON
        + sp.diff(axiale, COTE),
    )


def rotationnel_cartesien(
    champ: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le rotationnel, nul pour tout champ electrostatique."""
    selon_x, selon_y, selon_z = champ
    return (
        sp.simplify(sp.diff(selon_z, ORDONNEE) - sp.diff(selon_y, COTE)),
        sp.simplify(sp.diff(selon_x, COTE) - sp.diff(selon_z, ABSCISSE)),
        sp.simplify(sp.diff(selon_y, ABSCISSE) - sp.diff(selon_x, ORDONNEE)),
    )


def est_electrostatique(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> bool:
    """Vrai si le rotationnel est nul : le champ derive alors d'un potentiel."""
    return all(composante == 0 for composante in rotationnel_cartesien(champ))


def densite_de_charge(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """`eta = epsilon_0 div E` : la charge que ce champ implique."""
    return sp.simplify(PERMITTIVITE_DU_VIDE * divergence_cartesienne(champ))


def champ_de_coulomb(charge: float, distance: float) -> float:
    """`q / (4 pi epsilon_0 r^2)` ; leve a distance nulle."""
    if distance <= 0:
        msg = f"distance nulle ou negative : {distance} m"
        raise ValueError(msg)
    return charge / (4 * math.pi * PERMITTIVITE_DU_VIDE * distance**2)
