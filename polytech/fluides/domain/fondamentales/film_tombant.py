"""Un film d'huile qui tombe le long d'une paroi verticale.

Exercice du chapitre IV des planches. Un film d'epaisseur `h` s'ecoule,
regulier, incompressible, parallele et laminaire, le long d'une paroi
fixe et infinie placee en `x = 0` ; la gravite agit selon `-z`. Aucune
pression n'est appliquee : `P = P_atm` partout, l'huile tombe par simple
gravite.

Navier-Stokes se reduit a `mu d2w/dx2 = rho g`. L'huile adhere a la
paroi, `w(0) = 0`, et l'air n'exerce aucun cisaillement sur sa surface,
`dw/dx (h) = 0`. D'ou

`w(x) = -(rho g / (2 mu)) x (2 h - x)`,

une demi-parabole, la plus rapide en surface. Le debit par unite de
largeur vaut `rho g h^3 / (3 mu)`.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def _exiger_positifs(**grandeurs: Grandeur) -> None:
    for nom, valeur in grandeurs.items():
        if sp.sympify(valeur).is_positive is not True:
            msg = f"{nom} nulle ou negative : {valeur}"
            raise ValueError(msg)


def vitesse_du_film(
    abscisse: Grandeur,
    epaisseur: Grandeur,
    masse_volumique: Grandeur,
    viscosite: Grandeur,
    pesanteur: Grandeur,
) -> sp.Expr:
    """`w(x) = -(rho g / (2 mu)) x (2 h - x)` : negative, l'huile descend."""
    _exiger_positifs(epaisseur=epaisseur, viscosite=viscosite)
    facteur = masse_volumique * pesanteur / (2 * viscosite)
    return -facteur * abscisse * (2 * epaisseur - abscisse)


def debit_du_film(
    epaisseur: Grandeur,
    masse_volumique: Grandeur,
    viscosite: Grandeur,
    pesanteur: Grandeur,
) -> sp.Expr:
    """`q = rho g h^3 / (3 mu)`, vers le bas, par unite de largeur."""
    _exiger_positifs(epaisseur=epaisseur, viscosite=viscosite)
    return masse_volumique * pesanteur * sp.sympify(epaisseur) ** 3 / (3 * viscosite)
