"""Cours II.2 -- d'ou vient la loi de l'hydrostatique.

La loi ``dp/dz = -rho g`` n'est pas un postulat : elle exprime l'equilibre
vertical d'une tranche de fluide entre les altitudes ``z`` et ``z + dz``.
Trois forces s'y exercent -- la poussee du fluide du dessous, celle du
fluide du dessus, et le poids -- et leur somme nulle donne la loi. Ce module
mene ce bilan puis l'integre, afin que le corrige montre les deux etapes au
lieu de les annoncer.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.statique.symboles import RHO, G, Z

SECTION = sp.Symbol("S", positive=True)
"""Section horizontale de la tranche : elle se simplifie a la fin."""

EPAISSEUR = sp.Symbol(r"\mathrm{d}z", positive=True)
"""Epaisseur infinitesimale de la tranche consideree."""

ALTITUDE_BASSE = sp.Symbol("z_1")
ALTITUDE_HAUTE = sp.Symbol("z_2")

PRESSION = sp.Function("p")

_ORDRE_LINEAIRE = 2


def forces_sur_la_tranche() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Les trois forces verticales, dans l'ordre du bilan.

    Le fluide du dessous pousse vers le haut, celui du dessus vers le bas,
    et le poids tire vers le bas.
    """
    dessous = PRESSION(Z) * SECTION
    dessus = -PRESSION(Z + EPAISSEUR) * SECTION
    poids = -RHO * G * SECTION * EPAISSEUR
    return (dessous, dessus, poids)


def bilan_des_forces() -> sp.Expr:
    """Somme des forces verticales sur la tranche ``[z, z + dz]``."""
    return sp.Add(*forces_sur_la_tranche(), evaluate=False)


def developpement_au_premier_ordre() -> sp.Expr:
    """Developpe ``p(z + dz)`` autour de ``dz = 0``, seul terme non trivial."""
    return sp.series(bilan_des_forces(), EPAISSEUR, 0, _ORDRE_LINEAIRE).removeO()


def loi_locale() -> sp.Eq:
    """Equilibre de la tranche : ``dp/dz + rho g = 0`` (cours II.2)."""
    residu = sp.expand(developpement_au_premier_ordre() / (-SECTION * EPAISSEUR))
    return sp.Eq(sp.simplify(residu), 0)


def integration_sur_une_colonne() -> sp.Eq:
    """Integre ``dp = -rho g dz`` entre deux altitudes de la meme colonne.

    Le resultat ne retient que la *difference* d'altitude : c'est ce qui
    autorise la marche le long d'un manometre, et ce qui fait qu'une bosse
    se compense.
    """
    return sp.Eq(
        PRESSION(ALTITUDE_HAUTE) - PRESSION(ALTITUDE_BASSE),
        sp.factor(sp.integrate(-RHO * G, (Z, ALTITUDE_BASSE, ALTITUDE_HAUTE))),
        evaluate=False,
    )
