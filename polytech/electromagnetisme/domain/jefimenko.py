"""Les equations de Jefimenko : les champs, directement des sources retardees.

Planches du cours d'electronique (chapitre 1, partie I, planches 46 a
51). Resoudre les equations de Maxwell en jauge de Lorenz donne les
potentiels retardes ; les y reporter donne les champs eux-memes :

    E = (1 / 4 pi eps_0) integrale[ (R / R^3) rho + (R / (R^2 c)) d rho/dt
                                    - (1 / (R c^2)) dJ/dt ] dV',
    B = -(mu_0 / 4 pi) integrale[ (R / R^3) x J + (R / (R^2 c)) x dJ/dt ] dV',

ou `R = r - r'` et ou les sources se prennent au **temps retarde**
`t_r = t - R / c` : l'information electromagnetique voyage a la vitesse
de la lumiere. Sans derivees temporelles et sans retard, le premier
terme de chacune est la loi de Coulomb et celle de Biot et Savart.

Ce module porte l'**integrande** : la contribution d'un element de
source, dont une integration fait le champ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
    VITESSE_DE_LA_LUMIERE,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

if TYPE_CHECKING:
    Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]

VITESSE_TYPIQUE_DANS_UN_CIRCUIT: Final = VITESSE_DE_LA_LUMIERE / 3
"""Planche 51 : « c / 3 ~ 1e8 m/s is generally a good rule of thumb »."""


def temps_retarde(instant: sp.Expr, distance: sp.Expr, celerite: sp.Expr) -> sp.Expr:
    """`t_r = t - |r - r'| / c`."""
    if sp.sympify(celerite).is_nonpositive:
        msg = f"celerite nulle ou negative : {celerite}"
        raise ValueError(msg)
    return instant - distance / celerite


def _retarder(source: sp.Expr, distance: sp.Expr, celerite: sp.Expr) -> sp.Expr:
    return sp.sympify(source).subs(TEMPS, temps_retarde(TEMPS, distance, celerite))


def champ_electrique_elementaire(
    ecart: Vecteur, charge: sp.Expr, courant: Vecteur, *, celerite: sp.Expr
) -> Vecteur:
    """La contribution a `E`, par unite de volume de source, en `r' + ecart`."""
    vers = sp.Matrix(ecart)
    distance = sp.sqrt(vers.dot(vers))
    densite = _retarder(charge, distance, celerite)
    variation = _retarder(sp.diff(sp.sympify(charge), TEMPS), distance, celerite)
    rayonne = sp.Matrix(
        [_retarder(sp.diff(sp.sympify(c), TEMPS), distance, celerite) for c in courant]
    )
    proche = vers / distance**3 * densite + vers / (distance**2 * celerite) * variation
    total = (proche - rayonne / (distance * celerite**2)) / (
        4 * sp.pi * PERMITTIVITE_DU_VIDE
    )
    return (total[0], total[1], total[2])


def champ_magnetique_elementaire(
    ecart: Vecteur, courant: Vecteur, *, celerite: sp.Expr
) -> Vecteur:
    """La contribution a `B`, par unite de volume de source, en `r' + ecart`."""
    vers = sp.Matrix(ecart)
    distance = sp.sqrt(vers.dot(vers))
    densite = sp.Matrix([_retarder(c, distance, celerite) for c in courant])
    variation = sp.Matrix(
        [_retarder(sp.diff(sp.sympify(c), TEMPS), distance, celerite) for c in courant]
    )
    proche = (vers / distance**3).cross(densite)
    retarde = (vers / (distance**2 * celerite)).cross(variation)
    total = -PERMEABILITE_DU_VIDE / (4 * sp.pi) * (proche + retarde)
    return (total[0], total[1], total[2])


def poids_du_terme_retarde(distance: float, pulsation: float, celerite: float) -> float:
    """`omega R / c` : ce que pese le terme en `dJ/dt` devant Biot et Savart.

    Pour une source sinusoidale, c'est `2 pi R / lambda` : Biot et Savart
    reste une bonne approximation tant que le circuit est petit devant la
    longueur d'onde.
    """
    if celerite <= 0:
        msg = f"celerite nulle ou negative : {celerite} m/s"
        raise ValueError(msg)
    return pulsation * distance / celerite
