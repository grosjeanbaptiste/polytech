"""Le cisaillement simple entre deux plaques, et le viscosimetre a cylindres.

Chapitre I.2.1 des planches. Un fluide occupe l'ecart `l` entre une
plaque fixe et une plaque d'aire `A` tiree a la vitesse `V` par une force
`F`. Le profil de vitesse est lineaire, `u(y) = y V / l`, le gradient
vaut `V / l`, et la contrainte de cisaillement `tau = F / A`. Pour un
fluide newtonien, `F = tau A = mu A du/dy`.

Un viscosimetre a cylindres coaxiaux enroule ce cisaillement : le
cylindre interieur, de rayon `R` et de longueur `L`, tourne a la vitesse
angulaire `omega` dans un entrefer `l` petit devant `R`. La vitesse de sa
paroi est `omega R`, l'aire mouillee `2 pi R L`, et le couple mesure est
la force au rayon `R` : `C = mu 2 pi R^3 omega L / l`.
"""

from __future__ import annotations

import math

from polytech.fluides.domain.viscosite import (
    contrainte_de_newton,
    viscosite_depuis_la_contrainte,
)


def contrainte_tangentielle(force: float, aire: float) -> float:
    """`tau = F / A`, en pascals."""
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    return force / aire


def vitesse_de_couette(ordonnee: float, ecart: float, vitesse: float) -> float:
    """`u(y) = y V / l` : nulle sur la plaque fixe, `V` sur la plaque mobile."""
    if ecart <= 0:
        msg = f"ecart nul ou negatif : {ecart} m"
        raise ValueError(msg)
    if not 0 <= ordonnee <= ecart:
        msg = f"ordonnee hors de l'entrefer : {ordonnee} m pour un ecart de {ecart} m"
        raise ValueError(msg)
    return ordonnee * vitesse / ecart


def force_de_cisaillement(viscosite: float, aire: float, gradient: float) -> float:
    """`F = tau A = mu A du/dy`, en newtons."""
    return contrainte_de_newton(viscosite, gradient) * aire


def viscosite_au_viscosimetre(
    *,
    couple: float,
    rayon: float,
    entrefer: float,
    longueur: float,
    tours_par_minute: float,
) -> float:
    """`mu = C l / (2 pi R^3 omega L)`, en Pa.s."""
    if rayon <= 0 or entrefer <= 0 or longueur <= 0:
        msg = (
            f"geometrie invalide : rayon {rayon} m, entrefer {entrefer} m, "
            f"longueur {longueur} m"
        )
        raise ValueError(msg)
    if tours_par_minute == 0:
        msg = "cylindre immobile : aucun gradient de vitesse, donc aucune mesure"
        raise ValueError(msg)
    vitesse = tours_par_minute * 2 * math.pi / 60 * rayon
    aire = 2 * math.pi * rayon * longueur
    contrainte = contrainte_tangentielle(couple / rayon, aire)
    return viscosite_depuis_la_contrainte(contrainte, vitesse / entrefer)
