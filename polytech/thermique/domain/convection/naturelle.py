"""La convection naturelle, par Grashof, Rayleigh, et la lame de fluide.

Le formulaire donne la lame de fluide entre deux plaques en clair, avec
ses trois regimes :

* `Gr_b Pr < 1400` : `Nu_b = 1` -- le fluide ne bouge pas, il conduit ;
* `6000 < Gr_b Pr < 10^6` : `Nu_b = 0,11 (Gr_b Pr)^0,29` ;
* `10^6 < Gr_b Pr < 10^8` : `Nu_b = 0,40 (Gr_b Pr)^0,20`.

Le **trou** entre 1400 et 6000 est dans le formulaire, et il n'est pas
comble ici : c'est une zone de transition que le cours ne correle pas, et
y interpoler serait inventer une correlation. La garde le dit.
"""

from __future__ import annotations

import math
from typing import Final

PESANTEUR: Final = 9.81
"""Acceleration de la pesanteur, en m/s^2."""

CONDUCTION_PURE: Final = 1400.0
"""En dessous, le formulaire donne `Nu = 1` : la lame conduit sans bouger."""

DEBUT_DU_PREMIER_REGIME: Final = 6000.0
FIN_DU_PREMIER_REGIME: Final = 1e6
FIN_DU_SECOND_REGIME: Final = 1e8


def grashof(
    dilatation: float,
    ecart: float,
    longueur: float,
    viscosite_cinematique: float,
) -> float:
    """`Gr = g beta Delta T L^3 / nu^2`."""
    if viscosite_cinematique <= 0:
        msg = f"viscosite cinematique nulle ou negative : {viscosite_cinematique}"
        raise ValueError(msg)
    numerateur = PESANTEUR * dilatation * abs(ecart) * longueur**3
    return numerateur / viscosite_cinematique**2


def rayleigh(grashof_local: float, prandtl_local: float) -> float:
    """`Ra = Gr Pr`, le produit dont depend le regime."""
    return grashof_local * prandtl_local


def nusselt_lame_de_fluide(rayleigh_local: float) -> float:
    """La correlation du formulaire ; leve dans la zone qu'il ne correle pas."""
    if rayleigh_local < CONDUCTION_PURE:
        return 1.0
    if rayleigh_local < DEBUT_DU_PREMIER_REGIME:
        borne = f"{CONDUCTION_PURE} < {rayleigh_local:.3g} < {DEBUT_DU_PREMIER_REGIME}"
        msg = f"zone de transition non correlee par le formulaire : {borne}"
        raise ValueError(msg)
    if rayleigh_local <= FIN_DU_PREMIER_REGIME:
        return float(0.11 * rayleigh_local**0.29)
    if rayleigh_local <= FIN_DU_SECOND_REGIME:
        return float(0.40 * rayleigh_local**0.20)
    msg = (
        f"Gr Pr = {rayleigh_local:.3g} depasse la derniere plage du "
        f"formulaire ({FIN_DU_SECOND_REGIME:.0e})"
    )
    raise ValueError(msg)


def resistance_de_paroi_plane(
    epaisseur: float,
    conductivite: float,
    aire: float,
) -> float:
    """`e / (lambda S)`, en K/W : la forme la plus simple."""
    if epaisseur <= 0 or conductivite <= 0 or aire <= 0:
        msg = (
            f"paroi plane invalide : epaisseur {epaisseur} m, conductivite "
            f"{conductivite} W/(m K), aire {aire} m^2"
        )
        raise ValueError(msg)
    return epaisseur / (conductivite * aire)


def resistance_de_paroi_cylindrique(
    rayon_interieur: float,
    rayon_exterieur: float,
    conductivite: float,
    longueur: float,
) -> float:
    """`ln(r_e / r_i) / (2 pi lambda L)`, en K/W.

    Ce n'est **pas** `e / (lambda S)`. Transposer le cas plan au tube est
    l'erreur du chapitre, et l'ecart cesse d'etre negligeable des que
    l'epaisseur n'est plus petite devant le rayon.
    """
    if rayon_exterieur <= rayon_interieur:
        msg = (
            f"rayon exterieur non superieur a l'interieur : "
            f"{rayon_exterieur} <= {rayon_interieur} m"
        )
        raise ValueError(msg)
    if conductivite <= 0 or longueur <= 0:
        msg = f"conductivite {conductivite} ou longueur {longueur} non positive"
        raise ValueError(msg)
    return math.log(rayon_exterieur / rayon_interieur) / (
        2 * math.pi * conductivite * longueur
    )


def resistance_de_surface(echange: float, aire: float) -> float:
    """`1 / (K S)` : la resistance que le film de convection oppose."""
    if echange <= 0 or aire <= 0:
        msg = f"echange {echange} W/(m^2 K) ou aire {aire} m^2 non positif"
        raise ValueError(msg)
    return 1.0 / (echange * aire)


def coefficient_global_de_transmission(
    resistances: tuple[float, ...],
    aire: float,
) -> float:
    """`K = 1 / (S somme R)`, en W/(m^2 K).

    C'est la grandeur que les trois applications du fascicule emploient --
    la tablette, le double vitrage, le boiler -- et celle qui repond a sa
    question : les pertes au travers d'un vitrage se reduisent en ajoutant
    une resistance en serie, et l'air en est une bien meilleure que le
    verre.
    """
    if not resistances:
        msg = "aucune resistance : le coefficient global n'est pas defini"
        raise ValueError(msg)
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    return 1.0 / (aire * sum(resistances))


COEFFICIENT_DU_CYLINDRE: Final = 0.53
"""`Nu = 0,53 Ra^(1/4)` : la correlation du cylindre horizontal, laminaire."""

EXPOSANT_DU_CYLINDRE: Final = 0.25


def nusselt_du_cylindre_horizontal(rayleigh: float) -> float:
    """`0,53 Ra^(1/4)` : la geometrie a sa propre correlation.

    Le fascicule lui consacre une planche parce qu'un tube horizontal est
    la geometrie la plus courante en pratique -- une conduite de chauffage,
    un echangeur a faisceau -- et que la correlation de la plaque plane y
    serait fausse.
    """
    if rayleigh <= 0:
        msg = f"Rayleigh nul ou negatif : {rayleigh}"
        raise ValueError(msg)
    return float(COEFFICIENT_DU_CYLINDRE * rayleigh**EXPOSANT_DU_CYLINDRE)


def nusselt_d_un_espace_ferme(rayleigh: float) -> float:
    """La convection naturelle **interne** : lame d'air, double vitrage.

    Elle delegue a la correlation de la lame de fluide, qui est celle que
    le formulaire donne : sous le seuil de mise en mouvement, `Nu = 1` --
    le fluide ne bouge pas et ne fait que conduire. C'est ce qui rend un
    double vitrage efficace, et c'est pourquoi sa lame ne doit pas etre
    trop epaisse.
    """
    return nusselt_lame_de_fluide(rayleigh)
