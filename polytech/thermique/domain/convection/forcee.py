"""La convection forcee, par les correlations que le formulaire ecrit en clair.

Trois sont reprises mot pour mot, avec leurs domaines :

* plaque plane, laminaire : `Nu = 0,664 Re^0,5 Pr^(1/3)`, `Re_crit = 3 a 5.10^5` ;
* plaque plane, turbulent : `Nu = 0,037 Re^0,8 Pr^(1/3)` ;
* cylindre en travers : `Nu_d = C Re_d^n Pr^(1/3)`, avec les cinq plages
  de `(C, n)` que le formulaire tabule.

La table du cylindre est une **donnee**, pas une formule : elle est donc
transcrite telle quelle, bornes comprises, et une valeur de Reynolds hors
de ses plages leve au lieu d'extrapoler. Extrapoler une correlation
empirique hors de son domaine est l'erreur que ce chapitre apprend
justement a ne pas commettre.
"""

from __future__ import annotations

from typing import Final

REYNOLDS_CRITIQUE_PLAQUE: Final = 3e5
"""Le formulaire donne « 3 a 5.10^5 » : la borne basse est retenue."""

REYNOLDS_CRITIQUE_TUBE: Final = 2400
"""Valeur du formulaire pour un tube de section circulaire."""

_TIERS = 1.0 / 3.0

TABLE_DU_CYLINDRE: Final = (
    (0.4, 4.0, 0.989, 0.330),
    (4.0, 40.0, 0.911, 0.385),
    (40.0, 4000.0, 0.683, 0.466),
    (4000.0, 40000.0, 0.193, 0.618),
    (40000.0, 400000.0, 0.0266, 0.805),
)
"""Les cinq plages du formulaire : (Re min, Re max, C, n)."""


def reynolds(vitesse: float, longueur: float, viscosite_cinematique: float) -> float:
    """`Re = v L / nu`."""
    if viscosite_cinematique <= 0:
        msg = f"viscosite cinematique nulle ou negative : {viscosite_cinematique}"
        raise ValueError(msg)
    return vitesse * longueur / viscosite_cinematique


def prandtl(viscosite_cinematique: float, diffusivite: float) -> float:
    """`Pr = nu / alpha`."""
    if diffusivite <= 0:
        msg = f"diffusivite nulle ou negative : {diffusivite}"
        raise ValueError(msg)
    return viscosite_cinematique / diffusivite


def nusselt_plaque(reynolds_local: float, prandtl_local: float) -> float:
    """La correlation du formulaire, laminaire ou turbulente selon `Re`."""
    if reynolds_local < REYNOLDS_CRITIQUE_PLAQUE:
        return float(0.664 * reynolds_local**0.5 * prandtl_local**_TIERS)
    return float(0.037 * reynolds_local**0.8 * prandtl_local**_TIERS)


def nusselt_cylindre(reynolds_local: float, prandtl_local: float) -> float:
    """`Nu_d = C Re^n Pr^(1/3)` ; leve hors des plages tabulees."""
    for borne_basse, borne_haute, coefficient, exposant in TABLE_DU_CYLINDRE:
        if borne_basse <= reynolds_local <= borne_haute:
            return float(
                coefficient * reynolds_local**exposant * prandtl_local**_TIERS,
            )
    msg = (
        f"Reynolds hors des plages du formulaire : {reynolds_local:.3g} "
        f"n'est dans aucune des {len(TABLE_DU_CYLINDRE)} plages tabulees"
    )
    raise ValueError(msg)


def coefficient_d_echange(
    nusselt: float,
    conductivite: float,
    longueur: float,
) -> float:
    """`K = Nu lambda / L`, le retour du nombre sans dimension au coefficient."""
    if longueur <= 0:
        msg = f"longueur caracteristique nulle ou negative : {longueur}"
        raise ValueError(msg)
    return nusselt * conductivite / longueur
