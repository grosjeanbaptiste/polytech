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

from polytech.thermique.domain.convection.forcee_dimensions import (  # noqa: F401 -- re-export : noms historiques de ce module
    DIMENSIONS_FONDAMENTALES,
    VARIABLES_DE_LA_CONVECTION_FORCEE,
    groupes_sans_dimension,
    nombre_de_groupes_de_vaschy_buckingham,
)

REYNOLDS_CRITIQUE_PLAQUE: Final = 3e5
"""Le formulaire donne « 3 a 5.10^5 » : la borne basse est retenue."""

REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019: Final = 2400
"""Formulaire 2019 (tube de section circulaire), examens, fiche, tome 2 chap. 3.
Le tome 2, section 1.3.2, dit 2320 : voir
`couches_limites.REYNOLDS_TRANSITION_EN_TUBE_TOME_2_SECTION_1_3_2`."""
REYNOLDS_CRITIQUE_TUBE: Final = REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019
"""Nom historique, meme valeur."""

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


EXPOSANT_LAMINAIRE: Final = 0.5
"""L'exposant de Reynolds en regime laminaire, sur plaque plane."""

EXPOSANT_TURBULENT: Final = 0.8
"""Et en turbulent. Plus grand, donc l'echange croit plus vite."""


def coefficient_depuis_le_nusselt(
    nusselt: float,
    conductivite: float,
    longueur: float,
) -> float:
    """`K = Nu lambda / L` : le retour du nombre sans dimension au physique.

    C'est la derniere etape de toute correlation, et celle qu'on oublie :
    un Nusselt n'est pas un coefficient d'echange, il faut le diviser par
    la longueur caracteristique -- **la meme** que celle du Reynolds.
    """
    if longueur <= 0:
        msg = f"longueur caracteristique nulle ou negative : {longueur} m"
        raise ValueError(msg)
    if conductivite <= 0:
        msg = f"conductivite nulle ou negative : {conductivite} W/(m K)"
        raise ValueError(msg)
    return nusselt * conductivite / longueur


def effet_de_la_turbulence() -> tuple[float, float]:
    """Les deux exposants de Reynolds, laminaire puis turbulent.

    `0,5` contre `0,8` : la turbulence melange, donc transporte mieux, et
    l'echange croit plus vite avec la vitesse. C'est ce qui justifie de
    turbuler volontairement un ecoulement -- promoteurs de turbulence,
    rugosite, ailettes -- malgre la perte de charge.
    """
    return EXPOSANT_LAMINAIRE, EXPOSANT_TURBULENT


def couplage_vitesse_temperature() -> str:
    """Ce que le chapitre avoue d'entree, et qui explique tout le reste.

    « Couplage champ de vitesse / champ de temperature / flux a la
    paroi » : le champ de vitesse decide du champ de temperature, qui par
    la masse volumique decide du champ de vitesse. Le systeme est couple,
    donc non lineaire, donc insoluble en forme fermee -- et c'est pourquoi
    la convection se **correle** au lieu de se calculer.
    """
    return (
        "le champ de vitesse decide du champ de temperature, qui par la masse "
        "volumique decide du champ de vitesse : le systeme est couple, donc "
        "la convection se correle au lieu de se calculer"
    )
