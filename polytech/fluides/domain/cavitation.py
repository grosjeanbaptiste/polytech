"""L'aspiration d'une pompe : pression statique minimale et cavitation.

Bernoulli de la surface libre (vitesse nulle, pression `p_s`) au point
`h` metres au-dessus d'elle, dans la conduite d'aspiration :
`p = p_s - rho g h - rho V^2 / 2 - Delta p_L`. Le point d'aspiration est
le **minimum** de pression du circuit : si `p <= p_vapeur`, le liquide
bout et des bulles se forment (cavitation). `NPSH` (*net positive suction
head*) dit la marge, en metres de colonne de liquide :
`NPSH = (p + rho V^2 / 2 - p_v) / (rho g)`.
"""

from __future__ import annotations

from typing import Final

from polytech.fluides.domain.grandeurs import PESANTEUR

PRESSION_DE_VAPEUR_EAU_20C: Final = 2339.0
"""Pression de vapeur saturante de l'eau a 20 C, en pascals (tables usuelles)."""


def _exiger_masse_volumique(masse_volumique: float) -> None:
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)


def _exiger_perte(perte_de_charge: float) -> None:
    if perte_de_charge < 0:
        msg = f"une perte de charge est positive : {perte_de_charge} Pa"
        raise ValueError(msg)


def pression_dans_l_aspiration(
    pression_de_surface: float,
    masse_volumique: float,
    hauteur: float,
    vitesse: float,
    perte_de_charge: float,
) -> float:
    """`p_s - rho g h - rho V^2 / 2 - Delta p_L`, en pascals (peut etre negative)."""
    _exiger_masse_volumique(masse_volumique)
    _exiger_perte(perte_de_charge)
    return (
        pression_de_surface
        - masse_volumique * PESANTEUR * hauteur
        - masse_volumique * vitesse**2 / 2
        - perte_de_charge
    )


def risque_de_cavitation(pression: float, pression_de_vapeur: float) -> bool:
    """`p <= p_vapeur` : le liquide bout au point considere."""
    if pression_de_vapeur < 0:
        msg = f"pression de vapeur negative : {pression_de_vapeur} Pa"
        raise ValueError(msg)
    return pression <= pression_de_vapeur


def hauteur_d_aspiration_maximale(
    pression_de_surface: float,
    pression_de_vapeur: float,
    masse_volumique: float,
    vitesse: float,
    perte_de_charge: float,
) -> float:
    """La hauteur `h` ou `p = p_vapeur` : `(p_s - p_v - rho V^2/2 - dp_L) / (rho g)`."""
    _exiger_masse_volumique(masse_volumique)
    _exiger_perte(perte_de_charge)
    reste = (
        pression_de_surface
        - pression_de_vapeur
        - masse_volumique * vitesse**2 / 2
        - perte_de_charge
    )
    return reste / (masse_volumique * PESANTEUR)


def npsh_disponible(
    pression_a_l_entree: float,
    vitesse: float,
    pression_de_vapeur: float,
    masse_volumique: float,
) -> float:
    """`(p + rho V^2 / 2 - p_v) / (rho g)`, en metres de colonne de liquide."""
    _exiger_masse_volumique(masse_volumique)
    charge = pression_a_l_entree + masse_volumique * vitesse**2 / 2
    return (charge - pression_de_vapeur) / (masse_volumique * PESANTEUR)
