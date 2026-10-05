"""Les donnees du TP 2 de thermique : un camion refrigere, un four.

Source : `ThermiqueFondements/TP/TP_02_Camion_refrigere_Enonce.pdf`, page
5 pour les donnees, fiches techniques pages 2 et 3.

**Camion.** Le cas le plus defavorable de l'enonce : ete (30 degres) et
plein soleil (800 W/m2). Le toit fait 5 m par 2 m.

**Four.** Section rectangulaire de 1,6 m sur 0,6 m entre les deux plans
de resistances ; la tole de 1,2 m est a mi-hauteur, centree. Deux
emissivites ne sont **pas dans l'enonce** et viennent du corrige
officiel : celle de la tole (0,5) et celle des parois (0,7). La seconde
ne sert d'ailleurs a rien, une paroi adiabatique n'opposant aucune
resistance de surface.

Les aires d'echange se calculent par les cordes croisees, sur la moitie
superieure du four, la symetrie donnant le reste : origine au coin
gauche, au niveau de la tole.
"""

from __future__ import annotations

from typing import Final

from polytech.thermique.domain.grandeurs import ZERO_ABSOLU
from polytech.thermique.domain.paroi_exposee import ToitExpose
from polytech.thermique.domain.rayonnement.cordes_croisees import (
    aire_par_les_cordes,
    corde_tendue,
)
from polytech.thermique.domain.rayonnement.paroi_adiabatique import (
    EnceinteAParoiAdiabatique,
)

KELVIN: Final = ZERO_ABSOLU
SURFACE_DU_TOIT: Final = 5.0 * 2.0
EPAISSEUR, CONDUCTIVITE, CONVECTION_INTERIEURE = 0.03, 0.035, 8.0
INTERIEUR, ETE, SOLEIL = -13.0, 30.0, 800.0
PEINTURES: Final = {
    "AZ-93": (0.15, 0.91),
    "AZW/LA-II": (0.09, 0.91),
    "AZ-70-WIZT": (0.10, 0.91),
    "AZ-2000-IECW": (0.25, 0.90),
    "AZ-2100-IECW": (0.19, 0.90),
    "AZ-400-LSW": (0.15, 0.91),
    "RM-400": (0.45, 0.88),
    "ML-210-IB": (0.98, 0.91),
    "RM-550-IB": (0.97, 0.91),
    "TMD-560-IB": (0.95, 0.91),
    "AZ-1000-ECB": (0.97, 0.89),
    "RM-550-LSB": (0.97, 0.91),
    "MLS-85-SB": (0.98, 0.91),
    "MLS-85-SB-C": (0.98, 0.91),
}
"""Fiche technique 2 : absorptivite solaire et emissivite de chaque peinture."""

HAUTEUR, LARGEUR, LARGEUR_TOLE, EPAISSEUR_TOLE = 0.6, 1.6, 1.2, 0.01
RESISTANCES, TOLE_INITIALE = 900.0, 20.0
EMISSIVITE_RESISTANCES, EMISSIVITE_TOLE = 0.6, 0.5
EMISSIVITE_PAROIS: Final = 0.7
"""Supposee par l'officiel, et sans effet : une paroi adiabatique n'en a pas besoin."""
MASSE_VOLUMIQUE, CHALEUR_SPECIFIQUE, CONDUCTIVITE_TOLE = 7850.0, 416.0, 45.0
PAROIS_SELON_L_OFFICIEL: Final = 801.4
"""La temperature des parois que donne le corrige officiel, page 22."""


def conductance_interieure() -> float:
    """`1 / (1/K_i + e/lambda)` : convection interieure et isolant en serie."""
    return 1 / (1 / CONVECTION_INTERIEURE + EPAISSEUR / CONDUCTIVITE)


def toit_du_camion(peinture: str, convection: float) -> ToitExpose:
    """Le bilan du toit pour une peinture et un coefficient de convection."""
    absorptivite, emissivite = PEINTURES[peinture]
    return ToitExpose(
        conductance_interieure(),
        INTERIEUR + KELVIN,
        convection,
        ETE + KELVIN,
        absorptivite,
        SOLEIL,
        emissivite,
    )


def aires_du_four() -> tuple[float, float, float]:
    """`S1S2`, `S1S3`, `S2S3` par metre, des quatre quarts symetriques."""
    demi, marge = HAUTEUR / 2, (LARGEUR - LARGEUR_TOLE) / 2
    a, b = (0.0, demi), (LARGEUR, demi)
    c, d = (marge, 0.0), (LARGEUR - marge, 0.0)
    milieu, bas = (0.0, 0.0), (0.0, -demi)
    plafond_tole = aire_par_les_cordes(
        (corde_tendue(a, d), corde_tendue(b, c)),
        (corde_tendue(a, c), corde_tendue(b, d)),
    )
    plafond_paroi = aire_par_les_cordes(
        (corde_tendue(a, bas), corde_tendue(b, a)),
        (0.0, corde_tendue(b, c, bas)),
    )
    tole_paroi = aire_par_les_cordes(
        (corde_tendue(c, a), corde_tendue(d, milieu)),
        (corde_tendue(c, milieu), corde_tendue(d, a)),
    )
    return 2 * plafond_tole, 4 * plafond_paroi, 4 * tole_paroi


def enceinte_du_four() -> EnceinteAParoiAdiabatique:
    """Resistances, tole et parois, a l'instant initial."""
    s12, s13, s23 = aires_du_four()
    return EnceinteAParoiAdiabatique(
        aire_1=2 * LARGEUR,
        emissivite_1=EMISSIVITE_RESISTANCES,
        temperature_1=RESISTANCES + KELVIN,
        aire_2=2 * LARGEUR_TOLE,
        emissivite_2=EMISSIVITE_TOLE,
        temperature_2=TOLE_INITIALE + KELVIN,
        echange_12=s12,
        echange_13=s13,
        echange_23=s23,
    )


def vitesse_de_chauffe() -> float:
    """`dT/dt = phi / (rho c V)` par metre de longueur, en degres par seconde."""
    capacite = MASSE_VOLUMIQUE * CHALEUR_SPECIFIQUE * LARGEUR_TOLE * EPAISSEUR_TOLE
    return enceinte_du_four().flux() / capacite
