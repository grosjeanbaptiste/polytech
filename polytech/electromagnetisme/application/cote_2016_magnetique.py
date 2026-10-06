"""Exercice cote 2016, questions 2 et 6 : deux circuits magnetiques.

Dans le fer l'induction se lit sur la courbe du materiau ; dans l'air
elle se calcule. Les deux questions tiennent dans cette difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    champ_pour_les_amperes_tours,
    inductance_depuis_les_flux,
    potentiel_scalaire_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    CircuitAEntrefers,
    champ_dans_l_entrefer,
)


@dataclass(frozen=True, slots=True)
class Colonnes:
    """Les flux de la question 2, et l'induction des colonnes laterales."""

    flux_central: float
    flux_lateral: float
    induction_laterale: float


def etat_des_colonnes() -> Colonnes:
    """Le flux central se partage en deux, par symetrie."""
    section_centrale = en.LARGEURS[2] * en.EPAISSEUR
    section_laterale = en.LARGEURS[0] * en.EPAISSEUR
    flux_central = en.INDUCTION_LUE_A_250 * section_centrale
    flux_lateral = flux_central / 2
    return Colonnes(flux_central, flux_lateral, flux_lateral / section_laterale)


def longueurs_des_contours() -> tuple[float, float]:
    """La colonne centrale, puis le reste du tour d'une fenetre, a mi-epaisseur."""
    haute, fenetre, basse = en.HAUTEURS
    gauche, ouverture, centre = en.LARGEURS[:3]
    centrale = haute / 2 + fenetre + basse / 2
    culasse = centre / 2 + ouverture + gauche / 2
    return centrale, 2 * culasse + centrale


def courant_des_bobines() -> float:
    """`I = (H_c l_c + H_1 l_1) / N` : Ampere sur le tour d'une fenetre."""
    centrale, laterale = longueurs_des_contours()
    amperes_tours = potentiel_scalaire_magnetique(
        en.CHAMP_CENTRAL, centrale
    ) + potentiel_scalaire_magnetique(en.CHAMP_CENTRAL, laterale)
    return amperes_tours / en.SPIRES_Q2


def inductance_apparente() -> float:
    """`N Phi_1 / I` : ce que vaudrait `L11 + L12` si le circuit etait lineaire."""
    flux = etat_des_colonnes().flux_lateral
    return inductance_depuis_les_flux(en.SPIRES_Q2, (flux,), courant_des_bobines())


def _section_du_tore() -> float:
    return math.pi * en.RAYON_DE_SECTION**2


def _courant_du_tore() -> float:
    return en.E0_Q6 / en.R0_Q6


def flux_du_tore() -> float:
    """`Phi = B S`, l'induction etant lue pour 50 A/m."""
    return en.INDUCTION_LUE_A_50 * _section_du_tore()


def spires_du_tore() -> int:
    """`N = L I / Phi`, arrondi a l'entier : un nombre de spires."""
    return round(en.INDUCTANCE_VOULUE * _courant_du_tore() / flux_du_tore())


def champ_d_entrefer() -> float:
    """`H_e = B / mu_0`, le flux se conservant du fer a l'air."""
    return champ_dans_l_entrefer(en.INDUCTION_LUE_A_50)


def ecartement() -> float:
    """`l_e`, en metres : `N I = H l + 2 H_e l_e`, deux entrefers."""
    longueur = 2 * math.pi * en.RAYON_MOYEN
    dans_le_fer = potentiel_scalaire_magnetique(en.CHAMP_VOULU, longueur)
    reste = spires_du_tore() * _courant_du_tore() - dans_le_fer
    return reste / (2 * champ_d_entrefer())


def circuit_ecarte() -> CircuitAEntrefers:
    """Le meme circuit, reconstruit avec l'ecartement trouve."""
    return CircuitAEntrefers(
        champ_dans_le_fer=en.CHAMP_VOULU,
        longueur_du_fer=2 * math.pi * en.RAYON_MOYEN,
        induction=en.INDUCTION_LUE_A_50,
        epaisseur=ecartement(),
        entrefers=2,
    )


def demi_tores_jointifs() -> tuple[float, float]:
    """Sans entrefer : le champ dans le fer, puis l'inductance a saturation."""
    amperes_tours = spires_du_tore() * _courant_du_tore()
    champ = champ_pour_les_amperes_tours(amperes_tours, 2 * math.pi * en.RAYON_MOYEN)
    flux = en.INDUCTION_DE_SATURATION * _section_du_tore()
    return champ, inductance_depuis_les_flux(
        spires_du_tore(), (flux,), _courant_du_tore()
    )
