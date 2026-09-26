"""Macros des *etapes* de raisonnement, et non de leurs seules conclusions.

Le corrige enoncait certaines lois -- equilibre d'une tranche de fluide,
decomposition du gradient de vitesse, compensation d'une bosse -- avant de
les utiliser. Ces macros publient les etapes intermediaires que la
bibliotheque etablit reellement, pour que le document les montre.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex, nombre
from polytech.fluides.domain.cinematique.champ_generique import CHAMP_GENERIQUE
from polytech.fluides.domain.cinematique.decomposition import (
    gradient_de_vitesse,
    partie_antisymetrique,
    partie_symetrique,
)
from polytech.fluides.domain.cinematique.evolution import image_du_cercle
from polytech.fluides.domain.cinematique.existence import gradients_orthogonaux
from polytech.fluides.domain.cinematique.operateurs import (
    derivee_le_long_de_la_trajectoire,
)
from polytech.fluides.domain.cinematique.particule import (
    allongement_d_une_arete,
    dilatation_relative,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique
from polytech.fluides.domain.statique.bosse import compensation_de_la_bosse
from polytech.fluides.domain.statique.equilibre import (
    forces_sur_la_tranche,
    integration_sur_une_colonne,
    loi_locale,
)
from polytech.fluides.domain.statique.stevin import variation_de_pression

_DENIVELEE = sp.Symbol("h_2", positive=True)
_MILLIMETRE = Hauteur(metres=0.001)
"""Precision de lecture typique sur une echelle de manometre."""


_NOMS_DES_FORCES = ("PousseeDessous", "PousseeDessus", "PoidsTranche")


def _macros_hydrostatique() -> dict[str, str]:
    forces = {
        nom: expression_latex(force)
        for nom, force in zip(_NOMS_DES_FORCES, forces_sur_la_tranche(), strict=True)
    }
    return {
        **forces,
        "LoiLocaleHydrostatique": expression_latex(loi_locale()),
        "IntegrationColonne": expression_latex(integration_sur_une_colonne()),
        "CompensationBosse": expression_latex(compensation_de_la_bosse(_DENIVELEE)),
    }


def _erreur_de_lecture(fluide: MasseVolumique) -> str:
    """Erreur de pression induite par un millimetre d'erreur de lecture."""
    return nombre(variation_de_pression(fluide, _MILLIMETRE).pascals / 1_000, 2)


def _macros_sensibilite() -> dict[str, str]:
    return {
        "ErreurMercureKPa": _erreur_de_lecture(enonce.MERCURE),
        "ErreurEauKPa": _erreur_de_lecture(enonce.EAU),
    }


def _macros_decomposition() -> dict[str, str]:
    u, v = CHAMP_GENERIQUE.u, CHAMP_GENERIQUE.v
    gradient = gradient_de_vitesse(u, v)
    # `factor_terms` sort le 1/2 des coefficients croises : sans lui, le rendu
    # empile une fraction dans une fraction.
    lisible = sp.factor_terms
    return {
        "GradientVitesse": expression_latex(gradient),
        "PartSymetrique": expression_latex(
            partie_symetrique(gradient).applyfunc(lisible)
        ),
        "PartAntisymetrique": expression_latex(
            partie_antisymetrique(gradient).applyfunc(lisible),
        ),
        "AllongementArete": expression_latex(allongement_d_une_arete(u)),
        "DilatationRelative": expression_latex(dilatation_relative(u, v)),
        "DeriveeMaterielleChaine": expression_latex(
            derivee_le_long_de_la_trajectoire(sp.Function("f")(X, Y, T), u, v),
        ),
    }


def _macros_ecoulement_c() -> dict[str, str]:
    champ = enonce.CHAMP_C
    acceleration_x, acceleration_y = champ.acceleration()
    return {
        "ChampAccelerationCX": expression_latex(acceleration_x),
        "ChampAccelerationCY": expression_latex(acceleration_y),
        "EllipseC": expression_latex(image_du_cercle(champ.tenseur_taux_deformation())),
        "ProduitGradientsC": expression_latex(gradients_orthogonaux(champ.u, champ.v)),
        "PotentielPartielC": expression_latex(sp.integrate(champ.u, X)),
    }


def macros_derivations() -> dict[str, str]:
    """Rend les etapes intermediaires citees par le corrige."""
    return {
        **_macros_hydrostatique(),
        **_macros_sensibilite(),
        **_macros_decomposition(),
        **_macros_ecoulement_c(),
    }
