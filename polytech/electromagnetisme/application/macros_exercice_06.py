"""Macros de l'exercice 6 : l'inductance d'un circuit magnetique.

Les trois etapes sont publiees avec leurs termes, et deux publications
meritent leur raison.

Le **rapport des rayons** est publie a part. Il explique pourquoi
`H_T = 64` A/m est exact et non arrondi : les `2 pi` se simplifient et il
ne reste que `16/17,5 = 32/35`, qui multiplie `70` en un entier. Sans ce
rapport, le document ne pourrait que constater la coincidence.

Les **inductances partielles** le sont aussi -- celles qu'on obtiendrait
en oubliant une branche. Elles donnent l'ordre de grandeur de l'erreur,
et c'est ce qui distingue une faute visible d'une faute insidieuse : sept
fois trop petit se voit, quinze pour cent non.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import enonce_exercice_06 as donnees
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    BrancheFerromagnetique,
    champ_pour_les_amperes_tours,
    inductance_depuis_les_flux,
)

_D = donnees.DECIMALES_AFFICHEES
_F = donnees.DECIMALES_FINES


def _branches() -> tuple[BrancheFerromagnetique, BrancheFerromagnetique, float]:
    """Les deux branches et le champ deduit, calcules une seule fois."""
    branche_h = BrancheFerromagnetique(
        induction=donnees.INDUCTION_H,
        champ_lu=donnees.CHAMP_LU_H,
        section=donnees.SECTION_H,
        longueur=donnees.LONGUEUR_H,
    )
    champ_t = champ_pour_les_amperes_tours(
        branche_h.amperes_tours(),
        donnees.LONGUEUR_T,
    )
    branche_t = BrancheFerromagnetique(
        induction=donnees.INDUCTION_T,
        champ_lu=champ_t,
        section=donnees.SECTION_T,
        longueur=donnees.LONGUEUR_T,
    )
    return branche_h, branche_t, champ_t


def macros_donnees_six() -> dict[str, str]:
    """Les donnees de l'enonce, pour que le texte ne les tape pas."""
    return {
        "SixSpires": nombre(donnees.SPIRES, 0),
        "SixInductionH": nombre(donnees.INDUCTION_H, 1),
        "SixChampLuH": nombre(donnees.CHAMP_LU_H, 0),
        "SixRayonH": nombre(donnees.RAYON_H * 100, _D),
        "SixRayonT": nombre(donnees.RAYON_T * 100, _D),
        "SixInductionT": nombre(donnees.INDUCTION_T, 1),
    }


def macros_exercice_six() -> dict[str, str]:
    """Les trois etapes, leurs termes, et les deux inductances partielles."""
    branche_h, branche_t, champ_t = _branches()
    courant = branche_h.courant_pour(donnees.SPIRES)
    flux = (branche_t.flux(), branche_h.flux())
    complete = inductance_depuis_les_flux(donnees.SPIRES, flux, courant)
    sans_h = inductance_depuis_les_flux(donnees.SPIRES, (flux[0],), courant)
    sans_t = inductance_depuis_les_flux(donnees.SPIRES, (flux[1],), courant)
    return {
        "SixSectionH": nombre(donnees.SECTION_H * 1e4, 0),
        "SixSectionT": nombre(donnees.SECTION_T * 1e4, 0),
        "SixFluxH": nombre(branche_h.flux() * 1e3, _D),
        "SixFluxT": nombre(branche_t.flux() * 1e3, _D),
        "SixFluxTotal": nombre(sum(flux) * 1e3, _D),
        "SixLongueurH": nombre(donnees.LONGUEUR_H, 4),
        "SixLongueurT": nombre(donnees.LONGUEUR_T, 4),
        "SixAmperesTours": nombre(branche_h.amperes_tours(), _D),
        "SixCourant": nombre(courant, _F),
        "SixChampT": nombre(champ_t, _D),
        "SixRapportDesRayons": nombre(donnees.RAYON_H / donnees.RAYON_T, 6),
        "SixInductance": nombre(complete, _D),
        "SixInductanceSansH": nombre(sans_h, _D),
        "SixInductanceSansT": nombre(sans_t, _D),
        "SixFacteurOubliH": nombre(complete / sans_h, _D),
        "SixFacteurOubliT": nombre(complete / sans_t, _D),
    }
