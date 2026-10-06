"""Macros du TP 7 d'electronique, exercices 3.1 a 3.4 : les echelles LC.

Chaque valeur est rendue dans l'unite ou le document la compose --
microhenrys, picofarads, megahertz -- et sans arrondi intermediaire.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.electronique.application import echelles_tp_07 as ec
from polytech.electronique.application.formatage import nombre
from polytech.electronique.domain.chebyshev import elements_de_chebyshev

if TYPE_CHECKING:
    from polytech.electronique.domain.echelle_de_branches import EchelleTerminee

_MEGA = 2 * math.pi * 1e6


def _valeurs(filtre: EchelleTerminee) -> list[float]:
    """Les inductances et capacites du filtre, dans l'ordre des branches."""
    plates = []
    for branche in filtre.branches:
        plates += [v for v in (branche.inductance, branche.capacite) if v is not None]
    return plates


def _passe_bas() -> dict[str, str]:
    l1, c2 = _valeurs(ec.butterworth_dix_megahertz())
    c1, l2, _ = _valeurs(ec.passe_bas_d_ordre_trois(chebyshev=False))
    cheb_c, cheb_l, _ = _valeurs(ec.passe_bas_d_ordre_trois(chebyshev=True))
    g1, g2, _ = elements_de_chebyshev(ec.ONDULATION, 3)
    return {
        "TpSeptBwDeuxL": nombre(l1 * 1e6, 3),
        "TpSeptBwDeuxC": nombre(c2 * 1e12, 1),
        "TpSeptChargePaire": nombre(ec.charges_de_chebyshev(2)[2], 4),
        "TpSeptBwTroisC": nombre(c1 * 1e9, 1),
        "TpSeptBwTroisL": nombre(l2 * 1e6, 0),
        "TpSeptChTroisC": nombre(cheb_c * 1e9, 1),
        "TpSeptChTroisL": nombre(cheb_l * 1e6, 1),
        "TpSeptGUn": nombre(g1, 4),
        "TpSeptGDeux": nombre(g2, 4),
        "TpSeptEpsilon": nombre(math.sqrt(10 ** (ec.ONDULATION / 10) - 1), 4),
        "TpSeptRapportCheb": nombre(ec.rapport_de_coupure_du_chebyshev(3), 3),
    }


def _passe_bande() -> dict[str, str]:
    l1, c1, l2, c2, _, _ = _valeurs(ec.passe_bande())
    centre = ec.centre_geometrique()
    _, geo_c1, geo_l2, _, _, _ = _valeurs(ec.passe_bande(centre))
    bas, haut = ec.bords_de_la_bande()
    return {
        "TpSeptBandeLUn": nombre(l1 * 1e9, 2),
        "TpSeptBandeCUn": nombre(c1 * 1e15, 1),
        "TpSeptBandeLDeux": nombre(l2 * 1e9, 3),
        "TpSeptBandeCDeux": nombre(c2 * 1e12, 2),
        "TpSeptBordBas": nombre(bas / _MEGA, 1),
        "TpSeptBordHaut": nombre(haut / _MEGA, 1),
        "TpSeptPerteBord": nombre(ec.perte_a_huit_cents_megahertz(), 2),
        "TpSeptCentreGeo": nombre(centre / _MEGA, 1),
        "TpSeptGeoCUn": nombre(geo_c1 * 1e15, 1),
        "TpSeptGeoLDeux": nombre(geo_l2 * 1e9, 3),
    }


def _passe_haut() -> dict[str, str]:
    officiel, autre = ec.passe_haut(), ec.passe_haut(prototype_en_serie=False)
    c1, l2 = _valeurs(officiel)
    autre_l, autre_c = _valeurs(autre)
    g1, g2 = elements_de_chebyshev(ec.ONDULATION, 2)
    return {
        "TpSeptHautGUn": nombre(g1, 4),
        "TpSeptHautGDeux": nombre(g2, 4),
        "TpSeptHautC": nombre(c1 * 1e12, 2),
        "TpSeptHautL": nombre(l2 * 1e9, 1),
        "TpSeptHautCharge": nombre(officiel.charge, 2),
        "TpSeptAutreL": nombre(autre_l * 1e9, 2),
        "TpSeptAutreC": nombre(autre_c * 1e12, 2),
        "TpSeptAutreCharge": nombre(autre.charge, 2),
    }


def macros_des_echelles() -> dict[str, str]:
    """Les valeurs des quatre exercices sur les filtres LC en echelle."""
    return {**_passe_bas(), **_passe_bande(), **_passe_haut()}
