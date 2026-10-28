"""Les echanges radiatifs de la fiche de thermique, refaits par `polytech/`."""

from __future__ import annotations

import math

from polytech.maths.domain.comparaison import proches, proches_relativement
from polytech.thermique.domain.rayonnement.cordes_croisees import (
    aire_par_les_cordes,
    corde_tendue,
)
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.facteurs_de_forme import (
    Enceinte,
    aire_d_echange_direct,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    SurfaceGrise,
    flux_entre_deux_plans_gris,
    flux_entre_deux_surfaces,
)

SIGMA_T = emittance(1.0)
T1, T2 = 600.0, 300.0


def test_le_flux_noir_est_s1_f12_sigma_t1_4_moins_t2_4() -> None:
    """`Phi_12 = S_1 F_12 sigma (T_1^4 - T_2^4)` ; surfaces noires."""
    s1, f12 = 2.0, 0.4
    noire_1, noire_2 = SurfaceGrise(s1, 1.0, T1), SurfaceGrise(5.0, 1.0, T2)
    obtenu = flux_entre_deux_surfaces(noire_1, noire_2, f12)
    assert proches_relativement(
        obtenu, aire_d_echange_direct(s1, f12) * SIGMA_T * (T1**4 - T2**4), 1e-12
    )


def test_la_reciprocite_et_la_somme_des_facteurs() -> None:
    """`S_1 F_12 = S_2 F_21` et `somme_j F_1j = 1` ; `F_11 = 0` pour un convexe."""
    enceinte = Enceinte(
        (1.0, 4.0), {(0, 1): 1.0, (1, 0): 0.25, (0, 0): 0.0, (1, 1): 0.75}
    )
    assert enceinte.reciproque(0, 1)
    assert enceinte.est_fermee_depuis(0)
    assert enceinte.est_fermee_depuis(1)
    assert proches(enceinte.facteur_deduit(0, 1), 1.0, 1e-12)


def test_les_cordes_croisees_donnent_le_double_de_l_aire_d_echange() -> None:
    """Deux bandes jointives, perpendiculaires, de largeur 1 : `(2 - racine 2) / 2`."""
    croisees = [corde_tendue((0, 0), (0, 1)), corde_tendue((0, 0), (1, 0))]
    non_croisees = [corde_tendue((0, 1), (1, 0))]
    assert proches(
        aire_par_les_cordes(croisees, non_croisees), (2 - math.sqrt(2)) / 2, 1e-12
    )


def test_la_cavite_a_trois_surfaces_est_une_demi_somme() -> None:
    """`S_1 F_12 = (S_1 + S_2 - S_3) / 2` sur un triangle 3-4-5."""
    a, b, c = (0.0, 0.0), (3.0, 0.0), (0.0, 4.0)
    s1, s2, s3 = corde_tendue(a, b), corde_tendue(a, c), corde_tendue(b, c)
    assert proches(aire_par_les_cordes([s1, s2], [s3]), (s1 + s2 - s3) / 2, 1e-12)
    assert proches(aire_par_les_cordes([s1, s2], [s3]), 1.0, 1e-12)


def test_deux_zones_grises_sont_trois_resistances_en_serie() -> None:
    """`Phi = sigma dT4 / (rho_1/(eps_1 S_1) + 1/(S_1 F_12) + rho_2/(eps_2 S_2))`."""
    (e1, s1), (e2, s2), f12 = (0.8, 1.5), (0.5, 3.0), 0.6
    fiche = (
        SIGMA_T
        * (T1**4 - T2**4)
        / ((1 - e1) / (e1 * s1) + 1 / (s1 * f12) + (1 - e2) / (e2 * s2))
    )
    obtenu = flux_entre_deux_surfaces(
        SurfaceGrise(s1, e1, T1), SurfaceGrise(s2, e2, T2), f12
    )
    assert proches_relativement(obtenu, fiche, 1e-12)


def test_les_plans_paralleles_et_le_corps_entoure() -> None:
    """Plans : `1/e1 + 1/e2 - 1` ; corps entoure : `1/e1 + (S1/S2)(1/e2 - 1)`."""
    e1, e2, s1, s2 = 0.7, 0.4, 2.0, 8.0
    ecart = SIGMA_T * (T1**4 - T2**4)
    plans = flux_entre_deux_plans_gris(e1, e2, T1, T2, s1)
    assert proches_relativement(plans, s1 * ecart / (1 / e1 + 1 / e2 - 1), 1e-12)
    entoure = flux_entre_deux_surfaces(
        SurfaceGrise(s1, e1, T1), SurfaceGrise(s2, e2, T2), 1.0
    )
    assert proches_relativement(
        entoure, s1 * ecart / (1 / e1 + (s1 / s2) * (1 / e2 - 1)), 1e-12
    )
