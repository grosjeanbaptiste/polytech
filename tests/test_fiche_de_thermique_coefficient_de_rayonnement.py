"""Le coefficient de rayonnement de la fiche de thermique, refait par `polytech/`."""

from __future__ import annotations

from polytech.maths.domain.comparaison import eloignees, proches_relativement
from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
    coefficient_de_rayonnement_linearise,
)
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.facteurs_corriges import (
    facteur_corrige_a_trois_zones,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    SurfaceGrise,
    flux_entre_deux_surfaces,
)

SIGMA_T = emittance(1.0)
T1, T2 = 600.0, 300.0


def test_le_coefficient_de_rayonnement_est_f_sigma_dt4_sur_dt() -> None:
    """`K_r = F sigma (T_p^4 - T_a^4)/(T_p - T_a) ~ 4 F sigma T_m^3`, `F = eps`."""
    eps, tp, ta = 0.85, 330.0, 290.0
    fiche = eps * SIGMA_T * (tp**4 - ta**4) / (tp - ta)
    assert proches_relativement(
        coefficient_de_rayonnement_exact(eps, tp, ta), fiche, 1e-12
    )
    lineaire = 4 * eps * SIGMA_T * ((tp + ta) / 2) ** 3
    assert proches_relativement(
        coefficient_de_rayonnement_linearise(eps, tp, ta), lineaire, 1e-12
    )
    assert proches_relativement(lineaire, fiche, 0.01)
    face_a_un_grand_local = flux_entre_deux_surfaces(
        SurfaceGrise(1.0, eps, tp), SurfaceGrise(1e9, 1.0, ta), 1.0
    )
    assert proches_relativement(face_a_un_grand_local / (tp - ta), fiche, 1e-6)


def test_dans_une_grande_enceinte_son_emissivite_ne_joue_plus() -> None:
    """`S_1/S_2 -> 0` : reste `eps_1 S_1` ; une feuille peu emissive reduit."""
    s1, grande = 2.0, 1e9
    flux = flux_entre_deux_surfaces(
        SurfaceGrise(s1, 0.7, T1), SurfaceGrise(grande, 0.1, T2), 1.0
    )
    assert proches_relativement(flux, 0.7 * s1 * SIGMA_T * (T1**4 - T2**4), 1e-6)
    sans = facteur_corrige_a_trois_zones(
        (0.8, 0.8), rapport_des_aires=1.0, geometrique=0.6
    )
    avec = facteur_corrige_a_trois_zones(
        (0.8, 0.05), rapport_des_aires=1.0, geometrique=0.6
    )
    assert eloignees(sans, avec, 0.1)
    assert avec < sans
