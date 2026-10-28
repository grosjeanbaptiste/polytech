"""La methode des zones et les surfaces adiabatiques de la fiche, par `polytech/`."""

from __future__ import annotations

from polytech.maths.domain.comparaison import proches, proches_relativement
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.paroi_adiabatique import (
    EnceinteAParoiAdiabatique,
)
from polytech.thermique.domain.rayonnement.zones_grises import EnceinteGrise, Zone

AIRES = (2.0, 3.0, 1.0)
DIRECTES = ((0.9, 0.7, 0.4), (0.7, 1.7, 0.6), (0.4, 0.6, 0.0))
ZONES = (Zone(2.0, 0.8), Zone(3.0, 0.6), Zone.adiabatique(1.0))
TEMPERATURES = (600.0, 300.0, 0.0)
ENCEINTE = EnceinteGrise(ZONES, DIRECTES)
NOIRS = [emittance(t) for t in TEMPERATURES]
J = ENCEINTE.exitances_apparentes(NOIRS)


def _eclairement(i: int) -> float:
    """`I_i = somme_j F_ij J_j`, avec `S_i F_ij` lu dans les aires d'echange."""
    return sum(DIRECTES[i][j] * J[j] for j in range(3)) / AIRES[i]


def test_l_exitance_apparente_est_emise_plus_reflechie() -> None:
    """`J = eps sigma T^4 + rho I` pour chaque zone qui n'est pas adiabatique."""
    for i in (0, 1):
        eps = ZONES[i].emissivite
        assert proches_relativement(
            J[i], eps * NOIRS[i] + (1 - eps) * _eclairement(i), 1e-9
        )


def test_le_flux_d_une_zone_passe_par_sa_resistance_de_surface() -> None:
    """`Phi_i = eps_i S_i / rho_i (sigma T_i^4 - J_i)`."""
    flux = ENCEINTE.flux_nets(TEMPERATURES)
    for i in (0, 1):
        eps = ZONES[i].emissivite
        fiche = eps * AIRES[i] / (1 - eps) * (NOIRS[i] - J[i])
        assert proches_relativement(flux[i], fiche, 1e-9)


def test_le_flux_d_une_zone_est_la_somme_des_echanges_avec_les_autres() -> None:
    """`Phi_ij = S_i F_ij (J_i - J_j)`, et `Phi_i = somme_j Phi_ij`."""
    flux = ENCEINTE.flux_nets(TEMPERATURES)
    for i in range(3):
        somme = sum(DIRECTES[i][j] * (J[i] - J[j]) for j in range(3) if j != i)
        assert proches(flux[i], somme, 1e-6)


def test_une_surface_noire_confond_ses_deux_noeuds() -> None:
    """`rho = 0` : `J = sigma T^4`."""
    noire = EnceinteGrise((Zone(2.0, 1.0), Zone(3.0, 0.6), ZONES[2]), DIRECTES)
    assert proches_relativement(noire.exitances_apparentes(NOIRS)[0], NOIRS[0], 1e-9)


def test_la_surface_adiabatique_renvoie_tout_et_son_emissivite_ne_joue_pas() -> None:
    """`Phi_net = 0`, `J = sigma T^4`, et `sigma T_3^4 = somme ponderee des J`."""
    assert proches(ENCEINTE.flux_nets(TEMPERATURES)[2], 0.0, 1e-6)
    assert proches_relativement(J[2], _eclairement(2), 1e-9)  # J = eclairement recu
    t3 = ENCEINTE.temperature_d_equilibre(2, TEMPERATURES)
    assert proches_relativement(emittance(t3), J[2], 1e-9)
    ponderee = (DIRECTES[0][2] * J[0] + DIRECTES[1][2] * J[1]) / (
        DIRECTES[0][2] + DIRECTES[1][2]
    )
    assert proches_relativement(emittance(t3), ponderee, 1e-9)


def test_les_deux_modelisations_de_la_paroi_adiabatique_s_accordent() -> None:
    """Le schema a noeud flottant et la reduction en serie donnent la meme `T_3`."""
    paroi = EnceinteAParoiAdiabatique(2.0, 0.8, 600.0, 3.0, 0.6, 300.0, 0.7, 0.4, 0.6)
    attendue = ENCEINTE.temperature_d_equilibre(2, TEMPERATURES)
    assert proches_relativement(paroi.temperature_de_la_paroi(), attendue, 1e-9)
