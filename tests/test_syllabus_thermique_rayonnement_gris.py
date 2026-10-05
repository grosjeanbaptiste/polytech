"""Le syllabus de thermique, tome 1 : l'echange total entre surfaces grises.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 137 a
147 (deuxieme partie, chapitre 3, sections 3.1 a 3.6), lues sur les
pages rendues.

Les oracles : les equations (3.1) a (3.18). Le chapitre ne chiffre rien ;
chaque formule fermee est donc confrontee a la methode des zones, qui la
contient, et la methode elle-meme a ses bilans.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.facteurs_corriges import (
    facteur_corrige_a_trois_zones,
)
from polytech.thermique.domain.rayonnement.paroi_adiabatique import (
    EnceinteAParoiAdiabatique,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    SurfaceGrise,
    flux_entre_deux_plans_gris,
    flux_entre_deux_surfaces,
)
from polytech.thermique.domain.rayonnement.zones_grises import (
    EnceinteGrise,
    Zone,
)

DIRECTES = ((0.0, 1.0, 2.0), (1.0, 0.5, 2.5), (2.0, 2.5, 1.5))
"""Trois zones : `S = 3, 4, 6`, la deuxieme et la troisieme concaves."""
ZONES = (Zone(3.0, 0.8), Zone(4.0, 0.5), Zone(6.0, 0.3))
TEMPERATURES = (900.0, 600.0, 400.0)
ENCEINTE = EnceinteGrise(ZONES, DIRECTES)
NOIRS = [emittance(t) for t in TEMPERATURES]


def test_l_exitance_apparente_est_l_emis_plus_le_reflechi() -> None:
    """(3.1) et (3.5) : `J = eps sigma T^4 + rho I`, `S_i I_i = somme ij J_j`."""
    apparentes = ENCEINTE.exitances_apparentes(NOIRS)
    for i, zone in enumerate(ZONES):
        incident = sum(DIRECTES[i][j] * apparentes[j] for j in range(3)) / zone.aire
        attendue = zone.emissivite * NOIRS[i] + (1 - zone.emissivite) * incident
        assert apparentes[i] == pytest.approx(attendue)


def test_les_trois_ecritures_du_flux_net() -> None:
    """(3.2), (3.4) et (3.6) donnent le meme flux, et la somme est nulle."""
    apparentes = ENCEINTE.exitances_apparentes(NOIRS)
    flux = ENCEINTE.flux_nets(TEMPERATURES)
    for i, zone in enumerate(ZONES):
        incident = sum(DIRECTES[i][j] * apparentes[j] for j in range(3)) / zone.aire
        assert flux[i] == pytest.approx(zone.aire * (apparentes[i] - incident))
        surface = zone.aire * zone.emissivite / (1 - zone.emissivite)
        assert flux[i] == pytest.approx(surface * (NOIRS[i] - apparentes[i]))
    assert sum(flux) == pytest.approx(0.0, abs=1e-6)
    assert flux[0] > 0 > flux[2]


def _matrice_d() -> np.ndarray:
    matrice = np.array(DIRECTES)
    for i, zone in enumerate(ZONES):
        matrice[i, i] -= zone.aire / (1 - zone.emissivite)
    return matrice


def test_le_systeme_matriciel_3_7() -> None:
    """`somme J_j (ij - delta_ij S_i / rho_i) = -(S_i eps_i / rho_i) sigma T_i^4`."""
    second = [
        -z.aire * z.emissivite / (1 - z.emissivite) * noir
        for z, noir in zip(ZONES, NOIRS, strict=True)
    ]
    solution = np.linalg.solve(_matrice_d(), second)
    assert ENCEINTE.exitances_apparentes(NOIRS) == pytest.approx(solution)


def test_les_aires_d_echange_total_par_les_mineurs() -> None:
    """(3.8) : `S_iS_j = k_i [k_j (-D_ij / D) - delta_ij eps_j]`, `k = S eps / rho`."""
    inverse = np.linalg.inv(_matrice_d())
    totales = ENCEINTE.aires_d_echange_total()
    for i, zi in enumerate(ZONES):
        for j, zj in enumerate(ZONES):
            ki = zi.aire * zi.emissivite / (1 - zi.emissivite)
            kj = zj.aire * zj.emissivite / (1 - zj.emissivite)
            attendue = ki * (kj * -inverse[j, i] - (zj.emissivite if i == j else 0.0))
            assert totales[i][j] == pytest.approx(attendue)


def test_les_aires_d_echange_total_sont_reciproques_et_bouclent() -> None:
    """(3.10) et page 146 : `S_iS_j = S_jS_i`, `somme_j S_1S_j = S_1 eps_1`."""
    totales = ENCEINTE.aires_d_echange_total()
    for i, zone in enumerate(ZONES):
        assert sum(totales[i]) == pytest.approx(zone.aire * zone.emissivite)
        for j in range(3):
            assert totales[i][j] == pytest.approx(totales[j][i])


def test_le_flux_net_se_recompose_par_les_aires_d_echange_total() -> None:
    """Page 141 : `Phi_i,net = somme S_iS_j sigma (T_i^4 - T_j^4)`."""
    totales = ENCEINTE.aires_d_echange_total()
    flux = ENCEINTE.flux_nets(TEMPERATURES)
    for i in range(3):
        somme = sum(totales[i][j] * (NOIRS[i] - NOIRS[j]) for j in range(3))
        assert flux[i] == pytest.approx(somme)


def test_des_zones_noires_redonnent_les_aires_directes() -> None:
    """Sans reflexion, l'echange total est l'echange direct."""
    noire = EnceinteGrise(tuple(Zone(z.aire, 1.0) for z in ZONES), DIRECTES)
    for ligne, directe in zip(noire.aires_d_echange_total(), DIRECTES, strict=True):
        assert ligne == pytest.approx(directe)


def test_deux_zones_grises() -> None:
    """(3.11) : en serie, `rho_1 / eps_1 S_1`, `1 / 12` et `rho_2 / eps_2 S_2`."""
    directes = ((0.0, 2.0), (2.0, 3.0))
    enceinte = EnceinteGrise((Zone(2.0, 0.6), Zone(5.0, 0.9)), directes)
    resistance = 0.4 / (0.6 * 2.0) + 1 / 2.0 + 0.1 / (0.9 * 5.0)
    attendu = (emittance(700.0) - emittance(300.0)) / resistance
    flux = enceinte.flux_nets((700.0, 300.0))
    assert flux[0] == pytest.approx(attendu)
    assert flux[1] == pytest.approx(-attendu)
    par_le_schema = flux_entre_deux_surfaces(
        SurfaceGrise(2.0, 0.6, 700.0), SurfaceGrise(5.0, 0.9, 300.0), facteur=1.0
    )
    assert par_le_schema == pytest.approx(attendu)


def test_deux_plans_infinis_paralleles() -> None:
    """(3.12) : `F_12 corrige = 1 / (1/eps_1 + 1/eps_2 - 1)`."""
    enceinte = EnceinteGrise((Zone(1.0, 0.6), Zone(1.0, 0.3)), ((0.0, 1.0), (1.0, 0.0)))
    corrige = enceinte.aires_d_echange_total()[0][1]
    assert corrige == pytest.approx(1 / (1 / 0.6 + 1 / 0.3 - 1))
    flux = enceinte.flux_nets((500.0, 300.0))[0]
    assert flux == pytest.approx(
        flux_entre_deux_plans_gris(0.6, 0.3, 500.0, 300.0, 1.0)
    )


def test_les_reflexions_successives_entre_deux_plans() -> None:
    """Page 142 : `eps_1 eps_2 (1 + rho_1 rho_2 + ...)` est la forme (3.12)."""
    eps_1, eps_2 = 0.6, 0.3
    serie = sum(eps_1 * eps_2 * ((1 - eps_1) * (1 - eps_2)) ** n for n in range(200))
    assert serie == pytest.approx(1 / (1 / eps_1 + 1 / eps_2 - 1))


def test_deux_cylindres_coaxiaux() -> None:
    """Page 143 : `S_1S_2 = 1 / (1/(S_1 eps_1) + rho_2/(S_2 eps_2))`."""
    interieur, exterieur = 2.0, 5.0
    directes = ((0.0, interieur), (interieur, exterieur - interieur))
    enceinte = EnceinteGrise((Zone(interieur, 0.7), Zone(exterieur, 0.4)), directes)
    attendue = 1 / (1 / (interieur * 0.7) + 0.6 / (exterieur * 0.4))
    assert enceinte.aires_d_echange_total()[0][1] == pytest.approx(attendue)


def test_un_corps_convexe_dans_un_grand_local() -> None:
    """Page 143 : quand `S_1 / S_2` tend vers zero, le facteur corrige vaut `eps_1`."""
    grand = 1e7
    directes = ((0.0, 1.0), (1.0, grand - 1.0))
    enceinte = EnceinteGrise((Zone(1.0, 0.7), Zone(grand, 0.2)), directes)
    assert enceinte.aires_d_echange_total()[0][1] == pytest.approx(0.7, rel=1e-5)


AVEC_PAROI = ((0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (2.0, 3.0, 1.0))
"""Une source (3 m^2), un recepteur (4 m^2) et une paroi isolee (6 m^2)."""


def _a_paroi(emissivite: float | None) -> EnceinteGrise:
    paroi = Zone.adiabatique(6.0) if emissivite is None else Zone(6.0, emissivite)
    return EnceinteGrise((Zone(3.0, 0.8), Zone(4.0, 0.5), paroi), AVEC_PAROI)


def test_une_surface_adiabatique_ne_fournit_rien() -> None:
    """3.6.1 : `Phi_net,R = 0`, et la source donne ce que le recepteur recoit."""
    flux = _a_paroi(None).flux_nets((900.0, 500.0, 0.0))
    assert flux[2] == pytest.approx(0.0, abs=1e-9)
    assert flux[0] == pytest.approx(-flux[1])
    assert flux[0] > 0


def test_une_surface_adiabatique_est_un_reflecteur_parfait() -> None:
    """Page 144 : les echanges ne dependent pas de `rho_R`.

    Une paroi grise portee a sa temperature d'equilibre n'echange rien,
    quelle que soit son emissivite, et laisse les autres flux inchanges.
    """
    adiabatique = _a_paroi(None)
    equilibre = adiabatique.temperature_d_equilibre(2, (900.0, 500.0, 0.0))
    assert 500.0 < equilibre < 900.0
    reference = adiabatique.flux_nets((900.0, 500.0, 0.0))
    for emissivite in (0.2, 0.9):
        flux = _a_paroi(emissivite).flux_nets((900.0, 500.0, equilibre))
        assert flux[2] == pytest.approx(0.0, abs=1e-6)
        assert flux[0] == pytest.approx(reference[0])


def test_le_systeme_a_trois_zones_par_le_schema() -> None:
    """(3.15) et (3.16) : la paroi ajoute un chemin en parallele du direct."""
    noires = 1.0 + 2.0 * 3.0 / (2.0 + 3.0)
    resistance = 0.2 / (0.8 * 3.0) + 0.5 / (0.5 * 4.0) + 1 / noires
    totale = _a_paroi(None).aires_d_echange_total()[0][1]
    assert totale == pytest.approx(1 / resistance)
    ancienne = EnceinteAParoiAdiabatique(
        3.0, 0.8, 900.0, 4.0, 0.5, 500.0, 1.0, 2.0, 3.0
    )
    assert ancienne.echange_noir() == pytest.approx(noires)
    assert ancienne.echange_gris() == pytest.approx(totale)
    assert ancienne.flux() == pytest.approx(
        _a_paroi(None).flux_nets((900.0, 500.0, 0.0))[0]
    )
    equilibre = _a_paroi(None).temperature_d_equilibre(2, (900.0, 500.0, 0.0))
    assert ancienne.temperature_de_la_paroi() == pytest.approx(equilibre)


def test_les_aires_d_echange_total_sans_la_paroi_bouclent() -> None:
    """Page 146 : `somme_j S_1S_j|R = S_1 eps_1`, sommee hors des adiabatiques."""
    totales = _a_paroi(None).aires_d_echange_total()
    assert totales[0][0] + totales[0][1] == pytest.approx(3.0 * 0.8)
    assert totales[2] == pytest.approx([0.0, 0.0, 0.0])
    assert totales[0][2] == 0.0


def test_le_facteur_d_angle_corrige_a_trois_zones() -> None:
    """(3.17) et (3.18) : `F_12,R = F_12 + F_1R F_R2 / (1 - F_RR)`."""
    geometrique = 1.0 / 3.0 + (2.0 / 3.0) * (3.0 / 6.0) / (1 - 1.0 / 6.0)
    assert geometrique == pytest.approx((1.0 + 2.0 * 3.0 / 5.0) / 3.0)
    corrige = facteur_corrige_a_trois_zones(
        (0.8, 0.5), rapport_des_aires=3.0 / 4.0, geometrique=geometrique
    )
    totale = _a_paroi(None).aires_d_echange_total()[0][1]
    assert corrige == pytest.approx(totale / 3.0)
    with pytest.raises(ValueError, match="emissivite hors de"):
        facteur_corrige_a_trois_zones(
            (0.0, 0.5), rapport_des_aires=1.0, geometrique=0.5
        )


def test_une_enceinte_ouverte_est_refusee() -> None:
    """`somme_j ij = S_i` : sans cela le bilan ne se boucle pas."""
    with pytest.raises(ValueError, match="enceinte non fermee"):
        EnceinteGrise((Zone(2.0, 0.6), Zone(5.0, 0.9)), ((0.0, 1.0), (1.0, 3.0)))
    with pytest.raises(ValueError, match="aires d'echange non symetriques"):
        EnceinteGrise((Zone(2.0, 0.6), Zone(5.0, 0.9)), ((0.0, 2.0), (5.0, 0.0)))
    with pytest.raises(ValueError, match="emissivite hors de"):
        Zone(2.0, 1.5)
    with pytest.raises(ValueError, match="zone non adiabatique"):
        _a_paroi(0.5).temperature_d_equilibre(2, (900.0, 500.0, 0.0))
