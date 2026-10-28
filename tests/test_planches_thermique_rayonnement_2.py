"""Les planches de thermique : les echanges radiatifs entre surfaces.

Sources : dans `ThermiqueFondements/SLIDES/`,
`Slides_06_07_Rayonnement_2_Echanges_entre_surfaces.pdf`, 36 planches, et
`Slides_08_Rayonnement_3_Surfaces_grises.pdf`, 10 planches, lues une a une.

Les oracles : les exemples de la planche 6, les deux graphes de la
planche 19, l'individu de la planche 21 (60 W), les relations de la
planche 31, la linearisation de la planche 32 et le quizz du jeu 08.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
    coefficient_de_rayonnement_linearise,
)
from polytech.thermique.domain.rayonnement.cordes_croisees import (
    corde_croisee_de_deux_tubes,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    emittance,
    longueur_d_onde_du_maximum,
)
from polytech.thermique.domain.rayonnement.ecran_de_tubes import (
    aire_entre_deux_tubes,
    facteur_du_plan_vers_la_rangee,
)
from polytech.thermique.domain.rayonnement.enceinte_noire import (
    aire_dans_une_cavite_a_trois,
    flux_entre_surfaces_noires,
    flux_nets_noirs,
)
from polytech.thermique.domain.rayonnement.facteurs_de_forme import Enceinte
from polytech.thermique.domain.rayonnement.vitre_selective import VitreSelective
from polytech.thermique.domain.rayonnement.zones_grises import EnceinteGrise, Zone

MICRON = 1e-6


def test_une_surface_convexe_entouree() -> None:
    """Planche 6 : `F_12 = 1`, `F_21 = S_1 / S_2`, `F_22 = 1 - S_1 / S_2`."""
    interieure, exterieure = 2.0, 5.0
    enceinte = Enceinte(
        surfaces=(interieure, exterieure),
        facteurs={(0, 1): 1.0, (1, 0): 0.4, (1, 1): 0.6},
    )
    assert enceinte.reciproque(0, 1)
    assert enceinte.est_fermee_depuis(0)
    assert enceinte.est_fermee_depuis(1)
    assert enceinte.facteur_deduit(1, 0) == pytest.approx(interieure / exterieure)


def test_trois_surfaces_convexes() -> None:
    """Planche 6 : `12 = (S_1 + S_2 - S_3) / 2`, et les deux autres par permutation."""
    s1, s2, s3 = 3.0, 4.0, 6.0
    douze = aire_dans_une_cavite_a_trois(s1, s2, s3)
    treize = aire_dans_une_cavite_a_trois(s1, s3, s2)
    vingt_trois = aire_dans_une_cavite_a_trois(s2, s3, s1)
    assert (douze + treize, douze + vingt_trois, treize + vingt_trois) == (s1, s2, s3)


@pytest.mark.parametrize(("rapport", "lu"), [(1.0, 0.18), (2.0, 0.08), (5.5, 0.029)])
def test_le_facteur_entre_deux_cylindres_de_la_planche_19(
    rapport: float, lu: float
) -> None:
    """Planche 17 : `F_12 = (1/pi)[sqrt(X^2 - 1) + asin(1/X) - X]`, sans facteur 2.

    La planche le dit en rouge : « Erreur dans le syllabus : pg 131 pas de
    facteur 2 ».
    """
    facteur = aire_entre_deux_tubes(1.0, rapport) / math.pi
    ferme = (corde_croisee_de_deux_tubes(1.0, rapport) - rapport) / math.pi
    assert facteur == pytest.approx(ferme)
    assert facteur == pytest.approx(lu, abs=0.003)


@pytest.mark.parametrize(("rapport", "lu"), [(1.0, 1.0), (2.0, 0.66), (5.5, 0.27)])
def test_le_facteur_du_plan_vers_l_ecran_de_la_planche_19(
    rapport: float, lu: float
) -> None:
    """Planche 18 : `F_PT = 1 - sqrt(1 - (d/l)^2) + (d/l) acos(d/l)`."""
    inverse = 1 / rapport
    ferme = 1 - math.sqrt(1 - inverse**2) + inverse * math.acos(inverse)
    facteur = facteur_du_plan_vers_la_rangee(1.0, rapport)
    assert facteur == pytest.approx(ferme)
    assert facteur == pytest.approx(lu, abs=0.01)


def test_l_individu_dans_une_piece_noire_perd_60_watts() -> None:
    """Planche 21 : 1,8 m^2, `F_12 = 0,8`, 27 et 20 degres : 59,6 W."""
    flux = flux_entre_surfaces_noires(1.8 * 0.8, 273.0 + 27.0, 273.0 + 20.0)
    assert flux == pytest.approx(59.6, abs=0.15)
    directes = ((1.8 * 0.2, 1.8 * 0.8), (1.8 * 0.8, 30.0))
    nets = flux_nets_noirs(directes, (300.0, 293.0))
    assert nets[0] == pytest.approx(flux)


def test_le_meme_individu_dans_une_piece_grise() -> None:
    """Planches 36 du jeu 06 et 3 du jeu 08 : la question, sans reponse imprimee.

    Entre `eps_1 eps_2 F_12` et `eps_1` fois l'aire, comme la planche 31
    le borne, et moins que dans la piece noire.
    """
    murs = 60.0
    directes = ((1.8 * 0.2, 1.8 * 0.8), (1.8 * 0.8, murs - 1.8 * 0.8))
    piece = EnceinteGrise((Zone(1.8, 0.95), Zone(murs, 0.9)), directes)
    totales = piece.aires_d_echange_total()
    corrige = totales[0][1] / 1.8
    assert 0.95 * 0.9 * 0.8 <= corrige <= 0.95
    flux = piece.flux_nets((300.0, 293.0))[0]
    assert flux == pytest.approx(totales[0][1] * (emittance(300.0) - emittance(293.0)))
    assert flux < 59.6


def test_les_facteurs_corriges_se_ferment_sur_l_emissivite() -> None:
    """Planche 31 : `somme_j F_ij corrige = eps_i`, et la reciprocite."""
    directes = ((0.0, 1.0, 2.0), (1.0, 0.5, 2.5), (2.0, 2.5, 1.5))
    zones = (Zone(3.0, 0.8), Zone(4.0, 0.5), Zone(6.0, 0.3))
    totales = EnceinteGrise(zones, directes).aires_d_echange_total()
    for i, zone in enumerate(zones):
        assert sum(totales[i]) / zone.aire == pytest.approx(zone.emissivite)
        for j, autre in enumerate(zones):
            assert totales[i][j] == pytest.approx(totales[j][i])
            if i != j:
                direct = zone.emissivite * autre.emissivite * directes[i][j]
                assert totales[i][j] >= direct


def test_un_reflecteur_parfait_oppose_une_resistance_infinie() -> None:
    """Planche 23 : `rho / (eps S)` ; le flux net tend vers zero avec `eps`."""
    directes = ((0.0, 1.0), (1.0, 0.0))
    flux = [
        EnceinteGrise((Zone(1.0, 0.9), Zone(1.0, eps)), directes).flux_nets(
            (600.0, 300.0)
        )[0]
        for eps in (0.5, 0.05, 0.0005)
    ]
    assert flux == sorted(flux, reverse=True)
    assert flux[-1] < flux[0] / 500


def test_le_coefficient_de_transfert_radiatif_et_sa_linearisation() -> None:
    """Planche 32 : `K_r = F sigma (T_i^4 - T_p^4)/(T_i - T_p) ~ 4 F sigma T_m^3`."""
    exact = coefficient_de_rayonnement_exact(0.9, 300.0, 293.0)
    attendu = 0.9 * (emittance(300.0) - emittance(293.0)) / 7.0
    assert exact == pytest.approx(attendu)
    lineaire = coefficient_de_rayonnement_linearise(0.9, 300.0, 293.0)
    assert lineaire == pytest.approx(exact, rel=1e-3)
    loin = coefficient_de_rayonnement_linearise(0.9, 900.0, 293.0)
    assert loin < 0.85 * coefficient_de_rayonnement_exact(0.9, 900.0, 293.0)


def test_le_bilan_d_un_individu_couple_les_modes() -> None:
    """Planche 32 : `Phi_i = K_c S (T_i - T_a) + S_iS_p sigma (T_i^4 - T_p^4)`.

    Quand l'air et les parois sont a la meme temperature, les deux modes
    s'ajoutent en un seul coefficient, `K_c + K_r`.
    """
    surface, convection, emissivite = 1.8, 3.0, 0.95
    par_convection = convection * surface * (306.0 - 293.0)
    par_rayonnement = flux_entre_surfaces_noires(surface * emissivite, 306.0, 293.0)
    rayonne = coefficient_de_rayonnement_exact(emissivite, 306.0, 293.0)
    ensemble = surface * (convection + rayonne) * (306.0 - 293.0)
    assert par_convection + par_rayonnement == pytest.approx(ensemble)
    assert rayonne > convection


def test_le_soleil_se_voit_et_non_le_corps_humain() -> None:
    """Jeu 08, planche 6 : `lambda_max T = 2898 um K`, le visible de 0,38 a 0,78 um."""
    soleil = longueur_d_onde_du_maximum(5780.0) / MICRON
    corps = longueur_d_onde_du_maximum(310.0) / MICRON
    assert 0.38 < soleil < 0.78
    assert corps == pytest.approx(9.3, abs=0.1)
    assert corps > 10 * 0.78


def test_la_serre_laisse_entrer_et_ne_laisse_pas_sortir() -> None:
    """Jeu 08, planche 10 : la vitre transmet de 0,35 a 3 um, et plus rien au-dela."""
    vitre = VitreSelective(transmission_courte=0.85, reflexion=0.08, coupure=3 * MICRON)
    assert vitre.transmission(5780.0) > 0.8
    assert vitre.transmission(300.0) < 1e-3
    assert vitre.absorption(300.0) == pytest.approx(0.92, abs=1e-3)
