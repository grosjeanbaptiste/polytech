"""Le syllabus de thermique, tome 1 : les applications des surfaces grises.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 147 a
154 (deuxieme partie, chapitre 3, sections 3.7 a 3.10), lues sur les
pages rendues.

Les oracles : les ecrans thermiques de la section 3.7, l'equation (3.19)
et les courbes 3 a 6 de la figure 2.15, le plan gris fictif (3.20), et le
four tunnel de la section 3.10 -- chacun confronte a la methode des zones
ou a la regle des elastiques.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

from polytech.thermique.domain.rayonnement.cordes_croisees import (
    aire_par_les_cordes,
    corde_tendue,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    STEFAN_BOLTZMANN,
    emittance,
)
from polytech.thermique.domain.rayonnement.ecran_de_tubes import (
    facteur_du_plan_vers_la_rangee,
)
from polytech.thermique.domain.rayonnement.ecran_de_tubes_adosse import (
    facteur_corrige_de_l_ecran,
    facteur_vers_l_ecran_adosse,
    facteurs_vers_deux_rangees_adossees,
)
from polytech.thermique.domain.rayonnement.ecrans_thermiques import (
    flux_a_travers_des_ecrans,
    flux_avec_un_ecran_different,
    temperature_de_l_ecran,
    temperature_lue_par_la_canne,
)
from polytech.thermique.domain.rayonnement.four_tunnel import FourTunnel
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)
from polytech.thermique.domain.rayonnement.zones_grises import EnceinteGrise, Zone


def test_un_ecran_identique_divise_le_flux_par_deux() -> None:
    """3.7 : `T_3^4 = (T_1^4 + T_2^4) / 2`, et le flux est moitie moindre."""
    sans = flux_entre_deux_plans_gris(0.8, 0.8, 800.0, 400.0, 1.0)
    assert flux_a_travers_des_ecrans(0.8, 800.0, 400.0, ecrans=0) == pytest.approx(sans)
    avec = flux_a_travers_des_ecrans(0.8, 800.0, 400.0, ecrans=1)
    assert avec == pytest.approx(sans / 2)
    ecran = temperature_de_l_ecran(800.0, 400.0)
    assert ecran**4 == pytest.approx((800.0**4 + 400.0**4) / 2)
    amont = flux_entre_deux_plans_gris(0.8, 0.8, 800.0, ecran, 1.0)
    aval = flux_entre_deux_plans_gris(0.8, 0.8, ecran, 400.0, 1.0)
    assert amont == pytest.approx(aval)
    assert amont == pytest.approx(avec)


def test_n_ecrans_divisent_le_flux_par_n_plus_un() -> None:
    """`Phi_n = Phi_0 / (n + 1)`."""
    sans = flux_a_travers_des_ecrans(0.6, 700.0, 300.0, ecrans=0)
    for nombre in (2, 5, 20):
        avec = flux_a_travers_des_ecrans(0.6, 700.0, 300.0, ecrans=nombre)
        assert avec == pytest.approx(sans / (nombre + 1))
    with pytest.raises(ValueError, match="nombre d'ecrans negatif"):
        flux_a_travers_des_ecrans(0.6, 700.0, 300.0, ecrans=-1)


def test_un_ecran_d_une_autre_nature() -> None:
    """Page 148 : `Phi_1 / Phi_0 = sigma'' / (2 sigma')` ; l'aluminium poli gagne."""
    sans = flux_a_travers_des_ecrans(0.8, 800.0, 400.0, ecrans=0)
    brillant = flux_avec_un_ecran_different(0.8, 0.05, 800.0, 400.0)
    seconde = STEFAN_BOLTZMANN / (1 / 0.8 + 1 / 0.05 - 1)
    prime = STEFAN_BOLTZMANN / (2 / 0.8 - 1)
    assert brillant / sans == pytest.approx(seconde / (2 * prime))
    identique = flux_avec_un_ecran_different(0.8, 0.8, 800.0, 400.0)
    assert identique == pytest.approx(sans / 2)
    assert brillant < identique / 10


def test_l_ecran_par_la_methode_des_zones() -> None:
    """L'ecran est une paroi adiabatique entre deux enceintes a deux plans."""
    directes = ((0.0, 1.0), (1.0, 0.0))
    ecran = temperature_de_l_ecran(800.0, 400.0)
    amont = EnceinteGrise((Zone(1.0, 0.8), Zone(1.0, 0.05)), directes)
    flux = amont.flux_nets((800.0, ecran))[0]
    assert flux == pytest.approx(flux_avec_un_ecran_different(0.8, 0.05, 800.0, 400.0))


def test_la_canne_pyrometrique_lit_trop_bas() -> None:
    """Page 149 : `K S_c (t_g - t_c) = S_cS_p sigma (T_c^4 - T_p^4)`."""
    gaz, paroi = 1200.0, 600.0
    lue = temperature_lue_par_la_canne(gaz, paroi, convection=80.0, emissivite=0.8)
    assert paroi < lue < gaz
    convecte = 80.0 * (gaz - lue)
    rayonne = 0.8 * (emittance(lue) - emittance(paroi))
    assert convecte == pytest.approx(rayonne)
    brillante = temperature_lue_par_la_canne(
        gaz, paroi, convection=80.0, emissivite=0.1
    )
    assert lue < brillante < gaz
    ventilee = temperature_lue_par_la_canne(
        gaz, paroi, convection=800.0, emissivite=0.8
    )
    assert lue < ventilee < gaz


@pytest.mark.parametrize(
    ("rapport", "lue"), [(2.0, 0.88), (3.0, 0.72), (4.0, 0.59), (7.0, 0.38)]
)
def test_la_courbe_3_de_la_figure_2_15(rapport: float, lue: float) -> None:
    """3.8 : `F_PT,R = F_PT (2 - F_PT)`, l'ecran adosse a une paroi adiabatique."""
    direct = facteur_du_plan_vers_la_rangee(1.0, rapport)
    adosse = facteur_vers_l_ecran_adosse(1.0, rapport)
    assert adosse == pytest.approx(direct * (2 - direct))
    assert adosse == pytest.approx(direct + (1 - direct) * direct)
    assert adosse == pytest.approx(lue, abs=0.015)


def _ecran_en_zones(diametre: float, pas: float, emissivite: float) -> EnceinteGrise:
    """Par metre carre de plan : le plan noir, les tubes, la paroi adiabatique."""
    direct = facteur_du_plan_vers_la_rangee(diametre, pas)
    tubes = math.pi * diametre / pas
    directes = (
        (0.0, direct, 1 - direct),
        (direct, tubes - 2 * direct, direct),
        (1 - direct, direct, 0.0),
    )
    zones = (Zone(1.0, 1.0), Zone(tubes, emissivite), Zone.adiabatique(1.0))
    return EnceinteGrise(zones, directes)


def test_l_ecran_adosse_par_la_methode_des_zones() -> None:
    """(3.16) applique aux tubes noirs redonne la courbe 3."""
    totale = _ecran_en_zones(0.05, 0.1, 1.0).aires_d_echange_total()[0][1]
    assert totale == pytest.approx(facteur_vers_l_ecran_adosse(0.05, 0.1))


def test_le_facteur_corrige_d_un_ecran_de_tubes_gris() -> None:
    """(3.19) : `1 / ((rho_T/eps_T)(l/(pi d)) + 1/F_PT,R)`, contre les zones."""
    corrige = facteur_corrige_de_l_ecran(0.05, 0.1, emissivite=0.6)
    adosse = facteur_vers_l_ecran_adosse(0.05, 0.1)
    attendu = 1 / (0.4 / 0.6 * 0.1 / (math.pi * 0.05) + 1 / adosse)
    assert corrige == pytest.approx(attendu)
    totale = _ecran_en_zones(0.05, 0.1, 0.6).aires_d_echange_total()[0][1]
    assert corrige == pytest.approx(totale)
    assert facteur_corrige_de_l_ecran(0.05, 0.1, emissivite=1.0) == pytest.approx(
        adosse
    )
    with pytest.raises(ValueError, match="emissivite hors de"):
        facteur_corrige_de_l_ecran(0.05, 0.1, emissivite=0.0)


@pytest.mark.parametrize(
    ("rapport", "lues"),
    [
        (2.0, (0.71, 0.31, 0.98)),
        (3.0, (0.55, 0.365, 0.92)),
        (7.0, (0.315, 0.275, 0.61)),
    ],
)
def test_les_courbes_4_a_6_de_la_figure_2_15(
    rapport: float, lues: tuple[float, float, float]
) -> None:
    """Deux rangees adossees : la premiere, la seconde, et les deux.

    Chaque rangee arrete la fraction `F` de ce qui lui arrive, d'ou
    qu'il vienne ; l'abaque est retrouve a 0,03 pres.
    """
    premiere, seconde = facteurs_vers_deux_rangees_adossees(1.0, rapport)
    direct = facteur_du_plan_vers_la_rangee(1.0, rapport)
    assert premiere + seconde == pytest.approx(1 - (1 - direct) ** 4)
    assert premiere == pytest.approx(lues[0], abs=0.03)
    assert seconde == pytest.approx(lues[1], abs=0.03)
    assert premiere + seconde == pytest.approx(lues[2], abs=0.03)


def test_deux_rangees_adossees_par_la_methode_des_zones() -> None:
    """Le plan, deux rangees noires et la paroi : les memes echanges totaux."""
    direct = facteur_du_plan_vers_la_rangee(1.0, 2.0)
    passe = 1 - direct
    tubes = math.pi / 2.0
    seconde_directe = direct * passe
    entre_rangees = tubes - direct - seconde_directe
    directes = (
        (0.0, direct, seconde_directe, passe**2),
        (direct, 0.0, entre_rangees, seconde_directe),
        (seconde_directe, entre_rangees, 0.0, direct),
        (passe**2, seconde_directe, direct, 0.0),
    )
    zones = (Zone(1.0, 1.0), Zone(tubes, 1.0), Zone(tubes, 1.0), Zone.adiabatique(1.0))
    totales = EnceinteGrise(zones, directes).aires_d_echange_total()[0]
    attendues = facteurs_vers_deux_rangees_adossees(1.0, 2.0)
    assert (totales[1], totales[2]) == pytest.approx(attendues)


FOUR = FourTunnel(
    source=1.2, charge=1.5, demi_largeur=0.6, remplissage=0.8, emissivites=(0.7, 0.5)
)


def test_les_aires_directes_du_four_par_les_elastiques() -> None:
    """Page 153 : `12`, `11`, `1R` et `2R`, pour un metre de four."""
    haut_1, bas_1 = (0.6, 0.0), (0.6, -1.2)
    haut_2, bas_2 = (0.0, 0.0), (0.0, -1.5)
    croisees = (corde_tendue(haut_1, bas_2), corde_tendue(bas_1, haut_2))
    droites = (corde_tendue(haut_1, haut_2), corde_tendue(bas_1, bas_2))
    un_cote = aire_par_les_cordes(croisees, droites)
    directes = FOUR.aires_directes()
    assert directes["12"] == pytest.approx(0.8 * 2 * un_cote)
    en_face = aire_par_les_cordes((2 * math.hypot(1.2, 1.2),), (2 * 1.2,))
    assert directes["11"] == pytest.approx(2 * 0.2 * en_face)
    assert directes["1R"] == pytest.approx(2 * 1.2 - directes["11"] - directes["12"])
    assert directes["2R"] == pytest.approx(2 * 1.5 * 0.8 - directes["12"])


def test_l_aire_d_echange_total_du_four_par_les_zones() -> None:
    """Page 153 : les deux resistances de surface et le relais par les parois."""
    directes = FOUR.aires_directes()
    source, charge = 2 * 1.2, 2 * 1.5 * 0.8
    parois = directes["1R"] + directes["2R"]
    matrice = (
        (directes["11"], directes["12"], directes["1R"]),
        (directes["12"], 0.0, directes["2R"]),
        (directes["1R"], directes["2R"], 0.0),
    )
    zones = (Zone(source, 0.7), Zone(charge, 0.5), Zone.adiabatique(parois))
    totale = EnceinteGrise(zones, matrice).aires_d_echange_total()[0][1]
    assert FOUR.aire_d_echange_total() == pytest.approx(totale)
    relais = 1 / (1 / directes["1R"] + 1 / directes["2R"])
    resistance = 0.3 / (0.7 * source) + 0.5 / (0.5 * charge)
    resistance += 1 / (directes["12"] + relais)
    assert FOUR.aire_d_echange_total() == pytest.approx(1 / resistance)


def test_la_longueur_du_four_integre_l_echauffement() -> None:
    """Page 154 : `L = (q_m c / (S_1S_2 sigma)) int dT_2 / (T_1^4 - T_2^4)`."""
    debit, source = 900.0, 1300.0
    entree, sortie = 293.15, 1100.0
    longueur = FOUR.longueur(debit, source, entree=entree, sortie=sortie)

    def integrande(charge: float) -> float:
        return 1 / (source**4 - charge**4)

    integrale, _ = quad(integrande, entree, sortie)
    echange = FOUR.aire_d_echange_total() * STEFAN_BOLTZMANN
    assert longueur == pytest.approx(debit * integrale / echange)
    assert FOUR.longueur(
        2 * debit, source, entree=entree, sortie=sortie
    ) == pytest.approx(2 * longueur)
    with pytest.raises(ValueError, match="la charge ne depasse pas la source"):
        FOUR.longueur(debit, source, entree=entree, sortie=1300.0)


def test_un_four_dont_la_charge_est_plus_courte_que_la_source_est_refuse() -> None:
    """Le syllabus pose `L_2 > L_1` : la formule de `12` en depend."""
    with pytest.raises(ValueError, match="charge plus courte que la source"):
        FourTunnel(1.5, 1.2, 0.6, 0.8, (0.7, 0.5))
    with pytest.raises(ValueError, match="remplissage hors de"):
        FourTunnel(1.2, 1.5, 0.6, 1.2, (0.7, 0.5))
