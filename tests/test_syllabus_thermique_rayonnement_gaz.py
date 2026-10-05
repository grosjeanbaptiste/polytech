"""Le syllabus de thermique, tome 1 : le rayonnement des gaz et des flammes.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 155 a
177 (deuxieme partie, chapitre 4), lues sur les pages rendues.

Les oracles : les equations (4.1) a (4.26), et les deux exercices -- le
faisceau d'evaporateur (664 W/m) et l'evaporateur-ecran (25 113 W/m).
Les facteurs d'emission lus sur les abaques 4.7 et 4.8 sont recopies du
syllabus : la bibliotheque ne lit pas un abaque.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

from polytech.chemistry.domain.gaz import pression_partielle
from polytech.chemistry.domain.reactions import est_equilibree
from polytech.thermique.domain.rayonnement.corps_noir import (
    emittance,
    fraction_emise_sous,
)
from polytech.thermique.domain.rayonnement.ecran_de_tubes_adosse import (
    facteur_corrige_de_l_ecran,
    facteur_vers_l_ecran_adosse,
)
from polytech.thermique.domain.rayonnement.enceinte_noire import (
    flux_entre_surfaces_noires,
)
from polytech.thermique.domain.rayonnement.volume_gazeux import (
    absorption_d_une_sphere,
    aire_d_echange_avec_le_gaz,
    facteur_d_emission_du_melange,
    facteur_d_emission_hemispherique,
    flux_entre_le_gaz_et_son_enveloppe,
    rayon_equivalent,
    rayon_moyen,
)
from polytech.thermique.domain.rayonnement.zones_grises import EnceinteGrise, Zone
from polytech.thermique.domain.rayonnement_des_gaz import (
    absorptivite_du_trajet,
    longueur_moyenne_de_rayonnement,
    opacite,
    transmittance,
)

MICRON = 1e-6


def test_les_gaz_qui_rayonnent_viennent_de_la_combustion() -> None:
    """4.1 : `CO2` et `H2O`, et la suie du cracking `CH4 -> C + 2 H2`."""
    assert est_equilibree({"CH4": 1, "O2": 2}, {"CO2": 1, "H2O": 2})
    assert est_equilibree({"CH4": 1}, {"C": 1, "H2": 2})


def test_la_loi_de_beer_et_l_opacite() -> None:
    """(4.1) a (4.4) : `tau = exp(-a l)`, `alpha = 1 - tau`, `a l = b p l`."""
    assert transmittance(0.5, 2.0) == pytest.approx(math.exp(-1.0))
    assert absorptivite_du_trajet(0.5, 2.0) == pytest.approx(1 - math.exp(-1.0))
    concentre = absorptivite_du_trajet(4.0 * 0.2, 0.5)
    dilue = absorptivite_du_trajet(4.0 * 0.05, 2.0)
    assert opacite(0.2, 0.5) == pytest.approx(opacite(0.05, 2.0))
    assert concentre == pytest.approx(dilue)


def _dioxyde(longueur: float) -> float:
    """Fig. 4.3, schematisee : trois bandes, vers 2,7, 4,3 et 15 micrometres."""
    bandes = ((2.6, 2.8), (4.1, 4.5), (13.0, 17.0))
    dans_une_bande = any(bas <= longueur / MICRON < haut for bas, haut in bandes)
    return 30.0 if dans_une_bande else 0.0


COUPURES = tuple(b * MICRON for b in (2.6, 2.8, 4.1, 4.5, 13.0, 17.0))


def test_le_facteur_d_emission_d_un_volume_hemispherique() -> None:
    """(4.11) : `1 - exp(-b p l)`, pondere par le rayonnement du corps noir."""
    mince = facteur_d_emission_hemispherique(_dioxyde, 1e-4, 1200.0, COUPURES)
    double = facteur_d_emission_hemispherique(_dioxyde, 2e-4, 1200.0, COUPURES)
    assert double == pytest.approx(2 * mince, rel=5e-3)
    epais = facteur_d_emission_hemispherique(_dioxyde, 10.0, 1200.0, COUPURES)
    bandes = sum(
        fraction_emise_sous(haut, 1200.0) - fraction_emise_sous(bas, 1200.0)
        for bas, haut in zip(COUPURES[::2], COUPURES[1::2], strict=True)
    )
    assert epais == pytest.approx(bandes, rel=1e-3)
    assert epais == pytest.approx(0.124, abs=0.001)


def test_un_gaz_epais_emet_moins_quand_il_chauffe() -> None:
    """Fig. 4.7a : a opacite donnee, `eps_CO2` decroit aux hautes temperatures.

    Le maximum de Planck quitte les bandes pour les courtes longueurs
    d'onde. De meme (4.10) : `alpha_g` se lit a la temperature de la
    surface qui emet, non a celle du gaz.
    """
    emissions = [
        facteur_d_emission_hemispherique(_dioxyde, 10.0, temperature, COUPURES)
        for temperature in (1200.0, 1600.0, 2000.0, 2300.0)
    ]
    assert emissions == sorted(emissions, reverse=True)
    assert emissions[-1] < emissions[0] / 2


def test_le_melange_des_deux_gaz() -> None:
    """(4.12) : `eps = eps_CO2 + eps_H2O - delta`, le recouvrement des bandes."""
    assert facteur_d_emission_du_melange(0.037, 0.0075) == pytest.approx(0.0445)
    assert facteur_d_emission_du_melange(0.1, 0.2, recouvrement=0.03) == pytest.approx(
        0.27
    )
    with pytest.raises(ValueError, match="recouvrement negatif"):
        facteur_d_emission_du_melange(0.1, 0.2, recouvrement=-0.01)
    with pytest.raises(ValueError, match="facteur d'emission hors de"):
        facteur_d_emission_du_melange(0.7, 0.6)


def test_la_longueur_moyenne_des_rayons_d_une_sphere() -> None:
    """(4.16) et (4.17) : `L_o = int l dF = 4 V / S`, soit `4 R / 3`."""
    rayon = 0.75

    def corde(angle: float) -> float:
        poids = 2 * math.cos(angle) * math.sin(angle)
        return 2 * rayon * math.cos(angle) * poids

    integree, _ = quad(corde, 0.0, math.pi / 2)
    volume, surface = 4 / 3 * math.pi * rayon**3, 4 * math.pi * rayon**2
    assert longueur_moyenne_de_rayonnement(volume, surface) == pytest.approx(integree)
    assert integree == pytest.approx(4 * rayon / 3)


def test_l_absorption_d_une_sphere_de_gaz() -> None:
    """(4.14) : la moyenne de `1 - exp(-a l)` sur les rayons, en forme close."""
    coefficient, rayon = 1.3, 0.75

    def absorbe(angle: float) -> float:
        poids = 2 * math.cos(angle) * math.sin(angle)
        return (1 - math.exp(-coefficient * 2 * rayon * math.cos(angle))) * poids

    integree, _ = quad(absorbe, 0.0, math.pi / 2)
    assert absorption_d_une_sphere(coefficient, rayon) == pytest.approx(integree)
    assert absorption_d_une_sphere(1e-6, rayon) == pytest.approx(1e-6, rel=1e-3)
    assert absorption_d_une_sphere(1e3, rayon) == pytest.approx(1.0, abs=1e-5)


def test_le_rayon_equivalent_decroit_avec_l_opacite() -> None:
    """Fig. 4.10a : `L_eq` part de `L_o` et decroit quand le gaz s'epaissit."""
    rayon = 0.75
    mince_limite = 4 * rayon / 3
    rapports = []
    for coefficient in (1e-4, 0.2, 1.4, 3.0):
        absorbe = absorption_d_une_sphere(coefficient, rayon)
        rapports.append(rayon_equivalent(absorbe, coefficient) / mince_limite)
    assert rapports[0] == pytest.approx(1.0, abs=1e-4)
    assert rapports == sorted(rapports, reverse=True)
    assert 0.85 < rapports[2] < 0.95
    with pytest.raises(ValueError, match="absorption hors de"):
        rayon_equivalent(1.0, 0.5)


def test_l_erreur_du_rayon_moyen_s_annule_et_s_efface() -> None:
    """Fig. 4.10b : `xi = 1` pour une opacite, et de nouveau pour un gaz opaque."""
    rayon = 0.75
    moyen = rayon_moyen(4 / 3 * math.pi * rayon**3, 4 * math.pi * rayon**2)
    assert moyen == pytest.approx(0.88 * 4 * rayon / 3)

    def erreur(coefficient: float) -> float:
        exacte = absorption_d_une_sphere(coefficient, rayon)
        return absorptivite_du_trajet(coefficient, moyen) / exacte

    assert erreur(1e-4) == pytest.approx(0.88, abs=1e-3)
    assert erreur(0.5) < 1.0 < erreur(3.0)
    assert erreur(50.0) == pytest.approx(1.0, abs=1e-3)


def test_le_rayon_moyen_des_deux_exemples() -> None:
    """(4.18) et page 167 : le tube, `0,88 d` ; le faisceau en lignes."""
    diametre = 0.05
    tube = rayon_moyen(math.pi * diametre**2 / 4, math.pi * diametre)
    assert tube == pytest.approx(0.88 * diametre)
    assert tube == pytest.approx(
        3.52 * (math.pi * diametre**2 / 4) / (math.pi * diametre)
    )
    pas = 0.08
    maille = rayon_moyen(pas**2 - math.pi * diametre**2 / 4, math.pi * diametre)
    assert maille == pytest.approx(
        0.88 * (4 * pas**2 / (math.pi * diametre) - diametre)
    )
    with pytest.raises(ValueError, match="facteur de forme hors de"):
        rayon_moyen(1.0, 1.0, facteur=0.7)


def test_l_echange_entre_un_gaz_et_son_enveloppe_noire() -> None:
    """(4.19) : `sigma S (eps_g T_g^4 - alpha_g T_S^4)` ; le gaz n'est pas gris."""
    flux = flux_entre_le_gaz_et_son_enveloppe(
        2.0, gaz=(0.2, 1500.0), paroi=(0.3, 600.0)
    )
    attendu = 2.0 * (0.2 * emittance(1500.0) - 0.3 * emittance(600.0))
    assert flux == pytest.approx(attendu)
    a_l_equilibre = flux_entre_le_gaz_et_son_enveloppe(
        2.0, gaz=(0.2, 900.0), paroi=(0.2, 900.0)
    )
    assert a_l_equilibre == 0.0


def test_l_echange_entre_un_gaz_et_son_enveloppe_grise() -> None:
    """(4.20) : le facteur `(eps_S + 1) / 2`, entre les deux bornes de la page 168."""
    noire = flux_entre_le_gaz_et_son_enveloppe(2.0, gaz=(0.2, 1500.0), paroi=(0.3, 0.0))
    grise = flux_entre_le_gaz_et_son_enveloppe(
        2.0, gaz=(0.2, 1500.0), paroi=(0.3, 0.0), emissivite=0.9
    )
    assert grise == pytest.approx(0.95 * noire)
    assert 0.9 * noire < grise < noire
    with pytest.raises(ValueError, match="emissivite hors de"):
        flux_entre_le_gaz_et_son_enveloppe(
            2.0, gaz=(0.2, 1500.0), paroi=(0.3, 0.0), emissivite=1.5
        )


def test_le_gaz_le_recepteur_et_la_paroi_adiabatique() -> None:
    """(4.21) et (4.22), contre la methode des zones ou le gaz est une zone noire.

    Le gaz echange `S eps_g` avec chaque surface, et les surfaces ne se
    voient plus qu'a travers lui : leurs aires directes portent `tau`.
    """
    recepteur, paroi, gaz, vers_le_recepteur = 4.0, 6.0, 0.3, 0.5
    transmis = 1 - gaz
    vers_la_paroi = vers_le_recepteur * paroi / recepteur
    directes = (
        (0.0, recepteur * gaz, paroi * gaz),
        (recepteur * gaz, recepteur * (1 - vers_la_paroi) * transmis, 3.0 * transmis),
        (paroi * gaz, 3.0 * transmis, paroi * (1 - vers_le_recepteur) * transmis),
    )
    zones = (
        Zone((recepteur + paroi) * gaz, 1.0),
        Zone(recepteur, 0.8),
        Zone.adiabatique(paroi),
    )
    par_les_zones = EnceinteGrise(zones, directes).aires_d_echange_total()[0][1]
    aire = aire_d_echange_avec_le_gaz(
        recepteur, paroi, gaz=gaz, vers_le_recepteur=vers_le_recepteur, emissivite=0.8
    )
    assert aire == pytest.approx(par_les_zones)
    relais = (paroi / recepteur) / (1 / gaz + 1 / (vers_le_recepteur * transmis))
    noires = recepteur * (gaz + relais)
    assert aire == pytest.approx(1 / (1 / noires + 0.2 / (0.8 * recepteur)))


def test_sans_paroi_adiabatique_le_recepteur_ne_voit_que_le_gaz() -> None:
    """`F_R1 = 0` : il reste `S_1 eps_g`, en serie avec la resistance de surface."""
    aire = aire_d_echange_avec_le_gaz(
        4.0, 6.0, gaz=0.3, vers_le_recepteur=0.0, emissivite=1.0
    )
    assert aire == pytest.approx(4.0 * 0.3)


def test_l_exercice_du_faisceau_d_evaporateur() -> None:
    """Page 175 : tubes de 40 mm au pas de 60 mm, fumees a 950 degres, 664 W/m."""
    diametre, pas = 0.04, 0.06
    volume = math.sqrt(3) / 4 * pas**2 - math.pi / 8 * diametre**2
    surface = math.pi / 2 * diametre
    assert volume == pytest.approx(0.93e-3, abs=5e-6)
    assert surface == pytest.approx(0.0628, abs=5e-5)
    assert longueur_moyenne_de_rayonnement(volume, surface) == pytest.approx(
        0.0592, abs=5e-5
    )
    moyen = rayon_moyen(volume, surface)
    assert moyen == pytest.approx(0.0521, abs=5e-5)
    totale = 1.013
    dioxyde = pression_partielle(0.125, totale)
    vapeur = pression_partielle(0.066, totale)
    assert (dioxyde, vapeur) == pytest.approx((0.127, 0.0669), abs=5e-4)
    assert opacite(dioxyde, moyen) == pytest.approx(0.00662, abs=3e-5)
    assert opacite(vapeur, moyen) == pytest.approx(0.00348, abs=1e-5)
    lus = (0.037, 0.0075)
    emission = facteur_d_emission_du_melange(*lus)
    assert emission == pytest.approx(0.0445)
    flux = flux_entre_le_gaz_et_son_enveloppe(
        math.pi * diametre,
        gaz=(emission, 1223.15),
        paroi=(emission, 423.15),
        emissivite=0.9,
    )
    assert flux == pytest.approx(664.0, abs=1.5)
    convecte = math.pi * diametre * 60.0 * (950.0 - 150.0)
    assert convecte == pytest.approx(6030.0, abs=3.0)
    assert convecte > 9 * flux


def test_l_exercice_ii_4_2_l_evaporateur_ecran() -> None:
    """Page 177 : tubes de 57 mm au pas de 70 mm, flamme noire a 1400 degres.

    Le syllabus lit `F_PT,R = 0,89` « sur la courbe 3 » : c'est la
    courbe 1. La courbe 3 passe a 0,99 en `l / d = 1,23`.
    """
    diametre, pas = 0.057, 0.070
    surface = 0.2 / 0.8 * 70 / (math.pi * 57)
    imprime = 1 / (surface + 1 / 0.89)
    assert imprime == pytest.approx(0.82, abs=0.005)
    flux_imprime = pas * imprime * (emittance(1673.1) - emittance(588.1))
    assert flux_imprime == pytest.approx(25113.0, rel=2e-3)
    assert flux_imprime / 1.29e6 == pytest.approx(0.0194, abs=1e-4)
    assert pytest.approx(4300.0, rel=2e-3) == 83.3 / 0.0194
    assert facteur_vers_l_ecran_adosse(diametre, pas) == pytest.approx(0.994, abs=5e-4)
    ecran = facteur_corrige_de_l_ecran(diametre, pas, emissivite=0.8)
    assert ecran == pytest.approx(0.906, abs=5e-4)
    flux = flux_entre_surfaces_noires(pas * ecran, 1673.15, 588.15)
    assert flux / flux_imprime == pytest.approx(1.105, abs=0.005)
