"""Le syllabus de thermique, tome 1 : l'introduction.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 5 a 25
(sections 0.1 a 0.8), lues sur les pages rendues.

Les oracles : les relations (1) a (5), les conversions de la section 0.7
(`1 kcal = 4185,5 J`), la caracteristique du capteur solaire plan de la
section 0.8.1 et ses figures 12 a 15, et le module de refroidissement
TCM de la section 0.8.2 -- 5 W sur 4,5 mm x 4,5 mm, et les resistances
de la figure 19.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.thermique.domain.capteur_solaire import (
    CapteurPlan,
    resistance_de_pertes,
)
from polytech.thermique.domain.modes import (
    coefficient_superpose,
    flux_de_newton,
    resistance_depuis_l_echange,
)
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.reseaux import echauffement, en_parallele, en_serie
from polytech.thermique.domain.unites import (
    KILOCALORIE,
    joules_depuis_kilocalories,
    watts_depuis_kilocalories_par_heure,
)


def test_la_loi_de_newton_et_la_resistance_qu_elle_definit() -> None:
    """(1) : `phi = K_conv (t_p - t_fluide)` ; `K S` est une conductance."""
    assert flux_de_newton(12.0, 1.0, 30.0) == pytest.approx(360.0)
    assert resistance_depuis_l_echange(12.0, 2.0) == pytest.approx(1 / 24)


def test_convection_et_rayonnement_agissent_en_parallele() -> None:
    """(2) a (5) et fig. 8 : `K = K_conv + K_rayt`."""
    convection, rayonnement = 8.0, 5.0
    assert coefficient_superpose((convection, rayonnement)) == 13.0
    ensemble = en_parallele(1 / convection, 1 / rayonnement)
    assert ensemble == pytest.approx(1 / 13.0)
    with pytest.raises(ValueError, match="aucune resistance"):
        en_parallele()


def test_le_coefficient_global_traverse_trois_resistances_en_serie() -> None:
    """Fig. 11 : `1 / (K_eau S_i) + R_cond + 1 / (K_air S_e)`."""
    interieure = resistance_depuis_l_echange(2000.0, 0.05)
    exterieure = resistance_depuis_l_echange(40.0, 0.06)
    totale = en_serie(interieure, 0.002, exterieure)
    assert totale == pytest.approx(0.01 + 0.002 + 1 / 2.4)
    assert exterieure / totale > 0.97
    with pytest.raises(ValueError, match="resistance nulle ou negative"):
        en_serie(interieure, -0.002)


def test_les_conversions_de_la_section_0_7() -> None:
    """`1 kcal = 4185,5 J`, donc `1 kcal/h = 1,1626 W`.

    Le syllabus imprime 1,1627 : le quotient vaut 1,16264.
    """
    assert KILOCALORIE == 4185.5
    assert joules_depuis_kilocalories(2.0) == 8371.0
    par_heure = watts_depuis_kilocalories_par_heure(1.0)
    assert par_heure == pytest.approx(4185.5 / 3600)
    assert round(par_heure, 4) == 1.1626
    assert abs(par_heure - 1.1627) > 5e-5


T_EAU, T_AMB, FLUX_A = sp.symbols("t_eau t_amb Phi_A")
R_C, R_P = sp.symbols("R_c R_p", positive=True)


def test_le_flux_utile_du_capteur_par_superposition() -> None:
    """Fig. 12 et 13 : `Phi_ut = (t_amb - t_eau) / (R_c + R_p) + Phi_A R_p / (...)`."""
    absorbeur = sp.Symbol("t_A")
    bilan = sp.Eq((absorbeur - T_EAU) / R_C + (absorbeur - T_AMB) / R_P, FLUX_A)
    (temperature,) = sp.solve(bilan, absorbeur)
    par_le_noeud = (temperature - T_EAU) / R_C
    capteur = CapteurPlan(vers_l_eau=R_C, de_pertes=R_P, transmission_absorption=1)
    assert sp.simplify(par_le_noeud - capteur.flux_utile(FLUX_A, T_EAU - T_AMB)) == 0
    assert capteur.facteur() == R_P / (R_C + R_P)


def test_la_caracteristique_du_capteur_est_une_droite() -> None:
    """Fig. 14 : ordonnee `F_c tau_V alpha_A`, pente `-F_c K_p`."""
    capteur = CapteurPlan(vers_l_eau=0.01, de_pertes=0.1, transmission_absorption=0.8)
    surface, pertes = 2.0, 1 / (0.1 * 2.0)
    facteur = capteur.facteur()
    assert facteur == pytest.approx(0.1 / 0.11)
    assert capteur.rendement(0.0, surface) == pytest.approx(facteur * 0.8)
    pente = capteur.rendement(0.01, surface) - capteur.rendement(0.0, surface)
    assert pente / 0.01 == pytest.approx(-facteur * pertes)
    flux_solaire, ecart = 1600.0, 20.0
    utile = capteur.flux_utile(flux_solaire * 0.8, ecart)
    reduit = ecart / (flux_solaire / surface)
    assert utile / flux_solaire == pytest.approx(capteur.rendement(reduit, surface))


def test_le_double_vitrage_l_emporte_en_hiver_seulement() -> None:
    """Fig. 15 : il transmet moins et perd moins ; les deux droites se croisent."""
    simple = CapteurPlan(0.002, 1 / 14, 0.85)
    double = CapteurPlan(0.002, 1 / 8, 0.75)
    surface, ete, hiver = 2.0, 0.01, 0.08
    assert simple.rendement(ete, surface) > double.rendement(ete, surface)
    assert double.rendement(hiver, surface) > simple.rendement(hiver, surface)


def test_la_resistance_de_pertes_du_capteur() -> None:
    """Page 20 : l'arriere et l'avant en parallele, chacun fait de series."""
    pertes = resistance_de_pertes(
        isolant=2.0,
        arriere=(0.5, 0.5),
        lame=(0.4, 0.4),
        vitre=0.01,
        exterieur=(0.2, 0.2),
    )
    par_l_arriere = 2.0 + 0.25
    par_l_avant = 0.2 + 0.01 + 0.1
    assert pertes == pytest.approx(1 / (1 / par_l_arriere + 1 / par_l_avant))


def test_la_densite_de_flux_d_un_composant_du_tcm() -> None:
    """0.8.2 : 5 W sur 4,5 mm x 4,5 mm, soit 250 000 W/m^2.

    Le syllabus y voit « approximativement 1/10 du flux quittant la
    surface du soleil » a 6 000 degres ; la loi de Stefan-Boltzmann de sa
    deuxieme partie donne trois cents fois plus, non dix.
    """
    densite = 5 / 4.5e-3**2
    assert densite == pytest.approx(250_000, rel=0.02)
    rapport = emittance(6000.0 + 273.15) / densite
    assert rapport == pytest.approx(356, abs=1)
    assert emittance(6000.0) / densite == pytest.approx(298, abs=1)


RESISTANCES_DU_TCM = {
    "boitier": 1.6,
    "helium": 2.2,
    "piston": 1.0,
    "contact": 3.0,
    "silicium": 0.4,
}
"""Fig. 19, en degres par watt, sans la plaque froide (0,03)."""


def test_l_echauffement_d_un_composant_du_tcm() -> None:
    """`Delta t = Phi R`. La figure annonce `R_int = 8,0` ; la somme fait 8,2."""
    interne = en_serie(*RESISTANCES_DU_TCM.values())
    assert interne == pytest.approx(8.2)
    assert interne != pytest.approx(8.0, abs=0.05)
    assert echauffement(5.0, (interne, 0.03)) == pytest.approx(41.15)
    assert max(RESISTANCES_DU_TCM, key=RESISTANCES_DU_TCM.__getitem__) == "contact"
