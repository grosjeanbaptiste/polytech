"""Le syllabus de thermique, tome 1 : les lois du rayonnement thermique.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 105 a
118 (deuxieme partie, chapitre 1), lues sur les pages rendues.

Les oracles : les equations (1.4) a (1.14), les sommets de la figure 1.8,
les trois exemples de la loi du deplacement, la figure 1.11 et
l'exercice II.1.1 de la vitre.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import sympy as sp
from scipy.integrate import quad

from polytech.thermique.domain.rayonnement.corps_noir import (
    STEFAN_BOLTZMANN,
    WIEN,
    emittance,
    exitance_spectrale,
    fraction_emise_sous,
    longueur_d_onde_du_maximum,
    luminance_spectrale,
    proprietes_d_un_solide_reel,
)
from polytech.thermique.domain.rayonnement.facteurs_globaux import (
    Bande,
    facteur_global,
    facteur_global_par_bandes,
)
from polytech.thermique.domain.rayonnement.lambert import (
    ALPHA,
    PREMIERE_CONSTANTE,
    SECONDE_CONSTANTE,
    exitance_grise,
    exitance_par_integration,
    luminance_de_planck,
    luminance_du_corps_noir,
    luminance_emise,
)
from polytech.thermique.domain.rayonnement.vitre_selective import VitreSelective
from polytech.thermique.domain.rayonnement_des_gaz import transmittance

MICRON = 1e-6


def test_la_loi_de_lambert() -> None:
    """(1.4) : `M = pi L`, l'integrale de `cos(alpha)` sur l'hemisphere."""
    luminance = sp.Symbol("L", positive=True)
    assert exitance_par_integration(luminance) == sp.pi * luminance
    rasante = exitance_par_integration(luminance * sp.cos(ALPHA))
    assert rasante == 2 * sp.pi * luminance / 3


def test_les_constantes_de_la_loi_de_planck() -> None:
    """(1.11) : `c_1 = 1,191 10^8 W um^4/(m^2 sr)` et `c_2 = 14 388 um K`."""
    assert pytest.approx(1.191e8, rel=5e-4) == PREMIERE_CONSTANTE / MICRON**4
    assert pytest.approx(14388.0, rel=5e-5) == SECONDE_CONSTANTE / MICRON


def test_la_luminance_de_planck_est_l_exitance_spectrale_sur_pi() -> None:
    """Le schema de calcul de la page 109 : `m_lambda = pi l_lambda`."""
    luminance = luminance_de_planck(2 * MICRON, 1500.0)
    attendue = 1.191e8 * 2.0**-5 / math.expm1(14388.0 / 3000.0)
    assert luminance * MICRON == pytest.approx(attendue, rel=1e-3)
    assert luminance_de_planck(1e-8, 300.0) == 0.0
    exitance = exitance_spectrale(2 * MICRON, 1500.0)
    assert math.pi * luminance == pytest.approx(exitance)
    assert luminance_spectrale(2 * MICRON, 1500.0) == pytest.approx(luminance)
    with pytest.raises(ValueError, match="longueur d'onde nulle ou negative"):
        luminance_de_planck(0.0, 1500.0)
    with pytest.raises(ValueError, match="temperature nulle ou negative"):
        luminance_de_planck(2 * MICRON, 0.0)


@pytest.mark.parametrize(
    ("temperature", "sommet"), [(2000.0, 410.0), (1500.0, 98.0), (1000.0, 13.0)]
)
def test_les_sommets_de_la_figure_1_8(temperature: float, sommet: float) -> None:
    """Fig. 1.8, en kW/(m^2 um) : vers 410 a 2000 K, 100 a 1500 K, 13 a 1000 K."""
    au_maximum = longueur_d_onde_du_maximum(temperature)
    exitance = math.pi * luminance_de_planck(au_maximum, temperature)
    assert exitance * MICRON / 1e3 == pytest.approx(sommet, rel=0.02)
    for voisine in (0.95 * au_maximum, 1.05 * au_maximum):
        assert luminance_de_planck(voisine, temperature) < exitance / math.pi


@pytest.mark.parametrize(
    ("temperature", "imprimee"), [(300.0, 10.0), (1500.0, 2.0), (6000.0, 0.5)]
)
def test_la_loi_du_deplacement(temperature: float, imprimee: float) -> None:
    """(1.12) : « lambda_max T ~ 3000 um K » ; la figure 1.8 ecrit 2898."""
    assert pytest.approx(2898.0, abs=0.5) == WIEN / MICRON
    maximum = longueur_d_onde_du_maximum(temperature) / MICRON
    assert maximum == pytest.approx(imprimee, rel=0.04)
    assert maximum < imprimee


def test_la_loi_de_stefan_boltzmann_integre_celle_de_planck() -> None:
    """(1.10) : `M_o = sigma T^4`, `L_o = sigma T^4 / pi`."""
    assert pytest.approx(5.67e-8, rel=1e-3) == STEFAN_BOLTZMANN
    temperature = 1200.0
    integrale, _ = quad(luminance_de_planck, 1e-8, 1e-3, args=(temperature,))
    assert integrale == pytest.approx(luminance_du_corps_noir(temperature), rel=1e-6)
    lambertienne = math.pi * luminance_du_corps_noir(temperature)
    assert lambertienne == pytest.approx(emittance(temperature))


def test_l_emission_d_un_corps_quelconque() -> None:
    """(1.13) et (1.14) : `l = alpha l_o`, `M = epsilon sigma T^4`."""
    noire = luminance_de_planck(3 * MICRON, 800.0)
    assert luminance_emise(0.4, 3 * MICRON, 800.0) == pytest.approx(0.4 * noire)
    assert exitance_grise(0.8, 500.0) == pytest.approx(0.8 * 5.67e-8 * 500**4, rel=1e-3)
    with pytest.raises(ValueError, match="facteur hors de"):
        exitance_grise(1.2, 500.0)
    with pytest.raises(ValueError, match="facteur hors de"):
        luminance_emise(-0.1, 3 * MICRON, 800.0)


def test_un_diathermane_affaiblit_le_rayon() -> None:
    """(1.6) : `df / f = -a dx`, donc `f = f_0 exp(-a x)` ; rien si `a = 0`."""
    assert transmittance(0.0, 5.0) == 1.0
    assert transmittance(0.2, 5.0) == pytest.approx(math.exp(-1.0))
    pas = 1e-6
    pente = (transmittance(0.2, 5.0 + pas) - transmittance(0.2, 5.0 - pas)) / (2 * pas)
    assert pente / transmittance(0.2, 5.0) == pytest.approx(-0.2)


def test_absorbe_reflechi_et_transmis_font_un() -> None:
    """Page 110 : `alpha + rho + tau = 1` ; opaque, `rho = 1 - alpha`."""
    assert proprietes_d_un_solide_reel(0.06, 0.06) == pytest.approx((0.06, 0.06, 0.88))
    assert proprietes_d_un_solide_reel(0.7, 0.3)[2] == pytest.approx(0.0)


PLATRE = (Bande(4 * MICRON, 0.1), Bande(math.inf, 0.9))
"""Fig. 1.10, courbe 2 : faible sous 4 um, eleve au-dela."""
ALUMINIUM = (Bande(1 * MICRON, 0.3), Bande(4 * MICRON, 0.1), Bande(math.inf, 0.04))
"""Fig. 1.10, courbe 1 : un metal poli reflechit mieux l'infrarouge."""


def test_le_facteur_global_depend_du_spectre_incident() -> None:
    """(1.7) : `alpha = int(alpha_l f_l) / int(f_l)`, non intrinseque."""

    def platre(longueur: float) -> float:
        return 0.1 if longueur < 4 * MICRON else 0.9

    def solaire(longueur: float) -> float:
        return luminance_de_planck(longueur, 6000.0)

    def ambiant(longueur: float) -> float:
        return luminance_de_planck(longueur, 300.0)

    coupures = (4 * MICRON,)
    au_soleil = facteur_global(platre, solaire, coupures)
    a_l_ambiance = facteur_global(platre, ambiant, coupures)
    assert au_soleil == pytest.approx(0.1, abs=0.01)
    assert a_l_ambiance == pytest.approx(0.9, abs=0.01)
    assert facteur_global(lambda _: 0.6, solaire, coupures) == pytest.approx(0.6)


def test_le_facteur_par_bandes_est_l_integrale() -> None:
    """(1.9) : pondere par `k(lambda T)`, c'est la meme integrale."""

    def platre(longueur: float) -> float:
        return 0.1 if longueur < 4 * MICRON else 0.9

    for temperature in (300.0, 1000.0, 6000.0):
        par_bandes = facteur_global_par_bandes(PLATRE, temperature)

        def planck(longueur: float, t: float = temperature) -> float:
            return luminance_de_planck(longueur, t)

        integre = facteur_global(platre, planck, (4 * MICRON,))
        assert par_bandes == pytest.approx(integre, abs=2e-4)
    k = fraction_emise_sous(4 * MICRON, 1000.0)
    attendu = 0.1 * k + 0.9 * (1 - k)
    assert facteur_global_par_bandes(PLATRE, 1000.0) == pytest.approx(attendu)


def test_la_figure_1_11() -> None:
    """Le platre s'eclaircit quand la source chauffe, l'aluminium poli s'assombrit."""
    temperatures = (300.0, 1000.0, 3000.0, 6000.0)
    platre = [facteur_global_par_bandes(PLATRE, t) for t in temperatures]
    aluminium = [facteur_global_par_bandes(ALUMINIUM, t) for t in temperatures]
    assert platre == sorted(platre, reverse=True)
    assert aluminium == sorted(aluminium)
    assert platre[0] > 0.85
    assert platre[-1] < 0.15
    gris = (Bande(math.inf, 0.5),)
    assert {facteur_global_par_bandes(gris, t) for t in temperatures} == {0.5}


DESORDRE = (Bande(4 * MICRON, 0.1), Bande(MICRON, 0.9), Bande(math.inf, 0.9))


def test_des_bandes_mal_ordonnees_sont_refusees() -> None:
    with pytest.raises(ValueError, match="bandes non croissantes"):
        facteur_global_par_bandes(DESORDRE, 300.0)
    with pytest.raises(ValueError, match="derniere bande bornee"):
        facteur_global_par_bandes((Bande(4 * MICRON, 0.1),), 300.0)


VITRE = VitreSelective(transmission_courte=0.88, reflexion=0.06, coupure=3.5 * MICRON)


def test_l_exercice_ii_1_1_la_vitre_au_soleil() -> None:
    """`lambda T = 21 000 um K`, `k ~ 1`, donc `tau = 0,88` et `alpha = 0,06`."""
    fraction = fraction_emise_sous(3.5 * MICRON, 6000.0)
    assert fraction == pytest.approx(1.0, abs=0.015)
    assert VITRE.transmission(6000.0) == pytest.approx(0.88, abs=0.015)
    assert VITRE.absorption(6000.0) == pytest.approx(0.06, abs=0.015)
    transmis = (Bande(3.5 * MICRON, 0.88), Bande(math.inf, 0.0))
    par_bandes = facteur_global_par_bandes(transmis, 6000.0)
    assert par_bandes == pytest.approx(VITRE.transmission(6000.0))


def test_l_exercice_ii_1_1_la_vitre_a_l_ambiance() -> None:
    """`lambda T = 1050 um K` : opaque. Le syllabus lit `k ~ 0,1 %` ; c'est 0,056 %."""
    fraction = fraction_emise_sous(3.5 * MICRON, 300.0)
    assert fraction == pytest.approx(0.00056, abs=1e-5)
    assert VITRE.transmission(300.0) == pytest.approx(0.00049, abs=1e-5)
    assert VITRE.transmission(300.0) < 0.0009
    assert VITRE.absorption(300.0) == pytest.approx(0.94, abs=1e-3)


def test_une_raie_etroite_n_echappe_pas_a_l_integrale() -> None:
    """Les coupures decoupent l'integrale : sans elles, la raie serait manquee."""
    bas, haut = 0.50 * MICRON, 0.501 * MICRON

    def raie(longueur: float) -> float:
        return 1.0 if bas <= longueur < haut else 0.0

    def solaire(longueur: float) -> float:
        return luminance_de_planck(longueur, 6000.0)

    attendu = fraction_emise_sous(haut, 6000.0) - fraction_emise_sous(bas, 6000.0)
    assert facteur_global(raie, solaire, (bas, haut)) == pytest.approx(
        attendu, rel=1e-4
    )


def test_l_exitance_spectrale_s_integre_en_stefan_boltzmann() -> None:
    """1.4.1 et 1.4.2 : `M_o = integrale(m_lambda) = sigma T^4`, et `M_o = pi L_o`.

    `corps_noir.luminance_spectrale` rendait l'exitance sous le nom de la
    luminance : son integrale donnait `sigma T^4` au lieu de `sigma T^4 / pi`.
    """
    temperature = 1200.0
    bornes = (1e-8, WIEN / temperature, 1e-3)
    exitance = sum(
        quad(exitance_spectrale, bas, haut, args=(temperature,), limit=200)[0]
        for bas, haut in pairwise(bornes)
    )
    assert exitance == pytest.approx(emittance(temperature), rel=1e-6)
    luminance = sum(
        quad(luminance_spectrale, bas, haut, args=(temperature,), limit=200)[0]
        for bas, haut in pairwise(bornes)
    )
    assert luminance == pytest.approx(emittance(temperature) / math.pi, rel=1e-6)
    with pytest.raises(ValueError, match="longueur d'onde nulle ou negative"):
        exitance_spectrale(0.0, temperature)


def test_l_exitance_spectrale_s_eteint_aux_courtes_longueurs_d_onde() -> None:
    """La ou `exp(c_2 / lambda T)` deborde le flottant, la limite exacte est zero.

    La fonction levait `OverflowError` : a 300 K, des 0,06 micrometre.
    """
    assert exitance_spectrale(1e-8, 300.0) == 0.0
    assert exitance_spectrale(0.07 * MICRON, 300.0) > 0.0
    with pytest.raises(ValueError, match="temperature nulle ou negative"):
        exitance_spectrale(2 * MICRON, 0.0)
