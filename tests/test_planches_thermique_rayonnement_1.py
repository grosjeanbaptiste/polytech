"""Les planches de thermique : la nature du transfert radiatif.

Source : `ThermiqueFondements/SLIDES/Slides_05_Rayonnement_1_Nature_du_transfert.pdf`,
les 21 planches, lues une a une.

Les oracles : l'angle solide de la planche 5, la loi de Lambert de la
planche 8, les nombres des planches 15 et 16 -- 460 W/m^2, 4,095e-12 T^5,
98 % --, le spectre solaire de la planche 17 et les trois cas de la
planche 21.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.rayonnement.corps_noir import (
    emittance,
    fraction_emise_sous,
    longueur_d_onde_du_maximum,
    proprietes_d_un_solide_reel,
)
from polytech.thermique.domain.rayonnement.facteurs_globaux import (
    Bande,
    facteur_global_par_bandes,
)
from polytech.thermique.domain.rayonnement.lambert import (
    angle_solide,
    exitance_grise,
    exitance_par_integration,
    luminance_de_planck,
    luminance_emise,
    luminance_maximale,
)
from polytech.thermique.domain.rayonnement_des_gaz import transmittance

MICRON = 1e-6


def test_l_angle_solide_se_mesure_sur_la_sphere_unite() -> None:
    """Planche 5 : `omega = dA / r^2` ; 4 pi pour l'espace, 2 pi d'un cote du plan."""
    rayon = 2.5
    assert angle_solide(4 * math.pi * rayon**2, rayon) == pytest.approx(4 * math.pi)
    assert angle_solide(2 * math.pi * rayon**2, rayon) == pytest.approx(2 * math.pi)
    assert angle_solide(1.0, 1.0) == 1.0
    with pytest.raises(ValueError, match="rayon nul ou negatif"):
        angle_solide(1.0, 0.0)


def test_la_loi_de_lambert_des_planches() -> None:
    """Planche 8 : `M = int L cos(alpha) d(omega)`, soit `pi L` si `L` est uniforme."""
    luminance = sp.Symbol("L", positive=True)
    assert exitance_par_integration(luminance) == sp.pi * luminance


def test_la_loi_de_beer_pour_une_plaque() -> None:
    """Planche 10 : `tau = f(e) / f(0) = exp(-a e)`."""
    assert transmittance(120.0, 0.004) == pytest.approx(math.exp(-0.48))
    double = transmittance(120.0, 0.008)
    assert double == pytest.approx(transmittance(120.0, 0.004) ** 2)


def test_les_trois_facteurs_d_une_surface() -> None:
    """Planche 11 : `rho + tau + alpha = 1` ; opaque, `rho = 1 - alpha`."""
    assert sum(proprietes_d_un_solide_reel(0.3, 0.5)) == pytest.approx(1.0)
    assert proprietes_d_un_solide_reel(0.3, 0.7)[2] == pytest.approx(0.0)


def test_la_loi_de_kirchhoff_des_planches() -> None:
    """Planche 14 : `l = alpha l_o`, et pour un corps gris `L = eps L_o`."""
    noire = luminance_de_planck(4 * MICRON, 1000.0)
    assert luminance_emise(0.3, 4 * MICRON, 1000.0) == pytest.approx(0.3 * noire)
    assert exitance_grise(0.3, 1000.0) == pytest.approx(0.3 * emittance(1000.0))


def test_a_300_kelvins_un_corps_noir_emet_460_watts_par_metre_carre() -> None:
    """Planche 15 : « A T = 300 K, M_0 = 460 W/m^2 ! »."""
    assert emittance(300.0) == pytest.approx(460.0, abs=1.0)


@pytest.mark.parametrize(
    ("temperature", "sommet"), [(800.0, 1340.0), (1000.0, 4100.0), (1200.0, 10200.0)]
)
def test_le_maximum_de_la_loi_de_planck(temperature: float, sommet: float) -> None:
    """Planches 15 et 16 : `l_max = 4,095e-12 T^5`, en W/(m^2 um sr), et le graphe."""
    maximale = luminance_maximale(temperature) * MICRON
    assert maximale == pytest.approx(4.095e-12 * temperature**5, rel=1e-3)
    assert maximale == pytest.approx(sommet, rel=0.01)
    au_sommet = longueur_d_onde_du_maximum(temperature)
    assert au_sommet * temperature / MICRON == pytest.approx(2898.0, abs=0.5)
    for voisine in (0.97 * au_sommet, 1.03 * au_sommet):
        assert luminance_de_planck(voisine, temperature) < maximale / MICRON


def test_la_loi_de_planck_reduite_ne_depend_que_de_lambda_t() -> None:
    """Planche 16 : `l / l_max = f(lambda T)` ; 0,55 en 5000 um K, 0,08 en 10 000."""
    produit = 5000.0 * MICRON
    reduites = [
        luminance_de_planck(produit / t, t) / luminance_maximale(t)
        for t in (500.0, 1500.0, 5780.0)
    ]
    assert reduites[0] == pytest.approx(reduites[1])
    assert reduites[1] == pytest.approx(reduites[2])
    assert reduites[0] == pytest.approx(0.55, abs=0.03)
    lointaine = luminance_de_planck(10 * MICRON, 1000.0) / luminance_maximale(1000.0)
    assert lointaine == pytest.approx(0.08, abs=0.02)


def test_les_reperes_de_la_fraction_emise() -> None:
    """Planche 16 : un quart sous `lambda_max` ; 98 % de la moitie a huit fois."""
    temperature = 1000.0
    sommet = longueur_d_onde_du_maximum(temperature)
    assert fraction_emise_sous(sommet, temperature) == pytest.approx(0.25, abs=0.005)
    basse = fraction_emise_sous(sommet / 2, temperature)
    haute = fraction_emise_sous(8 * sommet, temperature)
    assert haute - basse == pytest.approx(0.98, abs=0.005)
    assert haute == pytest.approx(0.99, abs=0.003)


def test_le_spectre_solaire() -> None:
    """Planche 17 : 5780 K, maximum a 0,5 um, tout entre 0,25 et 4 um.

    « ~ 50 % UV et visible (< 0,8 um) » : le corps noir a 5780 K en donne 58 %.
    """
    sommet = longueur_d_onde_du_maximum(5780.0) / MICRON
    assert sommet == pytest.approx(0.5, abs=0.005)
    visible = fraction_emise_sous(0.8 * MICRON, 5780.0)
    assert visible == pytest.approx(0.58, abs=0.005)
    assert visible == pytest.approx(0.5, abs=0.1)
    entre = fraction_emise_sous(4 * MICRON, 5780.0) - fraction_emise_sous(
        0.25 * MICRON, 5780.0
    )
    assert entre > 0.97


CLAIR = (Bande(1.5 * MICRON, 0.1), Bande(math.inf, 0.95))
"""Planche 18 : un non-conducteur clair, faible aux courtes longueurs d'onde."""
SOMBRE = (Bande(1.5 * MICRON, 0.9), Bande(math.inf, 0.95))


def test_la_neige_fond_plus_vite_sous_un_radiant_que_sous_le_soleil() -> None:
    """Planche 21, cas 3 : a flux egal, 800 K contre 5800 K."""
    sous_le_radiant = facteur_global_par_bandes(CLAIR, 800.0)
    sous_le_soleil = facteur_global_par_bandes(CLAIR, 5800.0)
    assert sous_le_radiant > 0.9
    assert sous_le_soleil < 0.25
    assert sous_le_radiant > 4 * sous_le_soleil


def test_la_couleur_d_un_radiateur_ne_change_rien() -> None:
    """Planche 21, cas 1 : vers 350 K tout se joue dans l'infrarouge lointain."""
    clair = facteur_global_par_bandes(CLAIR, 350.0)
    sombre = facteur_global_par_bandes(SOMBRE, 350.0)
    assert clair == pytest.approx(sombre, abs=1e-6)
    assert clair == pytest.approx(0.95, abs=1e-6)


def test_un_panneau_solaire_veut_absorber_sans_emettre() -> None:
    """Planche 21, cas 2 : `alpha` eleve a 5800 K, `eps` faible a sa temperature."""
    selectif = (Bande(2.5 * MICRON, 0.95), Bande(math.inf, 0.1))
    assert facteur_global_par_bandes(selectif, 5800.0) > 0.9
    assert facteur_global_par_bandes(selectif, 350.0) < 0.11
