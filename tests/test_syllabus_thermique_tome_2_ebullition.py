"""Le syllabus de thermique, tome 2 : la transmission de chaleur par ebullition.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 79
a 94 (premiere partie, chapitre 6), lues sur les pages rendues.

Les oracles : l'equilibre d'une bulle et la surchauffe (6.1), les
relations de Rosenhow (6.2), de Zuber (6.3) et de Jens et Lottes (6.4),
et les exercices III.6.1 a III.6.3 -- 4,54 degres et 44 829 W/m^2 K,
205 040 W/m^2, 9,5 degres.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.ebullition import (
    FLUX_DES_CIRCUITS_DENSES,
    Pilotage,
    risque_de_burn_out,
)
from polytech.thermique.domain.ebullition_nucleee import (
    FluideBouillant,
    flux_de_zuber,
    surchauffe_de_jens_et_lottes,
    surchauffe_de_rosenhow,
    surchauffe_minimale,
    surpression_de_la_bulle,
)
from polytech.thermique.domain.tables.eau_saturee import (
    TABLE_DE_L_EAU_SATUREE,
    pression_de_saturation,
)

VAPEUR_A_35_BAR = FluideBouillant(
    chaleur_latente=1752200.0,
    liquide=810.0,
    vapeur=17.53,
    tension=0.028,
    viscosite=111.9e-6,
    chaleur_massique=4793.0,
    prandtl=0.852,
)
"""Exercice III.6.1 : l'etat sature a 35 bar, 242,5 degres."""
EAU_A_100 = FluideBouillant(
    chaleur_latente=2256900.0,
    liquide=958.0,
    vapeur=0.598,
    tension=58.78e-3,
    viscosite=283.1e-6,
    chaleur_massique=4218.0,
    prandtl=1.75,
)
"""La table 3.1 a 100 degres ; `rho_v` et `l_v` viennent de l'exercice III.5.1."""
PRESSION_CRITIQUE_EN_BAR = 221.2
"""La derniere ligne de la table 3.1 : 374,15 degres."""


def test_l_equilibre_mecanique_d_une_bulle() -> None:
    """Page 80 : `pi r^2 (p_b - p) = 2 pi r sigma`, d'ou `p_b - p = 2 sigma / r`."""
    rayon, tension = 5e-6, 58.78e-3
    surpression = surpression_de_la_bulle(tension, rayon)
    assert math.pi * rayon**2 * surpression == pytest.approx(
        2 * math.pi * rayon * tension
    )
    with pytest.raises(ValueError, match="rayon nul ou negatif"):
        surpression_de_la_bulle(tension, 0.0)


def test_la_surchauffe_minimale_d_un_liquide() -> None:
    """(6.1) : `t_l - t_sat = (2 sigma / r)(rho_l / (rho_l - rho_v)) dt_sat/dp`.

    Contre la courbe de saturation de la table 3.1 : a la surchauffe
    trouvee, la pression de saturation depasse bien la pression du
    liquide de `Dp_sigma`.
    """
    rayon = 20e-6
    pas = 0.5
    pente = 2 * pas / (pression_de_saturation(100.5) - pression_de_saturation(99.5))
    surchauffe = surchauffe_minimale(EAU_A_100, rayon, pente)
    kelvin = 958.0 / (958.0 - 0.598)
    attendue = 2 * 58.78e-3 / rayon * kelvin
    assert surchauffe == pytest.approx(attendue * pente)
    au_dessus = pression_de_saturation(100.0 + surchauffe) - 101330.0
    assert au_dessus == pytest.approx(attendue, rel=0.03)
    assert surchauffe_minimale(EAU_A_100, rayon / 10, pente) == pytest.approx(
        10 * surchauffe
    )


def test_la_relation_de_rosenhow_et_ses_puissances() -> None:
    """(6.2) : `phi ~ Dt^3`, `K ~ Dt^2`, `K ~ phi^0,67`."""
    simple = surchauffe_de_rosenhow(VAPEUR_A_35_BAR, 1e5)
    octuple = surchauffe_de_rosenhow(VAPEUR_A_35_BAR, 8e5)
    assert octuple == pytest.approx(2 * simple)
    assert (8e5 / octuple) / (1e5 / simple) == pytest.approx(8 ** (2 / 3))
    cuivre_propre = surchauffe_de_rosenhow(VAPEUR_A_35_BAR, 1e5, constante=0.003)
    assert cuivre_propre == pytest.approx(simple * 0.003 / 0.013)
    with pytest.raises(ValueError, match="constante hors de"):
        surchauffe_de_rosenhow(VAPEUR_A_35_BAR, 1e5, constante=0.05)


def test_l_exercice_iii_6_1_un_generateur_de_vapeur() -> None:
    """1662 tubes de 19 mm et 11,4 m : 1131 m^2, 4,54 degres, 44 829 W/m^2 K.

    L'enonce dit « la puissance est de 250 MW » ; le calcul divise
    230 000 000 W par la surface.
    """
    surface = 1662 * math.pi * 0.019 * 11.4
    assert surface == pytest.approx(1131.0, abs=0.5)
    densite = 230e6 / 1131.0
    assert densite == pytest.approx(203360.0, abs=5.0)
    assert pytest.approx(221043.0, abs=5.0) == 250e6 / 1131.0
    surchauffe = surchauffe_de_rosenhow(VAPEUR_A_35_BAR, 203360.0)
    assert surchauffe == pytest.approx(4.54, abs=0.005)
    assert 242.5 + surchauffe == pytest.approx(247.0, abs=0.05)
    assert 203360.0 / surchauffe == pytest.approx(44829.0, rel=1e-3)


def test_l_eau_bouillante_transmet_un_megawatt_par_metre_carre() -> None:
    """Page 86 : « pour l'eau, a la pression atmospherique, de l'ordre de 1 MW/m^2 »."""
    a_cent_degres = TABLE_DE_L_EAU_SATUREE[10]
    assert EAU_A_100.poids_apparent() == pytest.approx(9.81 * (958.0 - 0.598))
    assert a_cent_degres.tension == EAU_A_100.tension
    assert flux_de_zuber(EAU_A_100) == pytest.approx(1e6, rel=0.05)


def test_l_exercice_iii_6_2_le_freon_11() -> None:
    """1,113 bar : `phi_BO = 205 040 W/m^2`, deux fois le flux d'un circuit dense."""
    freon = FluideBouillant(
        chaleur_latente=182300.0, liquide=1479.0, vapeur=5.75, tension=0.019
    )
    critique = flux_de_zuber(freon)
    assert critique == pytest.approx(205040.0, abs=10.0)
    assert critique > 2 * FLUX_DES_CIRCUITS_DENSES
    assert not risque_de_burn_out(Pilotage.FLUX_IMPOSE, 1e5, critique)
    assert risque_de_burn_out(Pilotage.FLUX_IMPOSE, 3e5, critique)
    with pytest.raises(ValueError, match="proprietes du liquide manquantes"):
        surchauffe_de_rosenhow(freon, 1e5)


def test_le_maximum_de_la_figure_6_10_est_dix_fois_trop_bas() -> None:
    """Fig. 6.10 : `phi_BO / p_c` culminerait a « 1600 W/(m^2, bar) ».

    Avec les donnees du syllabus lui-meme, la relation de Zuber rend
    4400 a une atmosphere et 14 800 a 35 bar : le maximum est de l'ordre
    de 16 000, pas de 1600.
    """
    atmospherique = flux_de_zuber(EAU_A_100) / PRESSION_CRITIQUE_EN_BAR
    a_35_bar = flux_de_zuber(VAPEUR_A_35_BAR) / PRESSION_CRITIQUE_EN_BAR
    assert atmospherique == pytest.approx(4400.0, rel=0.03)
    assert a_35_bar == pytest.approx(14800.0, rel=0.01)
    assert a_35_bar > 9 * 1600.0
    assert 35.0 / PRESSION_CRITIQUE_EN_BAR < 1 / 3


def test_l_exercice_iii_6_3_un_crayon_de_reacteur() -> None:
    """(6.4) : `2,57 phi^0,25 exp(-p / 62)`. A 137 bar et 1,4 MW/m^2, 9,7 degres.

    Le syllabus arrondit a 9,5 et conclut a 344,5 degres.
    """
    surchauffe = surchauffe_de_jens_et_lottes(1.4e6, 137.0)
    assert surchauffe == pytest.approx(2.57 * 1.4e6**0.25 * math.exp(-137 / 62))
    assert surchauffe == pytest.approx(9.7, abs=0.01)
    assert 335.0 + surchauffe == pytest.approx(344.5, abs=0.25)
    assert surchauffe_de_jens_et_lottes(1.4e6, 1.0) > 5 * surchauffe
    with pytest.raises(ValueError, match="flux nul ou negatif"):
        surchauffe_de_jens_et_lottes(0.0, 137.0)
