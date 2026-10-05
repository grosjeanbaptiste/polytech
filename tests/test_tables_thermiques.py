"""Les tables de proprietes du syllabus de thermique : l'eau et l'air sec.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, tables
3.1 (page 45, « UK Steam Tables in SI Units ») et 3.2 (page 49,
« Properties of Dry Air at Low Pressure »), lues au zoom.

Une table transcrite est une lecture. Chaque ligne est donc recalculee
par ce que la table dit d'elle-meme : `Pr = mu c_p / lambda`,
`nu = mu / rho`, `gamma = c_p / c_v`, et, pour l'air, la loi des gaz
parfaits que porte la chimie.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from polytech.chemistry.domain.gaz import (
    AIR_SEC,
    constante_specifique,
    masse_volumique_du_gaz_parfait,
)
from polytech.fluides.domain.proprietes import viscosite_cinematique
from polytech.thermique.domain.tables.air_sec import (
    TABLE_DE_L_AIR_SEC,
    AirSec,
    air_sec,
)
from polytech.thermique.domain.tables.eau_saturee import (
    TABLE_DE_L_EAU_SATUREE,
    EauSaturee,
    eau_saturee,
    pression_de_saturation,
    temperature_de_saturation,
)
from polytech.thermique.domain.tables.interpolation import interpoler

ATMOSPHERE = 101325.0


def test_la_table_de_l_air_couvre_250_a_500_kelvins() -> None:
    temperatures = [ligne.temperature for ligne in TABLE_DE_L_AIR_SEC]
    assert temperatures == [250, 275, 300, 325, 350, 375, 400, 450, 500]


@pytest.mark.parametrize(
    "ligne", TABLE_DE_L_AIR_SEC, ids=lambda ligne: f"{ligne.temperature:g}K"
)
def test_chaque_ligne_de_l_air_se_recoupe(ligne: AirSec) -> None:
    """`c_p - c_v = R / M`, `Pr = mu c_p / lambda`, `rho = p M / (R T)`."""
    ecart = ligne.chaleur_massique - ligne.chaleur_a_volume_constant
    assert ecart == pytest.approx(constante_specifique(AIR_SEC), rel=1e-3)
    prandtl = ligne.viscosite * ligne.chaleur_massique / ligne.conductivite
    assert prandtl == pytest.approx(ligne.prandtl, abs=0.0015)
    parfait = masse_volumique_du_gaz_parfait(ATMOSPHERE, ligne.temperature, AIR_SEC)
    assert ligne.masse_volumique == pytest.approx(parfait, rel=2e-3)


@pytest.mark.parametrize(
    ("temperature", "gamma", "cinematique"),
    [(250.0, 1.401, 1.132e-5), (300.0, 1.400, 1.568e-5), (400.0, 1.395, 2.591e-5)],
)
def test_les_colonnes_derivees_de_l_air(
    temperature: float, gamma: float, cinematique: float
) -> None:
    """`gamma = c_p / c_v` et `nu = mu / rho`, contre les colonnes imprimees."""
    air = air_sec(temperature)
    assert air.gamma() == pytest.approx(gamma, abs=6e-4)
    assert air.viscosite_cinematique() == pytest.approx(cinematique, rel=5e-4)
    attendue = viscosite_cinematique(air.viscosite, air.masse_volumique)
    assert air.viscosite_cinematique() == attendue


def test_l_air_a_37_5_degres_est_celui_de_l_exercice_iii_3_2() -> None:
    """Page 46 : `lambda = 0,0271`, `mu = 1,895e-5`, `Pr = 0,705`."""
    air = air_sec(273.15 + 37.5)
    assert air.conductivite == pytest.approx(0.0271, abs=5e-5)
    assert air.viscosite == pytest.approx(1.895e-5, abs=5e-9)
    assert air.prandtl == pytest.approx(0.705, abs=1e-3)


def test_la_table_de_l_eau_couvre_zero_a_cent_trente_degres() -> None:
    temperatures = [ligne.temperature for ligne in TABLE_DE_L_EAU_SATUREE]
    assert temperatures == [0.01, *range(10, 140, 10)]
    assert eau_saturee(100.0).pression == pytest.approx(101330.0)


@pytest.mark.parametrize(
    "ligne", TABLE_DE_L_EAU_SATUREE, ids=lambda ligne: f"{ligne.temperature:g}C"
)
def test_chaque_ligne_de_l_eau_se_recoupe(ligne: EauSaturee) -> None:
    """`Pr = mu c_p / lambda`, et `mu / nu` est une masse volumique d'eau."""
    prandtl = ligne.viscosite * ligne.chaleur_massique / ligne.conductivite
    assert prandtl == pytest.approx(ligne.prandtl, rel=4e-3)
    masse_volumique = ligne.viscosite / ligne.viscosite_cinematique
    assert 930.0 < masse_volumique < 1002.0


def test_les_colonnes_de_l_eau_varient_sans_a_coup() -> None:
    """Une decimale mal lue ferait un saut : chaque colonne est monotone et lisse."""
    for colonne in ("pression", "tension", "viscosite", "conductivite", "prandtl"):
        valeurs = [getattr(ligne, colonne) for ligne in TABLE_DE_L_EAU_SATUREE[1:]]
        ecarts = [b - a for a, b in pairwise(valeurs)]
        assert all(e > 0 for e in ecarts) or all(e < 0 for e in ecarts), colonne
        seconds = [b - a for a, b in pairwise(ecarts)]
        arrondi = [round(s, 9) for s in seconds]
        assert all(s >= 0 for s in arrondi) or all(s <= 0 for s in arrondi), colonne


def test_l_eau_a_45_degres_est_celle_de_l_exercice_iii_3_1() -> None:
    """Page 46 : `c = 4180`, `lambda = 0,637`, `Pr = 3,95`, `nu = 0,607e-6`."""
    eau = eau_saturee(45.0)
    assert eau.chaleur_massique == pytest.approx(4180.0)
    assert eau.conductivite == pytest.approx(0.637)
    assert eau.prandtl == pytest.approx(3.95, abs=0.006)
    assert eau.viscosite_cinematique == pytest.approx(0.607e-6, abs=6e-10)


def test_l_interpolation_est_lineaire_et_refuse_d_extrapoler() -> None:
    assert air_sec(300.0) == TABLE_DE_L_AIR_SEC[2]
    milieu = air_sec(312.5)
    bas, haut = TABLE_DE_L_AIR_SEC[2], TABLE_DE_L_AIR_SEC[3]
    assert milieu.conductivite == pytest.approx(
        (bas.conductivite + haut.conductivite) / 2
    )
    assert milieu.temperature == 312.5
    with pytest.raises(ValueError, match="hors de la table"):
        air_sec(200.0)
    with pytest.raises(ValueError, match="hors de la table"):
        eau_saturee(140.0)
    with pytest.raises(ValueError, match="table vide"):
        interpoler((), 300.0)


def test_la_pression_de_saturation_suit_une_loi_en_exponentielle_de_1_sur_t() -> None:
    """`ln p_s` est presque lineaire en `1 / T` : la pente derive de 0,9 % par ligne.

    Une pression mal lue romprait cette regularite, que la monotonie
    seule ne voit pas.
    """
    lignes = TABLE_DE_L_EAU_SATUREE
    pentes = [
        math.log(haut.pression / bas.pression)
        / (1 / (haut.temperature + 273.15) - 1 / (bas.temperature + 273.15))
        for bas, haut in pairwise(lignes)
    ]
    for avant, apres in pairwise(pentes):
        assert apres / avant == pytest.approx(0.9917, abs=0.002)
    assert all(pente < 0 for pente in pentes)


def test_la_chaleur_massique_de_l_eau_passe_par_un_minimum() -> None:
    """Vers 35 degres ; la colonne est convexe, ce qu'une ligne mal lue romprait."""
    chaleurs = [ligne.chaleur_massique for ligne in TABLE_DE_L_EAU_SATUREE]
    assert min(chaleurs) == 4179.0
    ecarts = [b - a for a, b in pairwise(chaleurs)]
    assert all(apres > avant for avant, apres in pairwise(ecarts[:3]))
    assert all(apres >= avant for avant, apres in pairwise(ecarts))


def test_la_pression_de_saturation_s_interpole_en_logarithme() -> None:
    """Entre deux lignes, `ln p_s` est lineaire en `1 / T`.

    Exercice III.5.2 : « une table de vapeur donne p = 0,0476 bar » a
    32,03 degres. L'interpolation lineaire de la pression rendrait 0,0488.
    """
    assert pression_de_saturation(100.0) == pytest.approx(101330.0)
    assert pression_de_saturation(32.03) == pytest.approx(4760.0, abs=5.0)
    assert eau_saturee(32.03).pression == pytest.approx(4877.0, abs=5.0)
    with pytest.raises(ValueError, match="temperature hors de la table"):
        pression_de_saturation(200.0)


def test_la_temperature_de_saturation_des_exercices() -> None:
    """Exercices III.3.3 et III.4.1 : 105 degres a 1,2 bar, 120 degres a 2 bar."""
    assert temperature_de_saturation(1.2e5) == pytest.approx(105.0, abs=0.3)
    assert temperature_de_saturation(2.0e5) == pytest.approx(120.0, abs=0.3)
    aller = pression_de_saturation(57.3)
    assert temperature_de_saturation(aller) == pytest.approx(57.3)
    with pytest.raises(ValueError, match="pression hors de la table"):
        temperature_de_saturation(1e6)
