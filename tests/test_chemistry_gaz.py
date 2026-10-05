"""La loi des gaz parfaits, et le lien entre la chimie et la mecanique des fluides.

La constante molaire `R` est le produit du nombre d'Avogadro par la
constante de Boltzmann, tous deux exacts depuis 2019. Le cours de
mecanique des fluides ecrit `P = rho R T` avec la constante **specifique**
du gaz, `R / M` : c'est la masse molaire qui fait le pont, et elle vient
de la chimie. Le dernier test relie les deux contextes.
"""

from __future__ import annotations

import math

import pytest

from polytech.chemistry.domain.gaz import (
    AIR_SEC,
    CONSTANTE_DES_GAZ,
    constante_specifique,
    masse_volumique_du_gaz_parfait,
    pression_partielle,
)
from polytech.chemistry.domain.matiere import CONSTANTE_D_AVOGADRO
from polytech.fluides.domain.grandeurs import Pression
from polytech.fluides.domain.statique import gaz as fluides

BOLTZMANN = 1.380649e-23


def test_la_constante_des_gaz_est_avogadro_fois_boltzmann() -> None:
    assert math.isclose(CONSTANTE_DES_GAZ, CONSTANTE_D_AVOGADRO * BOLTZMANN)


def test_la_constante_specifique_de_l_air_vaut_environ_287() -> None:
    """`R / M = 8,314 / 0,02896`, en J/(kg K)."""
    assert math.isclose(constante_specifique(AIR_SEC), 8.314462618 / 0.02896)
    assert 287 <= constante_specifique(AIR_SEC) < 288


def test_la_masse_volumique_suit_p_m_sur_r_t() -> None:
    """Air a 101 325 Pa et 20 degres Celsius : environ 1,2 kg/m3."""
    rho = masse_volumique_du_gaz_parfait(101_325, 293.15, AIR_SEC)
    assert math.isclose(rho, 101_325 * 0.02896 / (8.314462618 * 293.15))
    assert 1.20 <= rho < 1.21


def test_une_temperature_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="temperature"):
        masse_volumique_du_gaz_parfait(101_325, 0.0, AIR_SEC)


def test_une_masse_molaire_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="masse molaire"):
        constante_specifique(-1.0)


def test_la_constante_des_fluides_est_celle_de_la_chimie_arrondie() -> None:
    """Le lien : un seul nombre, que le cours de fluides arrondit a 8,314."""
    assert round(CONSTANTE_DES_GAZ, 3) == fluides.CONSTANTE_DES_GAZ


def test_l_air_des_fluides_est_l_air_sec_de_la_chimie_arrondi() -> None:
    assert round(AIR_SEC, 3) == fluides.AIR.masse_molaire


def test_les_deux_contextes_donnent_la_meme_masse_volumique() -> None:
    """A l'arrondi des deux constantes pres : moins de deux pour mille."""
    chimie = masse_volumique_du_gaz_parfait(101_325, 293.15, AIR_SEC)
    mecanique = fluides.AIR.masse_volumique(Pression(pascals=101_325), 293.15)
    assert math.isclose(chimie, mecanique.kg_par_m3, rel_tol=2e-3)


def test_la_loi_de_dalton() -> None:
    """`p_i = x_i p` : les pressions partielles d'un melange font la pression totale.

    Les fumees de la page 175 du syllabus de thermique, tome 1 : 12,5 % de
    `CO2`, 6,6 % de `H2O`, 1,9 % de `O2` et 79 % de `N2`, sous 1,013 bar.
    """
    fractions = (0.125, 0.066, 0.019, 0.79)
    partielles = [pression_partielle(x, 1.013) for x in fractions]
    assert math.isclose(sum(partielles), 1.013)
    assert math.isclose(partielles[0], 0.1266, abs_tol=5e-5)
    with pytest.raises(ValueError, match="fraction molaire hors de"):
        pression_partielle(1.2, 1.013)
    with pytest.raises(ValueError, match="pression totale negative"):
        pression_partielle(0.5, -1.0)
