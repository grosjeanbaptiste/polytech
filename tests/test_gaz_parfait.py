"""L'hypothese « le poids de l'air est negligeable » doit etre chiffree.

Le corrige annoncait rho_air ~ 1,5 kg/m3 sans dire d'ou venait ce nombre,
ni ce que valait l'erreur commise en negligeant la colonne d'air.
"""

import pytest

from polytech.fluides.domain.grandeurs import Hauteur, Pression
from polytech.fluides.domain.statique.gaz import (
    AIR,
    TEMPERATURE_AMBIANTE,
    part_negligee,
)

RESERVOIR = Pression(pascals=130_000.0)
CIEL_GAZEUX = Hauteur(metres=0.5)


def test_la_masse_volumique_de_l_air_vient_de_la_loi_des_gaz_parfaits() -> None:
    """Rho = p M / (R T) : environ 1,5 kg/m3 a 130 kPa."""
    rho = AIR.masse_volumique(RESERVOIR, TEMPERATURE_AMBIANTE)
    assert rho.kg_par_m3 == pytest.approx(1.545, abs=0.01)


def test_l_air_est_bien_plus_leger_que_l_eau_de_trois_ordres() -> None:
    """C'est ce rapport qui justifie (H3), et il doit etre calcule."""
    rho = AIR.masse_volumique(RESERVOIR, TEMPERATURE_AMBIANTE)
    assert 600 < 1_000.0 / rho.kg_par_m3 < 700


def test_la_colonne_d_air_negligee_pese_une_fraction_infime() -> None:
    """Negliger 0,5 m d'air devant 130 kPa coute moins de 0,01 %."""
    part = part_negligee(AIR, RESERVOIR, CIEL_GAZEUX, TEMPERATURE_AMBIANTE)
    assert 0 < part < 1e-4


def test_une_temperature_nulle_ou_negative_est_refusee() -> None:
    """Zero kelvin ferait diverger la masse volumique : on leve."""
    with pytest.raises(ValueError, match="temperature"):
        AIR.masse_volumique(RESERVOIR, 0.0)
