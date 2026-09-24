"""Grandeurs du cours II.1 : pression, masse volumique, hauteur."""

import pytest

from fluides.domain.grandeurs import Hauteur, MasseVolumique, Pression


def test_la_pression_se_lit_en_kilopascals() -> None:
    assert Pression(pascals=129_652.0).kilopascals == pytest.approx(129.652)


def test_la_pression_relative_se_mesure_par_rapport_a_une_reference() -> None:
    absolue = Pression(pascals=129_652.0)
    atmospherique = Pression(pascals=85_600.0)
    assert absolue.relative_a(atmospherique).pascals == pytest.approx(44_052.0)


def test_une_pression_absolue_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="pression absolue"):
        Pression(pascals=-1.0)


def test_la_densite_relative_est_sans_dimension() -> None:
    mercure = MasseVolumique(kg_par_m3=13_600.0)
    eau = MasseVolumique(kg_par_m3=1_000.0)
    assert mercure.densite_relative(eau) == pytest.approx(13.6)


def test_une_masse_volumique_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="masse volumique"):
        MasseVolumique(kg_par_m3=0.0)


def test_une_hauteur_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="hauteur"):
        Hauteur(metres=-0.1)
