"""Grandeurs du cours II.1 : pression, masse volumique, hauteur."""

import pytest

from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique, Pression


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


def test_une_hauteur_se_met_a_l_echelle_sans_quitter_son_type() -> None:
    """Reduire une hauteur d'un rapport reste une hauteur.

    L'appelant devait sortir du type pour cela : il lisait `.metres`,
    multipliait, puis reconstruisait une `Hauteur`. Trois gestes dont deux
    n'apprenaient rien, et un flottant nu circulait entre-temps.
    """
    assert Hauteur(metres=6.0).mise_a_l_echelle(0.5) == Hauteur(metres=3.0)


def test_une_hauteur_mise_a_l_echelle_de_un_ne_change_pas() -> None:
    reference = Hauteur(metres=2.5)
    assert reference.mise_a_l_echelle(1.0) == reference


def test_un_rapport_negatif_est_refuse_par_l_invariant_de_la_hauteur() -> None:
    """La garde n'est pas redite : c'est celle du constructeur qui tient.

    Une hauteur negative est interdite a la construction ; la mise a
    l'echelle passe par la, donc elle herite de ce refus sans le repeter.
    """
    with pytest.raises(ValueError, match="hauteur negative interdite"):
        Hauteur(metres=4.0).mise_a_l_echelle(-0.5)
