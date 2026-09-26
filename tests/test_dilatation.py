"""Les masses volumiques de l'enonce dependent d'une temperature qu'il ne donne pas.

Le budget d'incertitude ne recensait que des incertitudes de *lecture*. Or
rho_Hg = 13600 kg/m3 est une valeur de reference proche de 0 C : a 20 C le
mercure vaut environ 13546. Ignorer cette dependance, c'est omettre une
source qui pese autant que toutes les autres reunies.
"""

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.domain.statique.dilatation import (
    MERCURE_DILATATION,
    DilatationThermique,
    variation_relative,
)


def test_le_mercure_se_dilate_d_environ_deux_pour_mille_pour_dix_kelvins() -> None:
    """Beta = 1,81e-4 /K : dix degres font 0,18 %."""
    assert variation_relative(MERCURE_DILATATION, 10.0) == pytest.approx(
        1.81e-3,
        rel=1e-3,
    )


def test_la_masse_volumique_diminue_quand_la_temperature_monte() -> None:
    """Le fluide se dilate : meme masse, plus de volume, moins de densite."""
    chaud = MERCURE_DILATATION.masse_volumique_a(enonce.MERCURE, ecart=20.0)
    assert chaud.kg_par_m3 < enonce.MERCURE.kg_par_m3


def test_l_ecart_a_vingt_degres_retrouve_la_valeur_tabulee() -> None:
    """13600 a 0 C donne environ 13551 a 20 C : la table dit 13546."""
    a_vingt = MERCURE_DILATATION.masse_volumique_a(enonce.MERCURE, ecart=20.0)
    assert a_vingt.kg_par_m3 == pytest.approx(13551.0, abs=5.0)


def test_un_ecart_nul_laisse_la_masse_volumique_inchangee() -> None:
    identique = MERCURE_DILATATION.masse_volumique_a(enonce.MERCURE, ecart=0.0)
    assert identique.kg_par_m3 == pytest.approx(enonce.MERCURE.kg_par_m3)


def test_un_coefficient_negatif_est_refuse() -> None:
    """L'eau au-dessous de 4 C serait un cas a part, hors de ce modele."""
    with pytest.raises(ValueError, match="coefficient"):
        DilatationThermique(coefficient=-1.0, nom="absurde")
