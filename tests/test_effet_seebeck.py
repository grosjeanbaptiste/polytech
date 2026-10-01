"""La section 6.8 du syllabus d'electricite generale : l'effet Seebeck.

« Dans une chaine metallique dont les extremites seules sont a la meme
temperature, il apparait en general une difference de potentiel entre ces
dernieres. » Le syllabus en donne les deux causes et, surtout, les
**chiffres** -- ce qui en fait un contenu verifiable :

* « on obtient des valeurs allant du microvolt a des dizaines de
  microvolts par degre Celsius » ;
* « pour le cuivre, il apparait une force electromotrice d'environ
  2,2 µV entre des points dont les temperatures different de 1 °C » ;
* « pour une chaine cuivre-constantan-cuivre, la force electromotrice
  resultante est de quelque 40 a 50 µV par degre Celsius ».

Le dernier chiffre explique quelque chose que le cours d'electronique
reprend de son cote : un thermocouple a 100 °C d'ecart ne delivre que
quatre a cinq millivolts, et c'est pourquoi toute chaine
d'instrumentation thermique commence par un amplificateur.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.thermoelectricite import (
    CAUSES_DE_L_EFFET_SEEBECK,
    COEFFICIENT_DU_CUIVRE,
    PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN,
    ecart_depuis_la_tension,
    tension_de_seebeck,
)


def test_le_cuivre_donne_deux_microvolts_par_degre() -> None:
    """« environ 2,2 µV » pour un degre d'ecart, dit le syllabus."""
    assert pytest.approx(2.2e-6) == COEFFICIENT_DU_CUIVRE
    assert tension_de_seebeck(COEFFICIENT_DU_CUIVRE, 1.0) == pytest.approx(2.2e-6)


def test_le_couple_cuivre_constantan_est_vingt_fois_plus_sensible() -> None:
    """40 a 50 µV/°C contre 2,2 : c'est pourquoi on fait des couples.

    Un metal seul donne un signal a la limite du mesurable ; l'interet
    d'un thermocouple est precisement ce facteur vingt, que le syllabus
    chiffre sans le commenter.
    """
    bas, haut = PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN
    assert bas / COEFFICIENT_DU_CUIVRE > 15
    assert haut / COEFFICIENT_DU_CUIVRE < 25


def test_un_thermocouple_a_cent_degres_delivre_quelques_millivolts() -> None:
    """Quatre a cinq millivolts, et c'est l'ordre de grandeur qui compte.

    Il explique ce que le cours d'electronique traite de son cote : une
    telle tension se perd dans le bruit d'une chaine d'acquisition non
    amplifiee, et c'est pourquoi un thermocouple commence toujours par un
    amplificateur d'instrumentation.
    """
    bas, haut = PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN
    assert tension_de_seebeck(bas, 100.0) == pytest.approx(4e-3)
    assert tension_de_seebeck(haut, 100.0) == pytest.approx(5e-3)


def test_la_tension_est_proportionnelle_a_l_ecart() -> None:
    """Le modele du syllabus est lineaire -- « par degre Celsius ».

    La vraie courbe d'un thermocouple ne l'est pas tout a fait, mais le
    syllabus ne donne pas de correction et ce module n'en invente pas.
    """
    double = tension_de_seebeck(COEFFICIENT_DU_CUIVRE, 50.0)
    simple = tension_de_seebeck(COEFFICIENT_DU_CUIVRE, 25.0)
    assert double == pytest.approx(2 * simple)


def test_l_ecart_se_retrouve_depuis_la_tension_mesuree() -> None:
    """C'est le sens dans lequel un thermocouple sert vraiment.

    On mesure une tension, on en deduit l'ecart de temperature.
    L'aller-retour doit etre exact, sans quoi la mesure ne vaudrait rien.
    """
    for ecart in (1.0, 37.5, 250.0):
        tension = tension_de_seebeck(COEFFICIENT_DU_CUIVRE, ecart)
        assert ecart_depuis_la_tension(COEFFICIENT_DU_CUIVRE, tension) == (
            pytest.approx(ecart)
        )


def test_les_deux_causes_que_le_syllabus_nomme() -> None:
    """Forces electromotrices de contact, et gradients de temperature.

    Le syllabus precise que les premieres « n'obeissent plus a la loi
    des chaines » -- et c'est pourquoi un effet apparait la ou une chaine
    isotherme n'en donnerait aucun.
    """
    assert len(CAUSES_DE_L_EFFET_SEEBECK) == 2
    assert "contact" in CAUSES_DE_L_EFFET_SEEBECK[0]
    assert "gradient" in CAUSES_DE_L_EFFET_SEEBECK[1]


def test_un_coefficient_negatif_est_refuse() -> None:
    """Le signe depend du sens de la jonction, non du materiau seul.

    Le syllabus ne donne que des grandeurs ; porter un coefficient
    negatif serait y ajouter une convention de signe qu'il ne pose pas.
    """
    with pytest.raises(ValueError, match="coefficient de Seebeck negatif"):
        tension_de_seebeck(-2.2e-6, 10.0)


def test_un_coefficient_nul_refuse_le_retour_a_l_ecart() -> None:
    """Sans sensibilite, aucune tension ne renseigne sur la temperature."""
    with pytest.raises(ValueError, match="coefficient de Seebeck nul"):
        ecart_depuis_la_tension(0.0, 1e-3)
