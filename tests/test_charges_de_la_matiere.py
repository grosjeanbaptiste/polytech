"""Les charges d'un conducteur : l'electricite generale s'adosse a la chimie.

Oracle : les exemples 1.3.2 et 1.4.1 du syllabus
(`Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`, pages
1-4 a 1-7) -- un millimetre cube de cuivre, `rho = 8960 kg/m3`,
`M = 63,546 g/mol`, `Z = 29`.

Le syllabus n'est pas tout a fait coherent avec lui-meme : il ecrit
`Q+ = 392 C` a l'exemple 1.3.2 et `394 C` a l'exemple 1.4.1, puis
`-13,56 C` et `-27,03 C` pour un et deux electrons perdus, alors que le
second devrait etre le double du premier. Le calcul refait ici donne
`394,3 C`, `-13,60 C` et `-27,19 C` ; les valeurs imprimees s'en ecartent
de moins d'un pour cent.
"""

from __future__ import annotations

import math

import pytest

from polytech.chemistry.domain.atomes import Atome
from polytech.electromagnetisme.domain.matiere import Echantillon

CUIVRE = Atome("Cu", numero_atomique=29, nombre_de_masse=63)
E, N_A = 1.602176634e-19, 6.02214076e23


def _millimetre_cube() -> Echantillon:
    return Echantillon(
        CUIVRE, masse_volumique=8960, masse_molaire=63.546e-3, volume=1e-9
    )


def test_le_nombre_de_protons_est_z_fois_le_nombre_d_atomes() -> None:
    """`N = Z rho V N_A / M`, que le syllabus arrondit a `2,4 . 10^21`."""
    attendu = 29 * 8960e-9 / 63.546e-3 * N_A
    assert math.isclose(_millimetre_cube().protons, attendu)


def test_la_charge_positive_totale() -> None:
    """`Q+ = N e` ; 394,3 C, ou le syllabus imprime 392 puis 394."""
    q_plus = _millimetre_cube().charge_positive
    assert math.isclose(q_plus, 29 * 8960e-9 / 63.546e-3 * N_A * E)
    assert math.isclose(q_plus, 394, rel_tol=0.01)
    assert math.isclose(q_plus, 392, rel_tol=0.01)


def test_l_echantillon_est_neutre() -> None:
    """`Q_t = Q+ + Q- = 0` : « evident a cause de la neutralite des atomes »."""
    echantillon = _millimetre_cube()
    assert echantillon.charge_positive + echantillon.charge_negative == 0


def test_la_charge_mobile_est_n_fois_q_plus_sur_z() -> None:
    """`Delta Q = -n Q+ / Z` ; le syllabus imprime -13,56 C et -27,03 C."""
    echantillon = _millimetre_cube()
    un, deux = echantillon.charge_mobile(1), echantillon.charge_mobile(2)
    assert math.isclose(un, -echantillon.charge_positive / 29)
    assert math.isclose(un, -13.56, rel_tol=0.01)
    assert math.isclose(deux, -27.03, rel_tol=0.01)


def test_deux_electrons_perdus_font_le_double_d_un_seul() -> None:
    """Ce que les deux valeurs imprimees du syllabus ne verifient pas."""
    echantillon = _millimetre_cube()
    assert math.isclose(echantillon.charge_mobile(2), 2 * echantillon.charge_mobile(1))
    assert not math.isclose(-27.03, 2 * -13.56, rel_tol=1e-3)


def test_la_charge_mobile_est_petite_devant_la_charge_totale() -> None:
    """La conclusion de l'exemple 1.4.1 : `n / Z`, soit quelques pour cent."""
    echantillon = _millimetre_cube()
    rapport = abs(echantillon.charge_mobile(2)) / echantillon.charge_positive
    assert math.isclose(rapport, 2 / 29)


def test_un_atome_ne_perd_pas_plus_d_electrons_qu_il_n_en_a() -> None:
    with pytest.raises(ValueError, match="plus d'electrons"):
        _millimetre_cube().charge_mobile(30)


def test_un_volume_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="echantillon sans matiere"):
        Echantillon(CUIVRE, masse_volumique=8960, masse_molaire=63.546e-3, volume=0)
