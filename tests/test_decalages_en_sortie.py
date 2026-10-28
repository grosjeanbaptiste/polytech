"""La tension de decalage en sortie d'un AOP : somme de trois contributions.

Non-inverseur : `G (e_0 - R_s I_+) + R_f I_-`. Inverseur : `-(R_f / R_e) V_e
+ (1 + R_f / R_e) e_0 + R_f I_-`. Oracles : TP 6 d'electronique, exercices 5
(LM741 : +10 mV, -9 mV, +0,63 mV) et 7 (inverseur 10 k / 50 k).
"""

from __future__ import annotations

import pytest

from polytech.electronique.domain.decalages_en_sortie import (
    Defauts,
    decalage_de_l_inverseur,
    decalage_du_non_inverseur,
)

LM741 = Defauts(1e-3, 90e-9, 70e-9)


def test_le_non_inverseur_du_lm741_rend_1_63_millivolt() -> None:
    decalage = decalage_du_non_inverseur(
        LM741, vers_la_masse=1e3, contre_reaction=9e3, source=10e3
    )
    assert decalage == pytest.approx(1.63e-3, abs=5e-6)


def test_sans_courant_seule_la_tension_de_decalage_subit_le_gain() -> None:
    decalage = decalage_du_non_inverseur(
        Defauts(1e-3, 0.0, 0.0),
        vers_la_masse=1e3,
        contre_reaction=9e3,
        source=10e3,
    )
    assert decalage == pytest.approx(10e-3, rel=1e-12)


def test_l_inverseur_de_l_exercice_7_rend_9_5_millivolts_puis_33_5() -> None:
    un = decalage_de_l_inverseur(
        0.0, Defauts(1e-3, 0.0, 70e-9), resistance_entree=10e3, contre_reaction=50e3
    )
    cinq = decalage_de_l_inverseur(
        0.0, Defauts(5e-3, 0.0, 70e-9), resistance_entree=10e3, contre_reaction=50e3
    )
    assert (un, cinq) == pytest.approx((9.5e-3, 33.5e-3), abs=5e-6)


def test_l_inverseur_inverse_le_signal_qu_il_recoit() -> None:
    decalage = decalage_de_l_inverseur(
        2e-3, Defauts(0.0, 0.0, 0.0), resistance_entree=10e3, contre_reaction=50e3
    )
    assert decalage == pytest.approx(-10e-3, rel=1e-12)


@pytest.mark.parametrize("resistance", [0.0, -1.0])
def test_une_resistance_d_entree_non_positive_leve(resistance: float) -> None:
    with pytest.raises(ValueError, match="non positives"):
        decalage_de_l_inverseur(
            0.0,
            Defauts(1e-3, 0.0, 0.0),
            resistance_entree=resistance,
            contre_reaction=1e3,
        )
