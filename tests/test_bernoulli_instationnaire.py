"""Bernoulli instationnaire : le terme `integrale de dV/dt ds` (fiche, piege).

Oracles : la seconde loi de Newton sur la colonne de fluide d'un tuyau
uniforme, `(p_1 - p_2) A = rho A L dV/dt` ; et Bernoulli stationnaire
(`fondamentales.bernoulli.charge`) quand rien n'accelere.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.fondamentales.bernoulli_instationnaire import (
    pression_en_aval_instationnaire,
    terme_instationnaire,
)
from polytech.fluides.domain.grandeurs import PESANTEUR


def test_un_tuyau_uniforme_donne_longueur_fois_acceleration() -> None:
    a, section = 0.5, 0.01
    debit_variation = a * section
    assert terme_instationnaire(debit_variation, [(4.0, section)]) == pytest.approx(
        4.0 * a
    )


def test_les_troncons_s_ajoutent_chacun_avec_sa_section() -> None:
    dq = 0.002
    attendu = dq * (1.0 / 0.01 + 2.0 / 0.02)
    assert terme_instationnaire(dq, [(1.0, 0.01), (2.0, 0.02)]) == pytest.approx(
        attendu
    )


def test_un_ecoulement_stationnaire_a_un_terme_nul() -> None:
    assert terme_instationnaire(0.0, [(3.0, 0.01)]) == 0.0


@pytest.mark.parametrize("troncons", [[], [(0.0, 0.01)], [(1.0, 0.0)], [(-1.0, 0.1)]])
def test_un_troncon_absent_ou_degenere_leve(
    troncons: list[tuple[float, float]],
) -> None:
    with pytest.raises(ValueError, match="troncon"):
        terme_instationnaire(0.1, troncons)


def test_la_colonne_horizontale_obeit_a_newton() -> None:
    rho, longueur, a = 1000.0, 4.0, 0.5
    terme = terme_instationnaire(a * 0.01, [(longueur, 0.01)])
    p2 = pression_en_aval_instationnaire(
        2e5,
        rho,
        vitesse_amont=1.0,
        vitesse_aval=1.0,
        altitude_amont=0.0,
        altitude_aval=0.0,
        terme=terme,
    )
    assert 2e5 - p2 == pytest.approx(rho * longueur * a)


def test_sans_acceleration_la_charge_de_bernoulli_se_conserve() -> None:
    rho, p1, v1, v2, z1, z2 = 1000.0, 2e5, 1.0, 3.0, 5.0, 2.0
    p2 = pression_en_aval_instationnaire(
        p1,
        rho,
        vitesse_amont=v1,
        vitesse_aval=v2,
        altitude_amont=z1,
        altitude_aval=z2,
        terme=0.0,
    )
    avant = p1 + rho * PESANTEUR * z1 + rho * v1**2 / 2
    apres = p2 + rho * PESANTEUR * z2 + rho * v2**2 / 2
    assert apres == pytest.approx(avant)


def test_une_acceleration_positive_abaisse_la_pression_en_aval() -> None:
    stationnaire = pression_en_aval_instationnaire(
        2e5,
        1000.0,
        vitesse_amont=1.0,
        vitesse_aval=1.0,
        altitude_amont=0.0,
        altitude_aval=0.0,
        terme=0.0,
    )
    instationnaire = pression_en_aval_instationnaire(
        2e5,
        1000.0,
        vitesse_amont=1.0,
        vitesse_aval=1.0,
        altitude_amont=0.0,
        altitude_aval=0.0,
        terme=2.0,
    )
    assert instationnaire == pytest.approx(stationnaire - 2000.0)


def test_une_deceleration_releve_la_pression_c_est_le_coup_de_belier() -> None:
    p = pression_en_aval_instationnaire(
        2e5,
        1000.0,
        vitesse_amont=1.0,
        vitesse_aval=1.0,
        altitude_amont=0.0,
        altitude_aval=0.0,
        terme=-2.0,
    )
    assert p == pytest.approx(2e5 + 2000.0)


def test_une_masse_volumique_non_positive_leve() -> None:
    with pytest.raises(ValueError, match="masse volumique"):
        pression_en_aval_instationnaire(
            2e5,
            0.0,
            vitesse_amont=1.0,
            vitesse_aval=1.0,
            altitude_amont=0.0,
            altitude_aval=0.0,
            terme=0.0,
        )
