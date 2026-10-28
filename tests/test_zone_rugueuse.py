"""La zone completement rugueuse : lambda ne depend plus de Re (fiche, conduite).

Oracle : Colebrook (planche 22, `coefficient_de_colebrook`) a Reynolds
enorme, et Nikuradse (planche 23, `coefficient_de_nikuradse`).
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.pertes_de_charge import (
    coefficient_de_colebrook,
    coefficient_de_nikuradse,
)
from polytech.fluides.domain.zone_rugueuse import (
    REYNOLDS_DE_RUGOSITE_RUGUEUX,
    coefficient_en_zone_rugueuse,
    est_completement_rugueux,
    reynolds_de_rugosite,
)


@pytest.mark.parametrize("rugosite", [1e-4, 1e-3, 1e-2, 5e-2])
def test_colebrook_a_reynolds_enorme_tend_vers_la_limite_rugueuse(
    rugosite: float,
) -> None:
    limite = coefficient_en_zone_rugueuse(rugosite)
    assert coefficient_de_colebrook(1e12, rugosite) == pytest.approx(limite, rel=1e-3)


def test_en_zone_rugueuse_lambda_ne_depend_pas_du_reynolds() -> None:
    premier = coefficient_de_colebrook(1e9, 0.05)
    second = coefficient_de_colebrook(1e11, 0.05)
    assert premier == pytest.approx(second, rel=1e-3)


def test_la_limite_est_celle_de_nikuradse_au_pour_cent_pres() -> None:
    assert coefficient_en_zone_rugueuse(0.01) == pytest.approx(
        coefficient_de_nikuradse(0.01), rel=1e-2
    )


def test_la_limite_ne_depend_que_de_la_rugosite_relative() -> None:
    assert coefficient_en_zone_rugueuse(0.01) != coefficient_en_zone_rugueuse(0.02)
    assert coefficient_en_zone_rugueuse(0.01) == pytest.approx(
        0.0379, abs=1e-4
    )  # le diagramme de Moody lit ~0,038


@pytest.mark.parametrize("rugosite", [0.0, -0.01, 1.0, 2.0])
def test_une_rugosite_hors_de_zero_un_leve(rugosite: float) -> None:
    with pytest.raises(ValueError, match="rugosite"):
        coefficient_en_zone_rugueuse(rugosite)


def test_le_reynolds_de_rugosite_est_re_eps_sur_d_racine_lambda_sur_8() -> None:
    attendu = 1e6 * 0.01 * (0.038 / 8) ** 0.5
    assert reynolds_de_rugosite(1e6, 0.01, 0.038) == pytest.approx(attendu)


def test_au_seuil_de_soixante_dix_la_paroi_est_completement_rugueuse() -> None:
    assert REYNOLDS_DE_RUGOSITE_RUGUEUX == 70
    assert est_completement_rugueux(1e6, 0.01)
    assert not est_completement_rugueux(1e4, 0.0001)


def test_le_critere_est_coherent_avec_la_constance_de_lambda() -> None:
    rugosite = 0.01
    re = 1e6
    assert est_completement_rugueux(re, rugosite)
    assert coefficient_de_colebrook(re, rugosite) == pytest.approx(
        coefficient_en_zone_rugueuse(rugosite), rel=2e-2
    )


@pytest.mark.parametrize(("re", "lam"), [(0.0, 0.03), (1e6, 0.0), (-1.0, 0.03)])
def test_le_reynolds_de_rugosite_leve_sur_entree_invalide(
    re: float, lam: float
) -> None:
    with pytest.raises(ValueError, match=r"Reynolds|lambda"):
        reynolds_de_rugosite(re, 0.01, lam)
