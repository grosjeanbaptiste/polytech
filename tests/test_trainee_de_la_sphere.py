"""La courbe C_D(Re) de la sphere et la crise de trainee (fascicule 8).

Oracles : la loi de Stokes `24 / Re` (planche de la sphere), la
correlation de Morrison (2013, valable jusqu'a Re = 10^6, lue dans
https://fluids.readthedocs.io/_modules/fluids/drag.html), et la fiche : la
couche limite devient turbulente, la traînée chute vers Re ~ 3.10^5.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.trainee import coefficient_de_stokes
from polytech.fluides.domain.trainee_de_la_sphere import (
    REYNOLDS_CRITIQUE_BALLE_DE_GOLF,
    REYNOLDS_CRITIQUE_SPHERE_LISSE,
    coefficient_de_trainee_de_la_sphere,
    est_apres_la_crise_de_trainee,
)


def test_aux_petits_reynolds_la_courbe_rejoint_la_loi_de_stokes() -> None:
    assert coefficient_de_trainee_de_la_sphere(0.1) == pytest.approx(
        coefficient_de_stokes(0.1), rel=5e-3
    )


@pytest.mark.parametrize(
    ("reynolds", "attendu"),
    [(1e3, 0.4841), (1e4, 0.3936), (1e5, 0.4247), (4e5, 0.0874), (1e6, 0.1415)],
)
def test_la_courbe_redonne_la_correlation_de_morrison(
    reynolds: float, attendu: float
) -> None:
    assert coefficient_de_trainee_de_la_sphere(reynolds) == pytest.approx(
        attendu, abs=5e-4
    )


def test_le_coefficient_est_presque_constant_avant_la_crise() -> None:
    valeurs = [coefficient_de_trainee_de_la_sphere(r) for r in (1e3, 1e4, 1e5, 2e5)]
    assert all(0.3 < v < 0.5 for v in valeurs)


def test_la_crise_de_trainee_divise_le_coefficient_par_plus_de_deux() -> None:
    avant = coefficient_de_trainee_de_la_sphere(2e5)
    apres = coefficient_de_trainee_de_la_sphere(4e5)
    assert avant / apres > 4


def test_la_crise_de_la_sphere_lisse_se_situe_vers_trois_cent_mille() -> None:
    assert coefficient_de_trainee_de_la_sphere(REYNOLDS_CRITIQUE_SPHERE_LISSE) < 0.2
    assert (
        coefficient_de_trainee_de_la_sphere(0.8 * REYNOLDS_CRITIQUE_SPHERE_LISSE) > 0.3
    )


@pytest.mark.parametrize("reynolds", [0.0, -1.0, 1.0001e6])
def test_hors_du_domaine_de_la_correlation_la_courbe_leve(reynolds: float) -> None:
    with pytest.raises(ValueError, match="Re"):
        coefficient_de_trainee_de_la_sphere(reynolds)


def test_la_crise_est_franchie_des_le_reynolds_critique() -> None:
    assert est_apres_la_crise_de_trainee(3e5, REYNOLDS_CRITIQUE_SPHERE_LISSE)
    assert not est_apres_la_crise_de_trainee(2.999e5, REYNOLDS_CRITIQUE_SPHERE_LISSE)


def test_la_crise_de_la_balle_de_golf_est_vers_cinquante_mille() -> None:
    assert REYNOLDS_CRITIQUE_BALLE_DE_GOLF == 5.0e4


def test_les_alveoles_font_franchir_la_crise_au_reynolds_du_jeu() -> None:
    en_vol = 2e5
    assert REYNOLDS_CRITIQUE_BALLE_DE_GOLF < REYNOLDS_CRITIQUE_SPHERE_LISSE
    assert est_apres_la_crise_de_trainee(en_vol, REYNOLDS_CRITIQUE_BALLE_DE_GOLF)
    assert not est_apres_la_crise_de_trainee(en_vol, REYNOLDS_CRITIQUE_SPHERE_LISSE)


@pytest.mark.parametrize(("re", "crit"), [(-1.0, 3e5), (1e5, 0.0), (1e5, -1.0)])
def test_un_reynolds_non_positif_leve_pour_la_crise(re: float, crit: float) -> None:
    with pytest.raises(ValueError, match="Re"):
        est_apres_la_crise_de_trainee(re, crit)
