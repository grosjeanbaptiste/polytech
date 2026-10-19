"""Activite 2, (H9) : la vitesse de la surface d'un reservoir que l'on remplit.

Le corrige tirait `V_s = (D_o/D_T)^2 V_o` d'un reservoir qui se vide, et en
deduisait `z_max (1 - r^4)`. Le reservoir est alimente : la masse donne
`A_T V_s = Q - A_o V_o`, donc `V_s = r^2 (V_eq - V_o)`, nulle a `z_max`.
(H9) y est exacte ; elle ne pese que pendant la montee (verification
independante du 2026-10-19).
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.application import activite_2_surface as surface


def _part_par_bernoulli(rapport: float, niveau_relatif: float) -> float:
    """Second chemin : resoudre masse + Bernoulli par bissection, sans formule."""
    r4 = rapport**4
    bas, haut = 0.0, 1.0  # V_o / V_eq, entre l'orifice sec et l'equilibre
    for _ in range(200):
        milieu = (bas + haut) / 2
        surface_ = r4 * (1 - milieu) ** 2
        if milieu**2 - surface_ < niveau_relatif:
            bas = milieu
        else:
            haut = milieu
    return r4 * (1 / bas - 1) ** 2


def test_a_la_hauteur_maximale_la_surface_est_immobile() -> None:
    assert surface.part_de_la_vitesse_de_surface(0.1, 1.0) == pytest.approx(0.0)
    assert surface.part_de_la_vitesse_de_surface(1 / 3, 1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("rapport", [0.1, 1 / 3, 0.5])
@pytest.mark.parametrize("niveau", [0.05, 0.25, 0.6, 0.95])
def test_la_part_se_recoupe_par_la_masse_et_bernoulli(
    rapport: float, niveau: float
) -> None:
    attendu = _part_par_bernoulli(rapport, niveau)
    trouve = surface.part_de_la_vitesse_de_surface(rapport, niveau)
    assert trouve == pytest.approx(attendu, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("rapport", [0.1, 1 / 3, 0.5])
@pytest.mark.parametrize("niveau", [0.25, 0.5, 0.9])
def test_des_le_quart_de_la_hauteur_la_part_reste_sous_r_puissance_4(
    rapport: float, niveau: float
) -> None:
    part = surface.part_de_la_vitesse_de_surface(rapport, niveau)
    assert part <= surface.borne_de_la_vitesse_de_surface(rapport)


def test_au_debut_du_remplissage_la_borne_ne_tient_plus() -> None:
    """Orifice presque sec : `V_o` part de zero, et la surface monte vite."""
    assert surface.part_de_la_vitesse_de_surface(1 / 3, 0.01) > (1 / 3) ** 4


def test_la_borne_est_r_puissance_4() -> None:
    assert surface.borne_de_la_vitesse_de_surface(0.1) == pytest.approx(1e-4)
    assert surface.borne_de_la_vitesse_de_surface(0.5) == pytest.approx(1 / 16)


@pytest.mark.parametrize("rapport", [0.0, 1.0, -0.2, 1.5, math.nan])
def test_un_orifice_hors_du_reservoir_leve(rapport: float) -> None:
    with pytest.raises(ValueError, match="rapport des diametres hors de"):
        surface.borne_de_la_vitesse_de_surface(rapport)
    with pytest.raises(ValueError, match="rapport des diametres hors de"):
        surface.part_de_la_vitesse_de_surface(rapport, 0.5)


@pytest.mark.parametrize("niveau", [0.0, -0.1, 1.2, math.nan])
def test_un_niveau_hors_de_la_montee_leve(niveau: float) -> None:
    with pytest.raises(ValueError, match="niveau relatif hors de"):
        surface.part_de_la_vitesse_de_surface(0.1, niveau)
