"""Une vitre selective : transparente au soleil, opaque a l'infrarouge.

Ses proprietes spectrales sont en marche d'escalier : sous la longueur
d'onde de coupure elle transmet `tau`, au-dessus elle ne transmet rien.
Ses proprietes **globales** dependent donc de la source : la fraction
d'energie emise sous la coupure, `k(lambda T)`, les pondere.

Oracles : l'officiel du TP 3 (0,0127 et 0,9043 a 6000 K), la conservation
`alpha + tau + rho = 1`, et l'opacite au rayonnement d'une piece.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.rayonnement.vitre_selective import (
    VitreSelective,
    coefficient_entre_plans_gris,
)

VITRE = VitreSelective(transmission_courte=0.917, reflexion=0.083, coupure=3.5e-6)


def test_au_soleil_la_vitre_transmet_quatre_vingt_dix_pour_cent() -> None:
    """La loi integree exacte donne `k = 0,9873` en `lambda T = 21 000 um K`."""
    assert VITRE.transmission(6000.0) == pytest.approx(0.9054, abs=2e-4)
    assert VITRE.absorption(6000.0) == pytest.approx(0.0116, abs=2e-4)


def test_l_officiel_lit_l_abaque_a_un_millieme_pres() -> None:
    """Il prend `k = 0,9861` : 0,9043 et 0,0127, a un millieme de l'exact."""
    assert VITRE.transmission(6000.0) == pytest.approx(0.917 * 0.9861, abs=1.5e-3)


def test_a_la_temperature_d_une_piece_elle_est_opaque() -> None:
    assert VITRE.transmission(300.0) == pytest.approx(0.0, abs=1e-3)
    assert VITRE.absorption(300.0) == pytest.approx(0.917, abs=1e-3)


@pytest.mark.parametrize("temperature", [300.0, 1500.0, 6000.0])
def test_absorption_transmission_et_reflexion_font_un(temperature: float) -> None:
    total = VITRE.absorption(temperature) + VITRE.transmission(temperature) + 0.083
    assert total == pytest.approx(1.0)


def test_une_vitre_qui_transmettrait_plus_que_ce_qu_elle_ne_reflechit_pas() -> None:
    with pytest.raises(ValueError, match="plus reflexion"):
        VitreSelective(transmission_courte=0.95, reflexion=0.1, coupure=3.5e-6)


def test_le_coefficient_entre_plans_redonne_le_flux() -> None:
    """`K (T1 - T2) = sigma (T1^4 - T2^4) / (1/eps1 + 1/eps2 - 1)`."""
    sigma = 5.670374419e-8
    k = coefficient_entre_plans_gris(0.1, 0.917, 356.0, 302.0)
    attendu = sigma * (356.0**4 - 302.0**4) / (1 / 0.1 + 1 / 0.917 - 1)
    assert k * (356.0 - 302.0) == pytest.approx(attendu)


def test_a_temperatures_egales_le_coefficient_est_la_derivee() -> None:
    sigma = 5.670374419e-8
    k = coefficient_entre_plans_gris(1.0, 1.0, 300.0, 300.0)
    assert k == pytest.approx(4 * sigma * 300.0**3)


def test_une_vitre_qui_absorbe_aussi_le_visible_conserve_l_energie() -> None:
    """Celle de l'enonce n'absorbe rien sous la coupure ; celle-ci, un dixieme."""
    teintee = VitreSelective(transmission_courte=0.8, reflexion=0.1, coupure=3.5e-6)
    total = teintee.absorption(6000.0) + teintee.transmission(6000.0) + 0.1
    assert total == pytest.approx(1.0)
