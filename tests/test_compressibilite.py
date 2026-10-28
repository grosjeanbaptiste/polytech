"""La variation de masse volumique rapportee a la masse volumique d'arret."""

from __future__ import annotations

import pytest

from polytech.fluides.domain.compressibilite import variation_relative_a_l_arret
from polytech.fluides.domain.regimes import (
    ecart_rapporte_a_rho,
    variation_de_masse_volumique,
)
from polytech.maths.domain.comparaison import proches


def test_a_l_arret_la_variation_est_nulle() -> None:
    assert variation_relative_a_l_arret(0.0) == 0.0


def test_elle_vaut_un_moins_l_inverse_du_rapport_d_arret() -> None:
    """`rho / rho_0 = 1 / (1 + v)`, `v` etant la variation rapportee a `rho`."""
    rapportee_a_rho = variation_de_masse_volumique(0.5)
    assert proches(
        variation_relative_a_l_arret(0.5), 1 - 1 / (1 + rapportee_a_rho), 1e-12
    )


def test_elle_reste_sous_un_et_croit_avec_le_mach() -> None:
    valeurs = [variation_relative_a_l_arret(m) for m in (0.1, 1.0, 5.0)]
    assert valeurs == sorted(valeurs)
    assert valeurs[-1] < 1.0


def test_un_mach_negatif_leve() -> None:
    with pytest.raises(ValueError, match="negatif"):
        variation_relative_a_l_arret(-0.1)


def test_l_alias_nomme_la_definition_rapportee_a_rho() -> None:
    assert ecart_rapporte_a_rho is variation_de_masse_volumique
    assert ecart_rapporte_a_rho(0.3) > variation_relative_a_l_arret(0.3)
