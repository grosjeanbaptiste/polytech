"""La bande d'une chaine d'etages du premier ordre, et le nombre d'etages.

L'oracle ne passe pas par la formule du retrecissement : il multiplie les
reponses complexes de chaque etage et lit leur module a la frequence
visee, comme un analyseur de reseau le ferait.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.cascade_d_etages import (
    bande_de_la_cascade,
    nombre_d_etages_en_cascade,
)


def _module_relatif(etages: int, bande_par_etage: float, frequence: float) -> float:
    reponse = complex(1)
    for _ in range(etages):
        reponse /= 1 + 1j * frequence / bande_par_etage
    return abs(reponse)


@pytest.mark.parametrize("etages", [1, 2, 3, 5])
def test_la_bande_de_la_cascade_est_a_moins_trois_decibels(etages: int) -> None:
    bande = bande_de_la_cascade(etages, 100e3)
    assert math.isclose(_module_relatif(etages, 100e3, bande), 1 / math.sqrt(2))


def test_deux_etages_ne_tiennent_pas_cent_kilohertz() -> None:
    """Gain 150, LM741 : deux etages de racine de 150 ne gardent pas 100 kHz."""
    par_etage = 1.5e6 / math.sqrt(150)
    assert _module_relatif(2, par_etage, 100e3) < 1 / math.sqrt(2)


def test_il_faut_trois_etages_pour_cent_cinquante_a_cent_kilohertz() -> None:
    assert nombre_d_etages_en_cascade(150, 1.5e6, 100e3) == 3


def test_un_gain_faible_tient_en_un_etage() -> None:
    assert nombre_d_etages_en_cascade(10, 1.5e6, 20e3) == 1


def test_une_bande_impossible_est_refusee() -> None:
    """Au-dela du produit gain-bande, aucun nombre d'etages ne suffit."""
    with pytest.raises(ValueError, match="aucune cascade"):
        nombre_d_etages_en_cascade(150, 1.5e6, 2e6)


def test_une_cascade_sans_etage_est_refusee() -> None:
    with pytest.raises(ValueError, match="cascade sans etage"):
        bande_de_la_cascade(0, 100e3)
