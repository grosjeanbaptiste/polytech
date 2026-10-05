"""Les reactions chimiques : la formule brute et le bilan des atomes.

Le syllabus de thermique (`ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`,
pages 156 et 172) ecrit le cracking du methane, `CH4 -> C + 2 H2`, pour
expliquer la suie des flammes de diffusion. La reaction vient de la
chimie : elle vit ici, et le cours de thermique l'importe.
"""

from __future__ import annotations

import pytest

from polytech.chemistry.domain.reactions import composition, est_equilibree


def test_la_composition_d_une_formule_brute() -> None:
    assert composition("CH4") == {"C": 1, "H": 4}
    assert composition("H2O") == {"H": 2, "O": 1}
    assert composition("CO2") == {"C": 1, "O": 2}
    assert composition("NH3") == {"N": 1, "H": 3}
    assert composition("C") == {"C": 1}
    assert composition("Cl2") == {"Cl": 2}
    assert composition("CH3COOH") == {"C": 2, "H": 4, "O": 2}


def test_une_formule_illisible_est_refusee() -> None:
    for formule in ("", "ch4", "2H", "H2-O"):
        with pytest.raises(ValueError, match="formule brute illisible"):
            composition(formule)


def test_le_cracking_du_methane_est_equilibre() -> None:
    """`CH4 -> C + 2 H2` : un carbone et quatre hydrogenes de chaque cote."""
    assert est_equilibree({"CH4": 1}, {"C": 1, "H2": 2})
    assert not est_equilibree({"CH4": 1}, {"C": 1, "H2": 1})


def test_la_combustion_du_methane_est_equilibree() -> None:
    """`CH4 + 2 O2 -> CO2 + 2 H2O` : les deux gaz qui rayonnent dans les fumees."""
    assert est_equilibree({"CH4": 1, "O2": 2}, {"CO2": 1, "H2O": 2})
    assert not est_equilibree({"CH4": 1, "O2": 1}, {"CO2": 1, "H2O": 2})


def test_un_coefficient_nul_ou_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="coefficient nul ou negatif"):
        est_equilibree({"CH4": 0}, {"C": 1, "H2": 2})
