"""Le TP 2 de thermique, recalcule et compare au corrige officiel.

Camion : l'officiel linearise le rayonnement en `K_ray` et itere ; la
bibliotheque resout le bilan non lineaire directement. Les deux doivent
donner les memes temperatures et les memes charges.

Four : l'officiel donne les aires d'echange, l'echange gris, la puissance,
la vitesse de chauffe, les radiosites et la temperature des parois. La
bibliotheque les refait par les cordes croisees et le schema a paroi
adiabatique.
"""

from __future__ import annotations

import pytest

from polytech.thermique.application.enonce_tp_02 import (
    PEINTURES,
    aires_du_four,
    enceinte_du_four,
    toit_du_camion,
    vitesse_de_chauffe,
)

KELVIN = 273.15


@pytest.mark.parametrize(
    ("peinture", "convection", "temperature", "charge"),
    [
        ("AZW/LA-II", 15.0, 31.3, 451.0),
        ("MLS-85-SB-C", 15.0, 62.53, 769.0),
        ("AZW/LA-II", 60.0, 30.43, 442.13),
        ("MLS-85-SB-C", 60.0, 41.03, 550.16),
    ],
)
def test_le_camion_rejoint_le_tableau_de_l_officiel(
    peinture: str, convection: float, temperature: float, charge: float
) -> None:
    toit = toit_du_camion(peinture, convection)
    assert toit.temperature() - KELVIN == pytest.approx(temperature, abs=0.02)
    assert toit.flux_vers_l_interieur(10.0) == pytest.approx(charge, abs=0.5)


def test_la_peinture_la_moins_absorbante_est_la_meilleure() -> None:
    charges = {
        nom: toit_du_camion(nom, 15.0).flux_vers_l_interieur(10.0) for nom in PEINTURES
    }
    assert min(charges, key=charges.__getitem__) == "AZW/LA-II"


def test_les_aires_d_echange_noires_du_four() -> None:
    s12, s13, s23 = aires_du_four()
    assert s12 + 1 / (1 / s13 + 1 / s23) == pytest.approx(2.34, abs=0.005)


def test_l_echange_gris_et_la_puissance_du_four() -> None:
    enceinte = enceinte_du_four()
    assert enceinte.echange_gris() == pytest.approx(0.95, abs=0.005)
    assert enceinte.flux() == pytest.approx(101625, rel=2e-3)


def test_la_vitesse_de_chauffe_initiale() -> None:
    assert vitesse_de_chauffe() == pytest.approx(2.59, abs=0.005)


def test_la_radiosite_de_la_resistance_rejoint_l_officiel() -> None:
    j1, _ = enceinte_du_four().radiosites()
    assert j1 == pytest.approx(86266, rel=2e-3)


def test_les_radiosites_tiennent_le_schema() -> None:
    """`J1 - J2 = phi / S1S2_noir` : la resistance geometrique du milieu."""
    enceinte = enceinte_du_four()
    j1, j2 = enceinte.radiosites()
    assert j1 - j2 == pytest.approx(enceinte.flux() / enceinte.echange_noir())
    assert enceinte.flux() == pytest.approx(101632.0, abs=1.0)
    assert enceinte.echange_noir() == pytest.approx(2.338, abs=5e-4)
    assert j1 - j2 == pytest.approx(43466.0, abs=1.0)
    assert j1 == pytest.approx(86232.0, abs=1.0)
    assert j2 == pytest.approx(42766.0, abs=1.0)


def test_la_radiosite_de_l_officiel_contredit_son_propre_flux() -> None:
    """Avec `J2 = 4,19e4`, le schema donnerait 103 700 W/m et non 101 625."""
    enceinte = enceinte_du_four()
    assert enceinte.radiosites()[0] - 4.19e4 == pytest.approx(44332.0, abs=1.0)
    flux_implique = enceinte.flux_pour_la_radiosite_2(4.19e4)
    assert flux_implique > 1.015 * 101625
    assert flux_implique == pytest.approx(103656.0, abs=1.0)
    assert flux_implique / enceinte.flux() - 1 == pytest.approx(0.02, abs=2e-3)


def test_les_parois_sont_a_802_degres() -> None:
    temperature = enceinte_du_four().temperature_de_la_paroi() - KELVIN
    assert temperature == pytest.approx(802.1, abs=0.05)


def test_seule_l_emissivite_de_la_tole_est_un_parametre_du_four() -> None:
    """La paroi adiabatique n'a aucune emissivite : le schema n'en porte pas."""
    enceinte = enceinte_du_four()
    assert enceinte.emissivite_2 == 0.5
    champs = {c for c in enceinte.__dataclass_fields__ if c.startswith("emissivite")}
    assert champs == {"emissivite_1", "emissivite_2"}
