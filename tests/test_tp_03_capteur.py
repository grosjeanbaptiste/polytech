"""Le TP 3 de thermique : le capteur solaire a air, resolu d'un seul coup.

L'officiel itere : il suppose des temperatures, en tire les coefficients,
resout, et recommence. La bibliotheque resout les trois bilans de noeud
**simultanement**, coefficients compris. Elle doit retomber sur ses
temperatures, et sur ses coefficients de premiere iteration quand on les
evalue aux temperatures qu'il avait supposees.

Le second chemin est le bilan global : ce que l'absorbeur recoit du
soleil doit se retrouver, exactement, dans l'air et dans les deux pertes.
"""

from __future__ import annotations

import pytest

from polytech.thermique.application.enonce_tp_03 import (
    KELVIN,
    coefficients,
    resoudre_le_capteur,
)


def test_les_trois_temperatures_rejoignent_l_officiel() -> None:
    etat = resoudre_le_capteur()
    assert etat.vitre - KELVIN == pytest.approx(29.55, abs=0.02)
    assert etat.absorbeur - KELVIN == pytest.approx(83.24, abs=0.05)
    assert etat.isolant - KELVIN == pytest.approx(72.27, abs=0.05)


def test_le_flux_transmis_a_l_air_rejoint_l_officiel_a_un_watt() -> None:
    assert resoudre_le_capteur().flux_vers_l_air == pytest.approx(354.95, abs=1.0)


def test_l_echauffement_de_l_air() -> None:
    assert resoudre_le_capteur().echauffement == pytest.approx(6.29, abs=0.02)


def test_le_bilan_global_se_ferme() -> None:
    etat = resoudre_le_capteur()
    sorties = etat.flux_vers_l_air + etat.perte_avant + etat.perte_arriere
    assert sorties == pytest.approx(etat.absorbe, abs=1e-6)


def test_aux_temperatures_supposees_on_retrouve_les_coefficients_de_l_officiel() -> (
    None
):
    """350 K pour l'absorbeur, 300 K pour la vitre et l'isolant."""
    k1, k2, kcn = coefficients(300.0, 350.0, 300.0)
    assert k1 == pytest.approx(0.78, abs=0.01)
    assert k2 == pytest.approx(7.09, abs=0.01)
    assert kcn == pytest.approx(2.88, abs=0.01)
