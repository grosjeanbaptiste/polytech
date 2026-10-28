"""Le TP 1 d'electronique : des champs aux circuits, les valeurs chiffrees.

Le TP est surtout conceptuel ; ses trois exercices chiffres sont refaits
et compares au corrige officiel :

- le ferrite (2.2.5) : l'officiel arrondit mu a 9e-4 avant de s'en
  servir, et trouve 450 uH ; la pente exacte du graphique donne 469 uH ;
- la boucle carree (3.2) : 153 nH -- l'officiel ecrit « mH », mais
  calcule bien avec des nanohenrys ;
- les quatre resistances (3.3) : 4,55 V et 5,5 kohms, recoupes par la
  loi des noeuds au point milieu.
"""

from __future__ import annotations

import pytest

from polytech.electronique.application.enonce_tp_01 import (
    croisements,
    ferrite,
    pont_de_resistances,
)


def test_la_permeabilite_du_ferrite() -> None:
    mu, relative, _, _ = ferrite()
    assert mu == pytest.approx(9.375e-4)
    assert relative == pytest.approx(746, abs=1)


def test_le_courant_maximal_avant_saturation() -> None:
    _, _, courant, _ = ferrite()
    assert courant == pytest.approx(0.160)


def test_l_inductance_exacte_et_celle_de_l_officiel() -> None:
    """469 uH avec la pente exacte ; 450 uH avec mu arrondi a 9e-4."""
    _, _, _, inductance = ferrite()
    assert inductance == pytest.approx(468.75e-6)
    assert pytest.approx(450e-6) == 9e-4 * 20**2 * 25e-6 / 0.02


def test_la_boucle_coupe_vers_dix_megahertz() -> None:
    inductance, basse, haute = croisements()
    assert inductance == pytest.approx(153.3e-9, abs=0.1e-9)
    assert inductance == pytest.approx(153.27e-9, abs=0.01e-9)
    assert basse == pytest.approx(10.38e6, rel=1e-3)
    assert haute == pytest.approx(10.38e12, rel=1e-3)


def test_le_pont_donne_quatre_volts_et_demi() -> None:
    tension, equivalente = pont_de_resistances()
    assert tension == pytest.approx(5 * 10 / 11)
    assert equivalente == pytest.approx(5500.0)


def test_la_loi_des_noeuds_tient_au_point_milieu() -> None:
    """Ce qui arrive par R1 et R2 repart par R3 et R4."""
    tension, _ = pont_de_resistances()
    arrivant = (5 - tension) / 1000 + (5 - tension) / 1000
    partant = tension / 10_000 + tension / 10_000
    assert arrivant == pytest.approx(partant)
