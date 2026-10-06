"""Mathematique, exercices 2 : les courbes que le corrige trace.

Chaque courbe est celle d'une solution que les autres tests controlent ;
ceux-ci verifient que les points publies sont bien pris sur elle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polytech.maths.application import courbes_exs_02 as co


def test_le_champ_quatre_courbes_entre_les_points_critiques() -> None:
    courbes = co.courbes_du_champ()
    assert sorted(courbes) == [-3, -1, 1, 3]
    for depart, (temps, ordonnees) in courbes.items():
        assert (temps[0], temps[-1]) == (-4.0, 4.0)
        milieu = len(temps) // 2
        assert temps[milieu] == pytest.approx(0.0, abs=1e-12)
        assert ordonnees[milieu] == pytest.approx(depart)
        assert np.all(np.abs(ordonnees) < 4)
        assert np.all(np.diff(ordonnees) * depart > 0)


def test_la_logistique_trois_departs() -> None:
    courbes = co.courbes_logistiques()
    assert sorted(courbes) == [0.1, 0.5, 2.5]
    for depart, (temps, ordonnees) in courbes.items():
        assert temps[0] == 0.0
        assert ordonnees[0] == pytest.approx(depart)
        croissance = depart * math.exp(temps[-1])
        assert ordonnees[-1] == pytest.approx(croissance / (1 - depart + croissance))
        assert abs(ordonnees[-1] - 1) < abs(depart - 1)


def test_le_ressort_battements_et_resonance() -> None:
    temps, battements, resonance = co.courbes_du_ressort()
    assert battements[0] == resonance[0] == 0.0
    attendue = (math.sin(temps[-1]) - temps[-1] * math.cos(temps[-1])) / 2
    assert resonance[-1] == pytest.approx(attendue)
    assert np.max(np.abs(battements)) < 1 / (1 - co.PULSATION_VOISINE) * 1.1
    assert np.max(np.abs(resonance)) > 0.45 * temps[-1]


def test_les_creneaux_tendent_vers_l_impulsionnelle() -> None:
    temps, creneaux, impulsionnelle = co.courbes_des_creneaux()
    assert (temps[0], temps[-1]) == (0.0, 4.0)
    assert sorted(creneaux) == [1, 2, 8]
    ecarts = [float(np.max(np.abs(creneaux[n] - impulsionnelle))) for n in (1, 2, 8)]
    assert ecarts[0] > ecarts[1] > ecarts[2]
    assert impulsionnelle[0] == 0.0
    assert temps[0] == 0.0


def test_le_courant_rejoint_son_regime_etabli() -> None:
    temps, courant, etabli = co.courbes_du_courant()
    assert (temps[0], temps[-1]) == (0.0, 15.0)
    assert courant[0] == pytest.approx(0.0, abs=1e-12)
    assert etabli[0] == pytest.approx(0.1)
    assert abs(courant[-1] - etabli[-1]) < 1e-4
    assert np.max(np.abs(etabli)) == pytest.approx(1 / math.sqrt(10), rel=1e-3)


def test_le_portrait_de_phase_part_d_un_cercle_et_va_a_l_origine() -> None:
    """Douze trajectoires, parties de points equirepartis sur un cercle."""
    trajectoires = co.trajectoires_de_phase()
    assert len(trajectoires) == co.TRAJECTOIRES
    for rang, (premiere, seconde) in enumerate(trajectoires):
        angle = 2 * math.pi * rang / co.TRAJECTOIRES
        assert premiere[0] == pytest.approx(co.RAYON * math.cos(angle), abs=1e-9)
        assert seconde[0] == pytest.approx(co.RAYON * math.sin(angle), abs=1e-9)
        assert math.hypot(premiere[-1], seconde[-1]) < 0.02
        rayons = np.hypot(premiere, seconde)
        assert np.all(np.diff(rayons) < 0)


def test_le_champ_de_pentes_un_segment_par_noeud() -> None:
    """Chaque segment est centre sur un noeud entier et porte la pente du champ."""
    segments = co.segments_du_champ()
    assert len(segments) == 9 * 9
    for (x1, y1), (x2, y2) in segments:
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        assert centre_x == pytest.approx(round(centre_x), abs=1e-12)
        assert centre_y == pytest.approx(round(centre_y), abs=1e-12)
        pente = math.sin(math.pi * round(centre_y) / 4)
        assert (y2 - y1) == pytest.approx(pente * (x2 - x1), abs=1e-12)
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(co.LONGUEUR)


def test_les_segments_se_publient_avec_des_ruptures() -> None:
    rendu = co.en_traits_separes(((0.0, 0.0), (1.0, 1.0)), ((2.0, 0.0), (3.0, 0.5)))
    assert rendu == "(0,0) (1,1) (nan,nan) (2,0) (3,0.5)"


def _indicielle(temps: float) -> float:
    """Page 14 : `1/3 - e^(-t)/2 + e^(-3t)/6`, nulle avant l'instant zero."""
    if temps < 0:
        return 0.0
    return 1 / 3 - math.exp(-temps) / 2 + math.exp(-3 * temps) / 6


@pytest.mark.parametrize("hauteur", [1, 2, 8])
def test_le_creneau_se_raccorde_a_la_fin_de_sa_duree(hauteur: int) -> None:
    """`h_n = n [s(t) - s(t - 1/n)]`, le second terme nul tant que `t < 1/n`."""
    temps, creneaux, _ = co.courbes_des_creneaux()
    for instant, valeur in zip(temps, creneaux[hauteur], strict=True):
        attendue = hauteur * (_indicielle(instant) - _indicielle(instant - 1 / hauteur))
        assert valeur == pytest.approx(attendue, rel=1e-9, abs=1e-12)


def test_les_battements_sont_ceux_de_la_pulsation_voisine() -> None:
    temps, battements, _ = co.courbes_du_ressort()
    omega = co.PULSATION_VOISINE
    for rang in (40, 200, -1):
        instant = temps[rang]
        attendue = (omega * math.sin(instant) - math.sin(omega * instant)) / (
            omega**2 - 1
        )
        assert battements[rang] == pytest.approx(attendue, rel=1e-9, abs=1e-12)
