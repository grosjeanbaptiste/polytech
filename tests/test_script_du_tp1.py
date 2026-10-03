"""Le code Python que le corrige du TP 1 montre, execute et recoupe.

Le document inclut `travaux/SignauxSystemes/lab-01/tp1.py` tel quel : le
code qu'il montre est donc celui que ce fichier execute. Chaque question
doit redonner ce que la bibliotheque calcule par ses propres fonctions --
sans quoi le lecteur recopierait un script qui ne fait pas ce que le
corrige annonce.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from polytech.signaux.application import enonce_lab_01 as e
from polytech.signaux.domain.echantillons import peigne, porte
from polytech.signaux.domain.spectre_estime import transformee_de_la_porte

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = (
    Path(__file__).resolve().parent.parent / "travaux/SignauxSystemes/lab-01/tp1.py"
)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tp1", SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"script introuvable : {SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q1_le_script_convolue_par_la_porte_de_dix() -> None:
    b, c = _script().question_1()
    assert np.array_equal(b, porte(e.UN_LARGEUR, e.UN_LARGEUR))
    assert c.max() == 10.0


def test_q2_le_script_convolue_par_le_peigne() -> None:
    b, c = _script().question_2()
    assert np.array_equal(b, peigne(e.DEUX_PERIODE, e.DEUX_IMPULSIONS))
    assert len(c) == 1000


def test_q3_le_script_rejoint_la_prediction() -> None:
    frequences, estime = _script().question_3()
    predit = transformee_de_la_porte(e.TROIS_DUREE, frequences)
    assert np.max(np.abs(estime - predit)) < 0.01


def test_q4_le_script_trouve_les_poles_du_circuit() -> None:
    _, poles, _ = _script().question_4()
    assert np.allclose(poles.real, -500.0)


def test_q5_le_script_trouve_deux_poles_instables() -> None:
    _, poles, _ = _script().question_5()
    assert int(np.sum(poles.real > 0)) == 2


def test_q6_le_script_trouve_le_pole_dominant() -> None:
    frequence, tau = _script().question_6()
    assert frequence == pytest.approx(1.0794, abs=1e-4)
    assert tau == pytest.approx(0.9818, abs=1e-4)
