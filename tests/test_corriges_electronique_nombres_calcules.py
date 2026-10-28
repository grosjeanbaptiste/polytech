r"""Les corriges de lab 3 et lab 5 d'electronique ne tapent plus ces nombres.

Chaque valeur est une macro que `polytech.electronique` calcule : le gain
total de la chaine, ses deux etages, la cadence par voie, les operandes du
bruit en courant, le gain vise du lab 5 et le gain qu'un etage tient.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from polytech.electronique.application.macros_lab_03 import macros_du_labo_trois
from polytech.electronique.application.macros_lab_05 import macros_du_labo_cinq

TRAVAUX: Final = Path(__file__).resolve().parent.parent / "travaux"
LAB_3: Final = TRAVAUX / "ElectroniqueInstrumentation" / "lab-03-instrumentation"
LAB_5: Final = TRAVAUX / "ElectroniqueInstrumentation" / "lab-05-amplificateur-audio"


def _source(dossier: Path, nom: str) -> str:
    return (dossier / nom).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("macro", "valeur"),
    [
        ("LabTroisAGainTotal", "5000"),
        ("LabTroisAGainA", "5"),
        ("LabTroisAGainB", "1000"),
        ("LabTroisBGain", "5000"),
        ("LabTroisBGainSecond", "4"),
        ("LabTroisDArret", "150"),
        ("LabTroisDCadenceMax", "30000"),
        ("LabTroisDVoies", "100"),
        ("LabTroisDCadenceVoie", "300"),
        ("LabTroisBSource", "32"),
        ("LabTroisBPireCourant", "100"),
        ("LabTroisBBande", "20000"),
    ],
)
def test_les_macros_du_lab_03(macro: str, valeur: str) -> None:
    assert macros_du_labo_trois()[macro] == valeur


@pytest.mark.parametrize(
    ("macro", "valeur"),
    [
        ("LabCinqCrete", "4"),
        ("LabCinqSourceVolt", "0.025"),
        ("LabCinqGainVise", "160"),
        ("LabCinqGainParEtage", "150"),
        ("LabCinqProduit", "3000"),
        ("LabCinqBandeAudio", "20"),
    ],
)
def test_les_macros_du_lab_05(macro: str, valeur: str) -> None:
    assert macros_du_labo_cinq()[macro] == valeur


@pytest.mark.parametrize(
    "motif",
    [
        r"G_\{tot\} = 5000",
        r"G_B = 5000/5 = 1000",
        r"\\times 5\$",
        r"\\times 1000\$",
        r"G = \\qty\{5\}\{\\volt\}/\\qty\{1\}\{\\milli\\volt\}\s*= 5000",
        r"\\num\{30000\}/100 = \\qty\{300\}",
        r"\(32 \\times 100\\times10\^\{-15\}\)\^2 \\times 20\\,000",
    ],
)
def test_le_lab_03_ne_tape_plus_ces_expressions(motif: str) -> None:
    assert re.search(motif, _source(LAB_3, "lab_03_instrumentation.tex")) is None


@pytest.mark.parametrize(
    "motif",
    [r"4/0\{,\}025 = 160", r"3\\,000/20 = 150", r"160 dépasse 150"],
)
def test_le_lab_05_ne_tape_plus_ces_expressions(motif: str) -> None:
    assert re.search(motif, _source(LAB_5, "lab_05_amplificateur_audio.tex")) is None


def test_les_resultats_publies_portent_les_nouvelles_macros() -> None:
    publie = _source(LAB_3, "resultats.tex") + _source(LAB_5, "resultats.tex")
    for macro in (*macros_du_labo_trois(), *macros_du_labo_cinq()):
        assert f"\\newcommand{{\\{macro}}}" in publie, macro
