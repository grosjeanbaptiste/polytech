"""Chaque donnee du devoir cote est citee : le passage est dans sa source.

La bibliotheque cite ses sources dans ses docstrings ; ce test retrouve
chaque passage dans le texte converti du document nomme -- ou, pour la
source exterieure, dans son fichier LaTeX. Une citation inventee, ou une
source renommee sans que la citation suive, le fait echouer. Le texte
converti decompose les accents : les deux cotes sont recomposes (NFC).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Final

import pytest

RACINE: Final = Path(__file__).resolve().parent.parent
COURS: Final = RACINE / "cours-markdown"
THERMIQUE: Final = "ThermiqueFondements/EXAMENS/"

CITATIONS: Final = (
    # L'air a 20 degres, introduction du cours de fluides.
    ("MecaniqueFluides/SLIDES/Slides_01_Introduction.md", "Air at 20°C and 1 atm"),
    ("MecaniqueFluides/SLIDES/Slides_01_Introduction.md", "1.83 × 10 –5"),
    ("MecaniqueFluides/SLIDES/Slides_01_Introduction.md", "1.52 × 10 –5"),
    # La courbe C_D(Re) et la couche limite turbulente, aerodynamique.
    (
        "MecaniqueFluides/SLIDES/Slides_08_Aerodynamique.md",
        "pour l’écoulement autour d’une sphère et d’un cylindre",
    ),
    ("MecaniqueFluides/SLIDES/Slides_08_Aerodynamique.md", "moins de trainée"),
    # Les regimes du tube isole, convection forcee (thermique, planche 31).
    (
        "ThermiqueFondements/SLIDES/Slides_09_Convection_forcee.md",
        "décollement (θ≈80 ° )",
    ),
    (
        "ThermiqueFondements/SLIDES/Slides_09_Convection_forcee.md",
        "transition avant décollement (θ≈110 ° )",
    ),
    # La cheminee de four, examen d'aout 2016.
    (THERMIQUE + "Exam_2016_Aout_Exercice_cote_Enonce.md", "de = 0,5 m"),
    (
        THERMIQUE + "Exam_2016_Aout_Exercice_cote_Enonce.md",
        "vent horizontal de 5 m/s Température extérieure Ta = Tenv = -10°C",
    ),
    # La cheminee d'usine, examens 2022 a 2025.
    (
        THERMIQUE + "Exam_2022_2025_Exercices_cotes_resolus.md",
        "vitesse du vent de 5 m _/_ s",
    ),
    (THERMIQUE + "Exam_2022_2025_Exercices_cotes_resolus.md", "1 _._ 040 _m_"),
    (THERMIQUE + "Exam_2022_2025_Exercices_cotes_resolus.md", "vaut _−_ 10"),
    # La serre, examen de janvier 2026.
    (THERMIQUE + "Exam_2026_Janvier.md", "Vitesse du vent : _U_ = 0 _,_ 75 m _/_ s"),
    (THERMIQUE + "Exam_2026_Janvier.md", "cercles de rayon _R_ = 2 _,_ 5 m"),
    (THERMIQUE + "Exam_2026_Janvier.md", "_T∞_ = 25"),
    (
        THERMIQUE + "Exam_2026_Janvier.md",
        "assimilé à un écoulement autour d’un cylindre",
    ),
)

CLEYNEN: Final = RACINE / "sources/externes/CleynenFluidDynamics/11/chap11.tex"
CLEYNEN_9: Final = RACINE / "sources/externes/CleynenFluidDynamics/9/chap9.tex"
CITATIONS_DE_CLEYNEN: Final = (
    r"\psi &=& U_\infty \sin \theta \left(r - \frac{R^2}{r}\right)",
    r"\phi &=&  K \frac{\cos \theta}{r}",
    r"\left.v_\theta\right|_{r=R} &=& - 2 U_\infty \sin \theta",
    "d’Alembert paradox",
)


def _nfc(texte: str) -> str:
    return unicodedata.normalize("NFC", texte)


@pytest.mark.parametrize(("document", "passage"), CITATIONS)
def test_le_passage_cite_est_dans_le_cours(document: str, passage: str) -> None:
    texte = _nfc((COURS / document).read_text(encoding="utf-8"))
    assert _nfc(passage) in texte


@pytest.mark.parametrize("passage", CITATIONS_DE_CLEYNEN)
def test_le_passage_cite_est_dans_cleynen(passage: str) -> None:
    if not CLEYNEN.is_file():
        pytest.skip("sources externes absentes : python -m outils.externes --rapatrier")
    assert passage in CLEYNEN.read_text(encoding="utf-8")


def test_le_sillage_instationnaire_n_est_pas_forcement_turbulent() -> None:
    """Cleynen, chapitre 9 : l'allee de von Karman est instationnaire."""
    if not CLEYNEN_9.is_file():
        pytest.skip("sources externes absentes : python -m outils.externes --rapatrier")
    assert "Not all unsteady flows are turbulent" in CLEYNEN_9.read_text(
        encoding="utf-8"
    )
