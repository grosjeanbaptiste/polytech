"""Programmation et algorithmique : les programmes C++ du corrige, compiles et lances.

Les programmes des dossiers de `travaux/ProgrammationAlgorithmique/`
sont compiles en C++17, tous avertissements traites comme des erreurs,
puis lances sur la saisie de leur scenario. Ce qu'ils affichent doit
etre, ligne pour ligne, ce que le modele Python annonce. Sans
compilateur sur le poste, ces tests se suspendent, en le disant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polytech.algorithmique.application.releve import releve
from polytech.algorithmique.application.scenarios import SCENARIOS
from polytech.algorithmique.infrastructure.compilation import (
    ProgrammeCpp,
    compilateur,
)

RACINE = Path(__file__).resolve().parent.parent
TRAVAIL = RACINE / "travaux" / "ProgrammationAlgorithmique"


def _exiger_un_compilateur() -> str:
    trouve = compilateur()
    if trouve is None:
        pytest.skip("aucun compilateur C++ (clang++ ou g++) sur ce poste")
    return trouve


@pytest.mark.parametrize("nom", sorted(SCENARIOS))
def test_le_programme_affiche_ce_que_le_modele_annonce(
    nom: str, tmp_path: Path
) -> None:
    _exiger_un_compilateur()
    scenario = SCENARIOS[nom]
    programme = ProgrammeCpp(TRAVAIL / scenario.dossier / scenario.source)
    sortie = programme.executer(scenario.saisie, tmp_path)
    assert releve(sortie) == scenario.modele(scenario.saisie)


@pytest.mark.parametrize("nom", sorted(SCENARIOS))
def test_chaque_scenario_designe_un_programme_du_dossier(nom: str) -> None:
    scenario = SCENARIOS[nom]
    assert (TRAVAIL / scenario.dossier / scenario.source).is_file()


def test_tous_les_programmes_du_dossier_ont_leur_scenario() -> None:
    """Un programme sans scenario serait un corrige que rien n'eprouve."""
    sources = {c.relative_to(TRAVAIL).as_posix() for c in TRAVAIL.glob("*/*.cpp")}
    assert sources == {f"{s.dossier}/{s.source}" for s in SCENARIOS.values()}


def test_un_programme_qui_ne_compile_pas_est_refuse(tmp_path: Path) -> None:
    """`void main` : le C++ standard l'interdit, et clang le dit."""
    _exiger_un_compilateur()
    source = tmp_path / "faux.cpp"
    source.write_text("void main() {}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="compilation refusee : "):
        ProgrammeCpp(source).executer("", tmp_path)


def test_un_programme_qui_echoue_est_refuse(tmp_path: Path) -> None:
    _exiger_un_compilateur()
    source = tmp_path / "echec.cpp"
    source.write_text("int main() { return 3; }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="le programme a rendu le code "):
        ProgrammeCpp(source).executer("", tmp_path)


def test_sans_compilateur_l_execution_le_dit(tmp_path: Path) -> None:
    source = tmp_path / "rien.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="aucun compilateur C"):
        ProgrammeCpp(source, candidats=("compilateur-absent",)).executer("", tmp_path)
    assert compilateur(("compilateur-absent",)) is None
