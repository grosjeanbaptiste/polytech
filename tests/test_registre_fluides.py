"""Les travaux de fluides au-dela de l'activite 1, et leur generation.

L'activite d'integration 1 a sa propre commande, inchangee. Les suivants
passent par un registre, comme en thermique : le nom du dossier designe
le jeu de macros. `python -m polytech.fluides` sans argument regenere
l'activite 1 **et** les travaux du registre ; avec un chemin, il ne
regenere que l'activite 1, comme avant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polytech.fluides.__main__ import cibles
from polytech.fluides.application.generation_du_travail import GenererMacrosDuTravail
from polytech.fluides.application.registre import JEUX_DE_MACROS, macros_du_travail
from polytech.fluides.infrastructure.latex import FichierDeMacros

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "MecaniqueFluides"


def test_chaque_jeu_du_registre_a_son_dossier() -> None:
    assert sorted(s for s in JEUX_DE_MACROS if not (TRAVAUX / s).is_dir()) == []


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("activite-99-inexistante")


def test_sans_argument_l_activite_1_et_le_registre_sont_regeneres() -> None:
    travaux = [travail for travail, _ in cibles([])]
    assert travaux == [None, *sorted(JEUX_DE_MACROS)]


def test_avec_un_chemin_seule_l_activite_1_est_regeneree() -> None:
    assert cibles(["ailleurs.tex"]) == [(None, Path("ailleurs.tex"))]


def test_la_commande_publie_le_jeu_du_travail(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    GenererMacrosDuTravail(
        publication=FichierDeMacros(chemin=chemin), travail="activite-integration-2"
    ).executer()
    texte = chemin.read_text(encoding="utf-8")
    assert all(
        f"\\{nom}" in texte for nom in macros_du_travail("activite-integration-2")
    )
