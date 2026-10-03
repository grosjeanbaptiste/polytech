"""Les travaux de thermique au-dela du TP 1, et leur generation.

Le TP 1 a sa propre commande, inchangee. Les suivants passent par un
registre, comme en electromagnetisme : le nom du dossier designe le jeu
de macros. `python -m polytech.thermique` sans argument regenere le TP 1
**et** les travaux du registre ; avec un chemin, il ne regenere que le
TP 1, comme avant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polytech.thermique.__main__ import cibles
from polytech.thermique.application.registre import JEUX_DE_MACROS, macros_du_travail
from polytech.thermique.infrastructure.latex import FichierDeMacros

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "ThermiqueFondements"


def test_chaque_jeu_du_registre_a_son_dossier() -> None:
    assert sorted(s for s in JEUX_DE_MACROS if not (TRAVAUX / s).is_dir()) == []


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("tp-99-inexistant")


def test_sans_argument_le_tp_1_et_le_registre_sont_regeneres() -> None:
    travaux = [travail for travail, _ in cibles([])]
    assert travaux == [None, *sorted(JEUX_DE_MACROS)]


def test_avec_un_chemin_seul_le_tp_1_est_regenere() -> None:
    assert cibles(["ailleurs.tex"]) == [(None, Path("ailleurs.tex"))]


def test_republier_a_l_identique_ne_touche_pas_le_fichier(tmp_path: Path) -> None:
    """Sinon le PDF passerait pour perime sans qu'aucune valeur n'ait change."""
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Aa": "1"})
    date = chemin.stat().st_mtime_ns
    FichierDeMacros(chemin=chemin).publier({"Aa": "1"})
    assert chemin.stat().st_mtime_ns == date
