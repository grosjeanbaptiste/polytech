"""Le script de mise a jour du corpus, eprouve sur ses refus.

Il enchaine quatre etapes -- rapatrier, transcrire, mettre en LaTeX,
composer -- dont chacune peut echouer. Ce qui compte est qu'un echec
**arrete** la chaine plutot que de laisser la suivante travailler sur un
corpus a moitie fait : une mise en LaTeX lancee apres une transcription
echouee produirait des `.tex` d'un Markdown qui n'existe pas.

Deux garanties sont verifiees a la lecture du script, non en l'executant,
parce qu'elles sont structurelles :

* `--manquants` sur la conversion. Sans lui le convertisseur reecrit tout
  son repertoire de sortie, et sur ce poste une reconversion introduit des
  caracteres de controle dans les documents deja transcrits (AGENTS.md).
* l'etude avant le rapatriement. Elle n'ecrit rien et annonce le volume :
  un telechargement de plusieurs Go ne doit pas partir sans etre chiffre.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

RACINE: Final = Path(__file__).resolve().parent.parent
SCRIPT: Final = RACINE / "outils" / "mettre-a-jour-le-corpus.sh"
BASH: Final = shutil.which("bash") or "/bin/bash"


def _texte() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_le_script_existe_et_est_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_la_syntaxe_est_correcte() -> None:
    acheve = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, "-n", str(SCRIPT)],
        capture_output=True,
        check=False,
    )
    assert acheve.returncode == 0, acheve.stderr.decode()


def test_la_conversion_est_bornee_aux_documents_manquants() -> None:
    """Sans `--manquants`, le script abimerait le corpus qu'il met a jour."""
    assert "--manquants" in _texte()


def test_l_etude_precede_le_rapatriement() -> None:
    """Le volume doit etre annonce avant d'engager le reseau.

    L'etude n'ecrit rien ; la lancer d'abord est le seul moment ou l'on peut
    encore renoncer a plusieurs Go.
    """
    texte = _texte()
    etude = texte.index("-m outils.moodle |")
    rapatriement = texte.index("--rapatrier \\")
    assert etude < rapatriement


def test_le_reseau_peut_etre_saute() -> None:
    """Rattraper le corpus sans rien telecharger doit rester possible."""
    assert "--sans-reseau" in _texte()


def test_un_drapeau_inconnu_est_refuse() -> None:
    """Un drapeau mal ecrit ne doit pas passer pour un nom de matiere."""
    acheve = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, str(SCRIPT), "--sans-resau"],
        capture_output=True,
        text=True,
        cwd=RACINE,
        check=False,
        env=dict(os.environ, PYTHON=sys.executable),
    )
    assert acheve.returncode != 0
    assert "drapeau inconnu" in acheve.stderr


def test_le_script_echoue_bruyamment_a_chaque_etape() -> None:
    """`set -euo pipefail`, et un `die` par etape : aucun echec ne passe."""
    texte = _texte()
    assert "set -euo pipefail" in texte
    assert texte.count("die '") >= 4


def test_le_script_ne_supprime_rien() -> None:
    """Une source otee laisse sa transcription derriere elle.

    Le convertisseur ecrit par-dessus sans purger : oter est une decision de
    l'utilisateur, jamais l'effet d'une mise a jour.
    """
    texte = _texte()
    assert " rm " not in texte
    assert "unlink" not in texte
    assert "-delete" not in texte


def test_un_interpreteur_absent_est_nomme(tmp_path: Path) -> None:
    """Un `python` introuvable doit se dire, non echouer trois lignes plus loin."""
    acheve = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=RACINE,
        env=dict(os.environ, PYTHON=str(tmp_path / "nulle_part")),
        check=False,
    )
    assert acheve.returncode != 0
    assert "interpreteur introuvable" in acheve.stderr


def test_les_quatre_etapes_sont_annoncees() -> None:
    """Un script muet ne se relit pas dans un journal."""
    texte = _texte()
    for etape in ("0/3", "1/3", "2/3", "3/3"):
        assert etape in texte, etape


@pytest.mark.parametrize(
    "outil",
    ["outils.moodle", "outils.conversion", "outils.latex", "composer-les-pdf.sh"],
)
def test_le_script_appelle_les_outils_du_depot(outil: str) -> None:
    """Et non une reimplantation : la regle vit dans les outils, pas ici."""
    assert outil in _texte()


def test_le_script_tient_sans_argument() -> None:
    """Son usage le plus courant, et celui qui echouait.

    Sous `set -u`, bash 3.2 -- celui de macOS -- tient un tableau vide pour
    une variable non definie : `"${matieres[@]}"` levait « unbound variable »
    des qu'aucune matiere n'etait nommee. Le script mourait donc sur
    l'invocation la plus simple, et aucun test ne l'avait vu -- celui de
    l'interpreteur absent passait parce qu'il mourait avant d'y arriver.

    On ne verifie pas ici qu'il va au bout (cela convertirait le corpus),
    mais qu'il franchit le depouillement des arguments.
    """
    acheve = subprocess.run(  # noqa: S603 -- chemin resolu par shutil.which
        [BASH, "-u", "-c", f'set -- ; source "{SCRIPT}" --sans-reseau'],
        capture_output=True,
        text=True,
        cwd=RACINE,
        env=dict(os.environ, PYTHON="/usr/bin/false"),
        check=False,
    )
    assert "unbound variable" not in acheve.stderr
