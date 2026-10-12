"""Chaque corrige recoit ses propres macros, et pas celles des autres.

La thermique et les fluides n'ont qu'un travail chacun : un seul jeu de
macros y suffisait. L'electromagnetisme en aura vingt et un, et un jeu
commun aurait deux defauts :

* chaque `resultats.tex` porterait les macros des vingt autres exercices,
  donc le test « aucune macro publiee ne reste inutilisee » deviendrait
  intenable ;
* une erreur dans un exercice ferait echouer la generation des vingt
  autres.

D'ou un **registre** : le nom du dossier de travail designe le jeu. Un
dossier sans entree au registre est refuse, et une entree sans dossier
aussi -- les deux listes doivent se correspondre exactement, sinon un
corrige se retrouverait sans macros ou un jeu sans consommateur.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polytech.electromagnetisme.application.registre import (
    JEUX_DE_MACROS,
    macros_du_travail,
)

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "Electromagnetisme"


def test_chaque_jeu_du_registre_a_son_dossier_de_travail() -> None:
    """Un jeu sans consommateur est un calcul fait pour rien."""
    sans_dossier = sorted(
        slug for slug in JEUX_DE_MACROS if not (TRAVAUX / slug).is_dir()
    )
    assert sans_dossier == []


FICHE = "fiche-de-revision"
"""La fiche de revision : generee par `outils.fiches`, sans jeu de macros."""


def test_chaque_dossier_de_travail_a_son_jeu_au_registre() -> None:
    """Un dossier sans jeu ne pourrait pas regenerer ses valeurs.

    C'est la garde qui tiendra la campagne : vingt et un dossiers a venir,
    et chacun doit etre branche sur ses calculs.
    """
    dossiers = sorted(
        p.name
        for p in TRAVAUX.iterdir()
        if p.is_dir() and (p / "activite.toml").is_file() and p.name != FICHE
    )
    assert sorted(set(dossiers) - set(JEUX_DE_MACROS)) == []


def test_chaque_jeu_publie_au_moins_une_macro() -> None:
    """Un jeu vide ferait echouer la publication, mais plus tard."""
    for slug in JEUX_DE_MACROS:
        assert macros_du_travail(slug), slug


def test_les_jeux_ne_partagent_aucun_nom_de_macro() -> None:
    """Deux exercices qui emploieraient le meme nom se melangeraient.

    Le prefixe de chaque jeu l'evite -- `Huit...` pour l'exercice 8 -- et
    ce test le verifie plutot que de compter sur la discipline.
    """
    vus: dict[str, str] = {}
    collisions = []
    for slug in sorted(JEUX_DE_MACROS):
        for nom in macros_du_travail(slug):
            if nom in vus:
                collisions.append(f"{nom} : {vus[nom]} et {slug}")
            vus[nom] = slug
    assert collisions == []


def test_un_travail_inconnu_est_refuse() -> None:
    """Demander les macros d'un dossier qui n'existe pas doit se dire."""
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("exs-99-inexistant")
