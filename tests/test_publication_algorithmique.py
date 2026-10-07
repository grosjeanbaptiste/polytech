r"""La couche de publication de l'algorithmique, dupliquee des autres matieres.

Les corriges de programmation ne tapent aucun resultat : ce qu'un
programme doit afficher vient d'un modele de la bibliotheque, publie en
macros `\newcommand`. La couche est dupliquee plutot que partagee : un
contexte borne qui importerait l'application d'un autre cesserait d'etre
borne. Ce fichier eprouve l'adaptateur, la mise en forme et le registre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polytech.algorithmique.application.formatage import machine, session
from polytech.algorithmique.application.generation import GenererMacros
from polytech.algorithmique.application.registre import (
    JEUX_DE_MACROS,
    macros_du_travail,
)
from polytech.algorithmique.infrastructure.latex import FichierDeMacros

if TYPE_CHECKING:
    from pathlib import Path


def test_le_fichier_publie_une_macro_par_valeur_triee(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Valeur": "2", "Base": "-5"})
    commandes = [
        ligne
        for ligne in chemin.read_text(encoding="utf-8").splitlines()
        if ligne.startswith(r"\newcommand")
    ]
    assert commandes == [r"\newcommand{\Base}{-5}", r"\newcommand{\Valeur}{2}"]


def test_l_entete_nomme_la_commande_qui_reconstruit(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Base": "-5"})
    entete = chemin.read_text(encoding="utf-8").splitlines()[0]
    assert "NE PAS EDITER A LA MAIN" in entete
    assert "polytech.algorithmique" in entete


def test_un_jeu_vide_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aucune macro a publier"):
        FichierDeMacros(chemin=tmp_path / "vide.tex").publier({})


def test_un_nom_non_alphabetique_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=tmp_path / "r.tex").publier({"Base1": "1"})


def test_republier_a_l_identique_ne_touche_pas_le_fichier(tmp_path: Path) -> None:
    """Sinon le PDF passerait pour perime sans qu'aucune valeur n'ait change."""
    chemin = tmp_path / "resultats.tex"
    publication = FichierDeMacros(chemin=chemin)
    publication.publier({"Aa": "1"})
    date = chemin.stat().st_mtime_ns
    publication.publier({"Aa": "1"})
    assert chemin.stat().st_mtime_ns == date


def test_republier_autre_chose_reecrit_le_fichier(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    publication = FichierDeMacros(chemin=chemin)
    publication.publier({"Aa": "1"})
    publication.publier({"Aa": "2"})
    assert r"\newcommand{\Aa}{2}" in chemin.read_text(encoding="utf-8")


def test_un_texte_de_machine_echappe_ses_tirets_bas() -> None:
    r"""`T_P` s'ecrit `T\_P` sous `\texttt` : sans cela LaTeX attend un indice."""
    assert machine("T_P : 11/max") == r"\texttt{T\_P : 11/max}"
    assert machine("PROF* T_P[10]") == r"\texttt{PROF* T\_P[10]}"
    assert machine("50 % & {ok} #1") == r"\texttt{50 \% \& \{ok\} \#1}"


def test_une_session_met_une_ligne_par_resultat() -> None:
    assert session(("TAB : a/x", "POSITION = 0")) == (
        r"\texttt{TAB : a/x} \\ \texttt{POSITION = 0}"
    )
    with pytest.raises(ValueError, match="session sans aucune ligne"):
        session(())


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("tp-99")


def test_chaque_jeu_du_registre_rend_des_macros() -> None:
    """Le registre ne nomme que des jeux qui se calculent."""
    assert JEUX_DE_MACROS
    for slug in JEUX_DE_MACROS:
        assert macros_du_travail(slug)


def test_la_commande_publie_le_jeu_du_travail(tmp_path: Path) -> None:
    """La commande ne rend rien : elle ecrit, par le port qu'on lui donne."""
    chemin = tmp_path / "resultats.tex"
    (travail,) = list(JEUX_DE_MACROS)[:1]
    GenererMacros(
        publication=FichierDeMacros(chemin=chemin), travail=travail
    ).executer()
    publie = chemin.read_text(encoding="utf-8")
    assert all(
        rf"\newcommand{{\{nom}}}" in publie for nom in macros_du_travail(travail)
    )
