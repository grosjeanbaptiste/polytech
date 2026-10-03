r"""La couche de publication de l'electronique, dupliquee des signaux.

Les corriges d'electronique et instrumentation ne tapent aucun nombre :
ils consomment des macros `\newcommand` que la bibliotheque calcule. La
couche est dupliquee plutot que partagee, comme les autres matieres le
font : un contexte borne qui importerait l'application d'un autre
cesserait d'etre borne. Ce fichier eprouve l'adaptateur, le formatage et
le registre, non les resultats d'un TP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from polytech.electronique.application.formatage import coordonnees, nombre
from polytech.electronique.application.registre import macros_du_travail
from polytech.electronique.infrastructure.latex import FichierDeMacros

if TYPE_CHECKING:
    from pathlib import Path


def test_le_fichier_publie_une_macro_par_valeur_triee(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Pole": "-500", "Gain": "5"})
    commandes = [
        ligne
        for ligne in chemin.read_text(encoding="utf-8").splitlines()
        if ligne.startswith(r"\newcommand")
    ]
    assert commandes == [r"\newcommand{\Gain}{5}", r"\newcommand{\Pole}{-500}"]


def test_l_entete_nomme_la_commande_qui_reconstruit(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Gain": "5"})
    entete = chemin.read_text(encoding="utf-8").splitlines()[0]
    assert "NE PAS EDITER A LA MAIN" in entete
    assert "polytech.electronique" in entete


def test_un_jeu_vide_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aucune macro a publier"):
        FichierDeMacros(chemin=tmp_path / "vide.tex").publier({})


def test_un_nom_non_alphabetique_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=tmp_path / "r.tex").publier({"Pole1": "1"})


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


def test_le_nombre_perd_ses_zeros_de_fin() -> None:
    assert nombre(0.20, 2) == "0.2"
    assert nombre(120.0, 3) == "120"


def test_une_courbe_devient_une_liste_de_coordonnees() -> None:
    rendu = coordonnees(np.array([0.0, 0.5]), np.array([1.0, -0.25]), 3)
    assert rendu == "(0,1) (0.5,-0.25)"


def test_deux_suites_de_longueurs_differentes_sont_refusees() -> None:
    with pytest.raises(ValueError, match="longueurs"):
        coordonnees(np.array([0.0, 1.0]), np.array([1.0]), 3)


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("tp-99-inexistant")
