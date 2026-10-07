r"""La couche de publication de l'optimisation, dupliquee des autres matieres.

Les corriges d'optimisation ne tapent aucun resultat : ils consomment des
macros `\newcommand` que la bibliotheque calcule -- ici des valeurs
**exactes**, rationnels et points. La couche est dupliquee plutot que
partagee : un contexte borne qui importerait l'application d'un autre
cesserait d'etre borne. Ce fichier eprouve l'adaptateur, la mise en forme
et le registre, non les resultats d'un jeu d'exercices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.optimisation.application.formatage import exact, intervalle, point
from polytech.optimisation.application.generation import GenererMacros
from polytech.optimisation.application.registre import (
    JEUX_DE_MACROS,
    macros_du_travail,
)
from polytech.optimisation.infrastructure.latex import FichierDeMacros

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
    assert "polytech.optimisation" in entete


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


def test_une_valeur_exacte_se_compose_en_latex() -> None:
    r"""Un entier en chiffres, une fraction en `\frac`, le signe devant."""
    grand = sp.Symbol("M", positive=True)
    assert exact(-19) == "-19"
    assert exact(sp.Rational(7, 2)) == r"\frac{7}{2}"
    assert exact(sp.Rational(-81, 4)) == r"-\frac{81}{4}"
    assert exact(0.5) == r"\frac{1}{2}"
    assert exact(6 * grand) == "6 M"
    assert exact(sp.oo) == r"\infty"


def test_un_point_est_la_liste_de_ses_coordonnees() -> None:
    assert point((-1, 5)) == r"(-1,\ 5)"
    assert point((sp.Rational(7, 2), 0)) == r"(\frac{7}{2},\ 0)"
    with pytest.raises(ValueError, match="point sans coordonnee"):
        point(())


def test_un_intervalle_omet_sa_borne_infinie() -> None:
    assert intervalle(2, 6) == r"\in [2,\ 6]"
    assert intervalle(-2, sp.oo) == r"\ge -2"
    assert intervalle(-sp.oo, 0) == r"\le 0"
    with pytest.raises(ValueError, match="intervalle sans borne"):
        intervalle(-sp.oo, sp.oo)


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("exs-99")


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
