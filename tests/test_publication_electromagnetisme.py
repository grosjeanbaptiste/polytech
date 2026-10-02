r"""La couche de publication de l'electromagnetisme, et pourquoi elle existe.

Les corriges pas a pas ne tapent aucun nombre : ils consomment des macros
`\newcommand` que la bibliotheque calcule et publie. La thermique et les
fluides avaient cette couche ; l'electromagnetisme non, et c'est ce qui
empechait d'ecrire ses vingt et un corriges « de la meme facon ».

Elle est **dupliquee** plutot que partagee, comme Thevenin l'est deja
entre l'electromagnetisme et l'electronique : un contexte borne qui
importerait l'application d'un autre cesserait d'etre borne. Le prix est
une trentaine de lignes ; le gain est qu'aucune des deux matieres ne peut
casser l'autre.

Ce fichier eprouve l'adaptateur et le port, non les resultats d'un
exercice donne -- ceux-la ont leurs propres tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)
from polytech.electromagnetisme.infrastructure.latex import FichierDeMacros

if TYPE_CHECKING:
    from pathlib import Path


def test_le_fichier_publie_une_macro_par_valeur(tmp_path: Path) -> None:
    r"""Une `\newcommand` par entree, triee par nom.

    Le tri rend le fichier **reproductible** : deux executions donnent le
    meme octet, donc un diff ne montre que ce qui a vraiment change.
    """
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"TensionAVide": "120", "Charge": "5"})
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    commandes = [ligne for ligne in lignes if ligne.startswith(r"\newcommand")]
    assert commandes == [
        r"\newcommand{\Charge}{5}",
        r"\newcommand{\TensionAVide}{120}",
    ]


def test_le_fichier_porte_un_entete_qui_interdit_la_main(tmp_path: Path) -> None:
    """Le fichier est genere : l'editer a la main perdrait la modification.

    L'en-tete nomme la commande qui le reconstruit, pour que le lecteur
    sache quoi relancer plutot que de chercher.
    """
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Tension": "120"})
    entete = chemin.read_text(encoding="utf-8").splitlines()[0]
    assert "NE PAS EDITER A LA MAIN" in entete
    assert "polytech.electromagnetisme" in entete


def test_un_jeu_vide_est_refuse(tmp_path: Path) -> None:
    """Publier zero macro ecrirait un fichier que le document inclut en vain.

    Et le document echouerait plus tard, sur une macro inconnue, loin de
    la cause. Le refus ici nomme la cause.
    """
    with pytest.raises(ValueError, match="aucune macro a publier"):
        FichierDeMacros(chemin=tmp_path / "vide.tex").publier({})


def test_un_nom_de_macro_non_alphabetique_est_refuse(tmp_path: Path) -> None:
    r"""`\newcommand` n'accepte que des lettres dans un nom de commande.

    Un chiffre ou un souligne y produirait une erreur de compilation
    LaTeX difficile a lire -- la garde la transforme en message clair.
    """
    chemin = tmp_path / "resultats.tex"
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=chemin).publier({"Tension1": "120"})
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=chemin).publier({"tension_a_vide": "120"})


def test_le_formatage_retire_les_zeros_de_fin() -> None:
    """`0.20` laisserait croire a une precision que l'enonce ne donne pas."""
    assert nombre(0.2, 2) == "0.2"
    assert nombre(120.0, 3) == "120"
    assert nombre(1.005, 2) == "1"


def test_le_formatage_refuse_un_nombre_de_decimales_negatif() -> None:
    """Il n'y a pas de sens a arrondir a un rang negatif ici."""
    with pytest.raises(ValueError, match="nombre de decimales negatif"):
        nombre(1.0, -1)


def test_l_expression_latex_ne_resimplifie_pas() -> None:
    """Le domaine a deja etabli la forme ; la refaire la defigurerait.

    Une mise en facteur voulue, ou l'ordre des membres d'une egalite,
    porte un sens pedagogique que `simplify` ne connait pas.
    """
    r, i = sp.symbols("R I")
    assert expression_latex(r * i + 0) == "I R"
    garde = sp.UnevaluatedExpr(r + r)
    assert "R" in expression_latex(garde)
