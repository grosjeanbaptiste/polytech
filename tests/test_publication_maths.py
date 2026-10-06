r"""La couche de publication de la mathematique, dupliquee de l'electronique.

Les corriges de mathematique ne tapent aucun resultat : ils consomment
des macros `\newcommand` que la bibliotheque calcule -- ici surtout des
*formules*, que `sympy` compose en LaTeX. La couche est dupliquee plutot
que partagee, comme les autres matieres le font : un contexte borne qui
importerait l'application d'un autre cesserait d'etre borne. Ce fichier
eprouve l'adaptateur, la mise en forme et le registre, non les resultats
d'une feuille d'exercices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from polytech.maths.application.formatage import coordonnees, formule, nombre
from polytech.maths.application.registre import JEUX_DE_MACROS, macros_du_travail
from polytech.maths.infrastructure.latex import FichierDeMacros

if TYPE_CHECKING:
    from pathlib import Path


def test_le_fichier_publie_une_macro_par_valeur_triee(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Rang": "2", "Det": "-5"})
    commandes = [
        ligne
        for ligne in chemin.read_text(encoding="utf-8").splitlines()
        if ligne.startswith(r"\newcommand")
    ]
    assert commandes == [r"\newcommand{\Det}{-5}", r"\newcommand{\Rang}{2}"]


def test_l_entete_nomme_la_commande_qui_reconstruit(tmp_path: Path) -> None:
    chemin = tmp_path / "resultats.tex"
    FichierDeMacros(chemin=chemin).publier({"Det": "-5"})
    entete = chemin.read_text(encoding="utf-8").splitlines()[0]
    assert "NE PAS EDITER A LA MAIN" in entete
    assert "polytech.maths" in entete


def test_un_jeu_vide_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aucune macro a publier"):
        FichierDeMacros(chemin=tmp_path / "vide.tex").publier({})


def test_un_nom_non_alphabetique_est_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nom de macro LaTeX invalide"):
        FichierDeMacros(chemin=tmp_path / "r.tex").publier({"Det1": "1"})


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
    assert nombre(-0.0001, 2) == "0"
    with pytest.raises(ValueError, match="nombre de decimales negatif"):
        nombre(1.0, -1)


def test_une_formule_est_composee_en_latex() -> None:
    """Une expression exacte devient le LaTeX que le document inclut en mode maths."""
    x = sp.Symbol("x")
    assert formule(sp.Rational(3, 16) * sp.pi) == r"\frac{3 \pi}{16}"
    assert formule(x**2 / 2) == r"\frac{x^{2}}{2}"
    assert formule(sp.log(x)) == r"\ln{\left(x \right)}"
    assert formule(sp.asin(x)) == r"\arcsin{\left(x \right)}"
    matrice = formule(sp.Matrix([[1, 2]]))
    assert matrice == r"\left(\begin{matrix}1 & 2\end{matrix}\right)"


def test_un_travail_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun jeu de macros"):
        macros_du_travail("exs-99")


def test_chaque_jeu_du_registre_rend_des_macros() -> None:
    """Le registre ne nomme que des jeux qui se calculent."""
    assert JEUX_DE_MACROS
    for slug in JEUX_DE_MACROS:
        assert macros_du_travail(slug)


def test_une_etape_par_parties_s_ecrit_le_crochet_d_abord() -> None:
    r"""`u v - \int v du` : le terme integre, puis l'integrale, signe resolu.

    `sympy` range l'integrale en tete et laisse `- \int (-f)` tel quel ;
    le lecteur attend `u v + \int f`.
    """
    x = sp.Symbol("x")
    etape = x**3 * sp.exp(x) - sp.Integral(3 * x**2 * sp.exp(x), x)
    assert formule(etape) == r"x^{3} e^{x} - \int 3 x^{2} e^{x}\, dx"
    retournee = sp.cos(x) * sp.cosh(x) - sp.Integral(-sp.sin(x) * sp.cosh(x), x)
    attendue = (
        r"\cos{\left(x \right)} \cosh{\left(x \right)} "
        r"+ \int \sin{\left(x \right)} \cosh{\left(x \right)}\, dx"
    )
    assert formule(retournee) == attendue
    seule = sp.Integral(x, x)
    assert formule(seule) == r"\int x\, dx"


def test_une_courbe_devient_une_liste_de_coordonnees() -> None:
    rendu = coordonnees(np.array([0.0, 0.5]), np.array([1.0, -0.25]), 3)
    assert rendu == "(0,1) (0.5,-0.25)"


def test_deux_suites_de_longueurs_differentes_sont_refusees() -> None:
    with pytest.raises(ValueError, match="longueurs"):
        coordonnees(np.array([0.0, 1.0]), np.array([1.0]), 3)


def test_l_unite_imaginaire_s_ecrit_j_comme_dans_la_feuille() -> None:
    assert formule(sp.Rational(1, 2) - 3 * sp.I / 2) == r"\frac{1}{2} - \frac{3 j}{2}"
