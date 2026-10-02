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
    forme_logarithmique,
    forme_signee,
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


def test_republier_un_contenu_identique_ne_touche_pas_le_fichier(
    tmp_path: Path,
) -> None:
    """Sinon la date de modification bouge, et neuf PDF passent pour perimes.

    Defaut constate le jour du dixieme corrige : regenerer les macros
    reecrivait les neuf `resultats.tex` existants a l'identique, poussant
    leur date de modification. La garde « le PDF n'est pas anterieur a
    ses valeurs » tombait alors pour les neuf travaux d'un coup, alors
    qu'aucune valeur n'avait change.

    Le remede n'est pas de recompiler neuf documents a chaque passage,
    c'est de ne pas ecrire quand il n'y a rien a ecrire. La date de
    modification redevient alors ce qu'elle pretend etre : la date du
    dernier **changement** de valeur.
    """
    chemin = tmp_path / "resultats.tex"
    publication = FichierDeMacros(chemin=chemin)
    publication.publier({"Aa": "1"})
    date = chemin.stat().st_mtime_ns
    publication.publier({"Aa": "1"})
    assert chemin.stat().st_mtime_ns == date


def test_republier_un_contenu_different_reecrit_le_fichier(
    tmp_path: Path,
) -> None:
    """Le revers de la garde precedente, sans lequel elle serait nuisible.

    Ne plus ecrire quand rien ne change ne doit pas devenir ne plus
    ecrire du tout : une valeur qui bouge doit arriver dans le fichier,
    sinon le document composerait l'ancienne.
    """
    chemin = tmp_path / "resultats.tex"
    publication = FichierDeMacros(chemin=chemin)
    publication.publier({"Aa": "1"})
    publication.publier({"Aa": "2"})
    assert r"\newcommand{\Aa}{2}" in chemin.read_text(encoding="utf-8")


def test_le_logarithme_naturel_s_ecrit_ln_et_non_log() -> None:
    r"""Convention francaise, et ambiguite reelle : `log` est le decimal.

    Sympy rend `\log` pour le logarithme naturel, ce qui est l'usage
    anglo-saxon. En notation francaise `\log` designe le logarithme
    **decimal** et `\ln` le naturel : composer `\log` dans un corrige
    francais ecrirait donc une autre fonction que celle calculee.

    Le cas se presente au potentiel du cylindre charge, qui est
    logarithmique. Aucune autre macro du depot n'en contient, de sorte
    que ce changement de notation ne touche rien d'existant.
    """
    rendu = expression_latex(sp.log(sp.Symbol("x", positive=True)))
    assert r"\ln" in rendu
    assert r"\log" not in rendu


def test_la_forme_signee_sort_le_moins_devant_la_fraction() -> None:
    """`-(A+B)` doit se composer `-frac{...}` et non `frac{-...}`.

    `sp.together` reunit bien les termes mais laisse le signe **dans le
    numerateur** : `frac{I(-R^2 - y(a-y))}{2 pi R^2 y}`. Les corriges
    manuscrits ecrivent au contraire le moins devant la fraction, et le
    numerateur en positif -- ce qui se lit mieux et evite de chercher ou
    le signe se trouve.

    Le cas se presente aux sept regions de l'exercice 7, dont les trois
    expressions sont toutes negatives.
    """
    rayon, ordonnee = sp.symbols("R y", positive=True)
    expression = -1 / ordonnee - 1 / rayon
    rendu = forme_signee(expression)
    assert rendu.startswith("- \\frac")
    assert "-" not in rendu[len("- \\frac") :]


def test_la_forme_signee_laisse_une_expression_positive_intacte() -> None:
    """Sur une expression positive, elle ne doit rien signer.

    Sans quoi elle introduirait un signe absent de l'original -- une
    mise en forme ne doit jamais changer la valeur, seulement ou le
    signe se lit.
    """
    ordonnee = sp.Symbol("y", positive=True)
    rendu = forme_signee(1 / ordonnee + 2 / ordonnee)
    assert not rendu.startswith("-")


def test_un_multiple_de_logarithme_se_rend_coefficient_fois_logarithme() -> None:
    r"""`k (ln d - ln(a + d))` se compose `k \ln(d / (a + d))`.

    Sympy integre `1/x` en une **difference** de logarithmes, et la
    recombine en un logarithme de puissance -- forme exacte, illisible,
    et que personne n'ecrit. Le rapport voulu est donc donne par
    l'appelant : c'est lui qui sait lequel le manuscrit emploie.
    """
    a, d, k = sp.symbols("a d k", positive=True)
    expression = k * sp.log(d) - k * sp.log(a + d)
    rendu = forme_logarithmique(expression, d / (a + d))
    assert rendu == r"k \ln{\left(\frac{d}{a + d} \right)}"


def test_le_rapport_inverse_change_le_signe_du_coefficient() -> None:
    """`ln(d/(a+d)) = -ln((a+d)/d)` : le meme nombre, ecrit des deux facons."""
    a, d, k = sp.symbols("a d k", positive=True)
    expression = k * sp.log(d) - k * sp.log(a + d)
    rendu = forme_logarithmique(expression, (a + d) / d)
    assert rendu == r"- k \ln{\left(\frac{a + d}{d} \right)}"


def test_une_expression_qui_n_est_pas_un_multiple_du_logarithme_est_refusee() -> None:
    """Sans quoi le « coefficient » contiendrait lui-meme un logarithme.

    L'outil ne doit pas rendre une forme fausse en silence : si
    l'expression n'est pas proportionnelle au logarithme annonce, il le
    dit.
    """
    a, d, k = sp.symbols("a d k", positive=True)
    with pytest.raises(ValueError, match="pas un multiple"):
        forme_logarithmique(k * sp.log(d), d / (a + d))
