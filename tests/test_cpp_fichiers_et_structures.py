"""Programmation et algorithmique : gestion des fichiers, structures, TP 3.

Sources : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 136 a 158, et
`ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_S3_Gestion_des_fichiers.pdf`,
planches 2 a 39. Ouvrir, ecrire, relire un fichier ; la boucle
`while (!eof)` ; les structures, leur passage par valeur ou par
reference, les tableaux de structures ; les operations du TP 3.

Chaque resultat a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit deux ecarts :

* planche 37 du S3, la fonction `Doublon` laisse des doublons : apres
  avoir efface l'element `j`, elle passe au suivant sans reexaminer celui
  qui a pris sa place. La planche 38 en donne la version juste ;
* planche 153 de la partie 1 (21 du S3), `*a.x = 1` ne compile pas : il
  faut `(*a).x` ou `a->x`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from polytech.algorithmique.domain.enregistrements import (
    Article,
    permuter_deux_a_deux,
    position_de,
    sans_doublons,
    trier_selon,
)
from polytech.algorithmique.domain.fichiers import Disque, tours_de_la_boucle_eof
from polytech.algorithmique.domain.flux_de_sortie import Ecran
from polytech.algorithmique.domain.planches_fautives import (
    doublon_de_la_planche_37,
)
from polytech.algorithmique.domain.tableaux import TableauStatique

PROF = (
    "5\nZProf1;Informatique\nProf2;Informatique\n"
    "AProf3;Mathematique\nProf4;Physique\nProf5;Chimie\n"
)
ARTICLES = (Article("vis", "B"), Article("clou", "A"), Article("ecrou", "C"))
PERMUTER = 2
"""Le choix 2 du menu du TP 3."""


def test_ecrire_puis_relire_le_fichier_de_la_planche_138() -> None:
    """« Donnees », 5, puis cinq lignes `H 1` a `o 5` : relu, `H ; 1` a `o ; 5`."""
    disque = Disque()
    ecriture = Ecran().ecrire("Donnees").endl().ecrire(5).endl()
    for lettre, nombre in zip("Hello", range(1, 6), strict=True):
        ecriture.ecrire(lettre).ecrire(" ").ecrire(nombre).endl()
    disque.ecrire("Mon_fichier.txt", ecriture.texte)
    lecture = disque.lire("Mon_fichier.txt")
    assert lecture.mot() == "Donnees"
    nombre = int(lecture.mot())
    lignes = [f"{lecture.mot()} ; {lecture.mot()}" for _ in range(nombre)]
    assert lignes == ["H ; 1", "e ; 2", "l ; 3", "l ; 4", "o ; 5"]


def test_rouvrir_un_fichier_en_ecriture_ecrase_son_contenu() -> None:
    """Planche 137 : la reouverture ecrase ; avec `ios::app`, elle ecrit a la suite."""
    disque = Disque()
    disque.ecrire("f.txt", "Donnees\n5\n")
    disque.ecrire("f.txt", "Autre\n")
    assert disque.contenu("f.txt") == "Autre\n"
    disque.ecrire("f.txt", "Suite\n", a_la_suite=True)
    assert disque.contenu("f.txt") == "Autre\nSuite\n"
    disque.ecrire("neuf.txt", "x", a_la_suite=True)
    assert disque.contenu("neuf.txt") == "x"


def test_ouvrir_en_lecture_un_fichier_absent_echoue() -> None:
    """Planche 140 : « on verifie si l'ouverture se passe bien »."""
    disque = Disque()
    assert not disque.existe("Etudiant.txt")
    with pytest.raises(FileNotFoundError, match="n'existe pas"):
        disque.lire("Etudiant.txt")
    disque.ecrire("Etudiant.txt", "4\nEtud1\nEtud2\nEtud3\nEtud4\n")
    assert disque.existe("Etudiant.txt")


def test_la_boucle_while_non_eof_fait_un_tour_de_trop() -> None:
    """Planche 139 : un tour de trop, qui affiche des valeurs sans objet."""
    assert tours_de_la_boucle_eof("H 1\ne 2\nl 3\n", champs=2) == 4
    assert tours_de_la_boucle_eof("H 1\ne 2\nl 3", champs=2) == 3
    assert tours_de_la_boucle_eof("", champs=2) == 1


def test_la_boucle_eof_sur_un_enregistrement_incomplet() -> None:
    """Un mot colle a la fin du fichier leve `eof` ; un blanc apres lui, non."""
    assert tours_de_la_boucle_eof("H 1\ne", champs=2) == 2
    assert tours_de_la_boucle_eof("H 1\ne ", champs=2) == 2
    assert tours_de_la_boucle_eof("H 1\ne 2\n\n", champs=2) == 3
    assert tours_de_la_boucle_eof("  ", champs=2) == 1
    assert tours_de_la_boucle_eof("x", champs=2) == 1
    assert tours_de_la_boucle_eof("a b c", champs=1) == 3


def test_le_fichier_prof_de_la_planche_141() -> None:
    """Le nombre, un `ignore`, puis deux `getline` : jusqu'au `;`, puis la ligne."""
    disque = Disque()
    disque.ecrire("PROF.txt", PROF)
    lecture = disque.lire("PROF.txt")
    nombre = int(lecture.mot())
    lecture.ignorer()
    professeurs = [(lecture.ligne(";"), lecture.ligne()) for _ in range(nombre)]
    assert professeurs[0] == ("ZProf1", "Informatique")
    assert professeurs[4] == ("Prof5", "Chimie")
    assert lecture.epuise()


def test_sans_le_ignore_le_premier_nom_commence_par_un_saut_de_ligne() -> None:
    """Le `lecture.ignore()` de la planche 141 absorbe le saut qui suit le nombre."""
    disque = Disque()
    disque.ecrire("PROF.txt", PROF)
    lecture = disque.lire("PROF.txt")
    lecture.mot()
    assert lecture.ligne(";") == "\nZProf1"


def test_la_saisie_d_un_nom_compose_a_la_planche_28_du_s3() -> None:
    """`getline` rend « Van den bossche » ; `cin >>` ne rend que « Van »."""
    disque = Disque()
    disque.ecrire("saisie", "Van den bossche\nVan Cool\n")
    clavier = disque.lire("saisie")
    assert clavier.ligne() == "Van den bossche"
    assert clavier.mot() == "Van"


def test_le_tp_3_ne_trie_qu_un_tableau_deja_saisi() -> None:
    """Planche 39 du S3 : un drapeau dit si la saisie a eu lieu avant d'operer."""
    saisi = False
    tableau: tuple[Article, ...] = ()
    for choix in (PERMUTER, 3):
        if not saisi:
            tableau, saisi = ARTICLES, True
        if choix == PERMUTER:
            tableau = permuter_deux_a_deux(tableau)
    assert saisi
    assert [a.nom for a in tableau] == ["ecrou", "clou", "vis"]


def test_une_structure_passee_par_valeur_n_est_pas_modifiee() -> None:
    """Planches 148 et 153 : par valeur « 00 » ; par retour ou par reference « 12 »."""
    v = Article("", "0")
    copie = replace(v, nom="Smartphone", code="3")
    assert v == Article("", "0")
    assert copie == Article("Smartphone", "3")
    v = copie
    assert v.nom == "Smartphone"


def test_l_affectation_copie_toute_la_structure() -> None:
    """Planche 151, `T[i] = Nouv();` ; TP 3, `Tab[i] = Tab[i+1]` : d'un seul geste."""
    tableau = TableauStatique(10, list(ARTICLES))
    tableau.supprimer(0)
    assert tableau.elements == [Article("clou", "A"), Article("ecrou", "C")]
    tableau.inserer(1, Article("rivet", "D"))
    assert [a.nom for a in tableau.elements] == ["clou", "rivet", "ecrou"]


def test_permuter_deux_a_deux_au_tp_3() -> None:
    """Feuille de TP de 2019, page 7 : le premier avec le dernier, et ainsi de suite.

    Les transparents ne precisent pas « deux a deux » ; la feuille le
    fait. Avec un nombre impair d'elements, celui du milieu reste en place.
    """
    quatre = (*ARTICLES, Article("rivet", "D"))
    assert [a.nom for a in permuter_deux_a_deux(quatre)] == [
        "rivet",
        "ecrou",
        "clou",
        "vis",
    ]
    assert [a.nom for a in permuter_deux_a_deux(ARTICLES)] == ["ecrou", "clou", "vis"]
    assert permuter_deux_a_deux(permuter_deux_a_deux(quatre)) == quatre
    assert permuter_deux_a_deux(()) == ()


def test_trier_selon_le_code_puis_selon_le_nom() -> None:
    """TP 3, points 3 et 4."""
    par_code = trier_selon(ARTICLES, lambda a: a.code)
    assert [a.code for a in par_code] == ["A", "B", "C"]
    par_nom = trier_selon(ARTICLES, lambda a: a.nom)
    assert [a.nom for a in par_nom] == ["clou", "ecrou", "vis"]


def test_le_tri_garde_l_ordre_des_elements_de_meme_cle() -> None:
    """Deux articles de meme code restent dans l'ordre ou ils etaient."""
    articles = (Article("vis", "B"), Article("clou", "A"), Article("axe", "B"))
    assert [a.nom for a in trier_selon(articles, lambda a: a.code)] == [
        "clou",
        "vis",
        "axe",
    ]


def test_chercher_la_position_selon_le_nom_et_le_code() -> None:
    """TP 3, point 7 : les deux champs doivent correspondre."""
    assert position_de(ARTICLES, "clou", "A") == 1
    assert position_de(ARTICLES, "clou", "B") == len(ARTICLES)
    assert position_de(ARTICLES, "rivet", "A") == len(ARTICLES)


def test_effacer_les_doublons() -> None:
    """Planche 38 du S3 : `Ax Ax Ax By By Ax` devient `Ax By`."""
    a, b = Article("A", "x"), Article("B", "y")
    assert sans_doublons((a, a, a, b, b, a)) == (a, b)
    assert sans_doublons((a, Article("A", "y"), b)) == (a, Article("A", "y"), b)
    assert sans_doublons(()) == ()


def test_la_fonction_doublon_de_la_planche_37_en_laisse_passer() -> None:
    """Sur `Ax Ax Ax By By Ax`, elle rend `Ax Ax By` : trois elements, non deux."""
    a, b = Article("A", "x"), Article("B", "y")
    entree = [a, a, a, b, b, a]
    assert doublon_de_la_planche_37(entree) == [a, a, b]
    assert list(sans_doublons(entree)) == [a, b]


def test_la_societe_de_la_planche_152() -> None:
    """« Fortune: 116.11 An de naissance: 1984 »."""
    ecran = Ecran().ecrire(" Fortune: ").ecrire(116.11).ecrire(" ")
    ecran.ecrire("An de naissance: ").ecrire(1984)
    assert ecran.texte == " Fortune: 116.11 An de naissance: 1984"
