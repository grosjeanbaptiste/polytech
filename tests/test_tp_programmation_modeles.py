"""Programmation et algorithmique : les TP 1 a 6, ce que chaque programme doit afficher.

Source : `ProgrammationAlgorithmique/TP/TP_01_a_06_2019.pdf`. Chaque TP
demande un programme a menu. Un modele Python, bati sur le domaine, lit
la meme saisie et dit ce qui doit s'afficher : c'est l'oracle auquel le
programme C++ du corrige est compare (`test_tp_programmation_cpp.py`).
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.application import tp_conteneurs, tp_heritage
from polytech.algorithmique.application import tp_enregistrements as tp3
from polytech.algorithmique.application import tp_objets as tp4
from polytech.algorithmique.application import tp_tableaux as tp12
from polytech.algorithmique.application.releve import releve
from polytech.algorithmique.application.scenarios import SCENARIOS
from polytech.algorithmique.domain.enregistrements import permuter_deux_a_deux

# --- Le releve ---------------------------------------------------------------


def test_le_releve_ne_garde_que_les_resultats() -> None:
    """Les invites et le menu sont ecartes ; un resultat suit parfois une invite."""
    sortie = (
        " 1_ Saisie\nValeur a ajouter : TAB1 : 9 1 2\nMIN = 1 MAX = 9\n 8_ Quitter\n"
    )
    assert releve(sortie) == ("TAB1 : 9 1 2", "MIN = 1 MAX = 9")


# --- TP 1 et TP 2 ------------------------------------------------------------


def test_tp2_l_exemple_de_la_feuille() -> None:
    """Page 6 : `1 2 3 4 5`, puis `6 1 2 4 5`, puis `4 6 1 4 5`."""
    lignes = tp12.derouler("5\n1\n1 2 3 4 5\n6\n7\n7\n8\n")
    assert lignes[1:4] == ("TAB2 : 1 2 3 4 5", "TAB2 : 6 1 2 4 5", "TAB2 : 4 6 1 4 5")


def test_tp1_la_saisie_de_n_est_redemandee() -> None:
    """0 et 11 sont refuses ; 5 est le premier N dans [1, 10]."""
    lignes = tp12.derouler(SCENARIOS["tp1_tp2"].saisie)
    assert lignes[0] == "TAB1 : 1 2 3 4 5"
    assert lignes[4:8] == (
        "MIN = 1 MAX = 5",
        "TAB1 : 9 1 2 3 4 5",
        "TAB1 : 1 2 3 4 5 9",
        "TAB1 : 2 3 4 5",
    )


def test_tp2_le_resume_sort_a_la_fin_du_programme() -> None:
    """Les valeurs sorties par reference au point 6 sont encore la apres la boucle."""
    lignes = tp12.derouler(SCENARIOS["tp1_tp2"].saisie)
    assert lignes[-1] == "TAB2 au transvasement : MIN = 1 MAX = 5 MOYENNE = 3"


def test_tp1_un_tableau_plein_refuse_l_ajout() -> None:
    lignes = tp12.derouler("10\n1\n1 2 3 4 5 6 7 8 9 10\n3\n99\n8\n")
    assert lignes[1:] == ("TAB1 est plein", "TAB1 : 1 2 3 4 5 6 7 8 9 10")


def test_tp1_un_tableau_vide_le_dit() -> None:
    assert tp12.derouler("1\n5\n6\n7\n8\n") == ("TAB1 est vide",) * 2 + (
        "TAB2 est vide",
    )


# --- TP 3 --------------------------------------------------------------------


def test_tp3_le_scenario_complet() -> None:
    assert tp3.derouler(SCENARIOS["tp3"].saisie) == (
        "TAB : zoe/b ali/c zoe/b bob/a ali/c",
        "TAB : ali/c bob/a zoe/b ali/c zoe/b",
        "TAB : bob/a zoe/b zoe/b ali/c ali/c",
        "TAB : ali/c ali/c bob/a zoe/b zoe/b",
        "TAB : ali/c bob/a zoe/b",
        "POSITION = 2",
        "TAB : bob/a zoe/b",
        "TAB : bob/a eve/z zoe/b",
        "Ajout impossible",
        "TAB : bob/a eve/z zoe/b",
    )


def test_tp3_permuter_c_est_le_premier_avec_le_dernier() -> None:
    """La feuille le dit en toutes lettres ; la bibliotheque portait l'autre lecture."""
    lignes = tp3.derouler("1\n3\na x\nb y\nc z\n2\n9\n")
    assert lignes[1] == "TAB : c/z b/y a/x"
    assert permuter_deux_a_deux(("a", "b", "c")) == ("b", "a", "c")


def test_tp3_une_position_hors_du_tableau_est_refusee() -> None:
    lignes = tp3.derouler("1\n1\na x\n5\n3\n7\nzz q\n9\n")
    assert lignes[1:] == ("Position invalide", "TAB : a/x", "POSITION = -1")


# --- TP 4 --------------------------------------------------------------------


def test_tp4_dix_constructeurs_avant_le_premier_prof() -> None:
    """`PROF T_P[10]` construit dix objets : le premier prof saisi porte l'Id 11."""
    lignes = tp4.derouler(SCENARIOS["tp4"].saisie)
    debut = lignes.index("Declaration de PROF T_P[10] :")
    assert lignes[debut + 1 : debut + 11] == tuple(
        f"+ PROF {i} construit" for i in range(1, 11)
    )
    assert "T_P : 11/max" in lignes
    assert "T_P : 12/lea 11/max" in lignes


def test_tp4_la_case_liberee_garde_sa_copie() -> None:
    """Apres « supprimer le debut », deux cases portent l'Id 11 : deux destructions."""
    lignes = tp4.derouler(SCENARIOS["tp4"].saisie)
    fin = lignes[lignes.index("Fin du programme :") + 1 :]
    assert len(fin) == 10
    assert fin[-2:] == ("- PROF 11 detruit", "- PROF 11 detruit")
    assert "- PROF 12 detruit" not in fin


def test_tp4_dynamique_les_identifiants_commencent_a_un() -> None:
    """Rien n'est construit a la declaration ; `delete` detruit au bon moment."""
    assert tp4.derouler_dynamique(SCENARIOS["tp4_dynamique"].saisie) == (
        "Declaration de PROF* T_P[10] : rien n'est construit",
        "+ PROF 1 construit",
        "T_P : 1/max",
        "+ PROF 2 construit",
        "T_P : 2/lea 1/max",
        "POSITION = 1",
        "POSITION = -1",
        "- PROF 2 detruit",
        "T_P : 1/max",
        "+ PROF 3 construit",
        "T_P : 3/bob 1/max",
        "Fin du programme :",
        "- PROF 3 detruit",
        "- PROF 1 detruit",
    )


# --- TP 5 et TP 6 ------------------------------------------------------------


def test_tp5_le_meme_prof_malgre_un_autre_identifiant() -> None:
    """`operator==` compare le nom et le service : l'Id n'y entre pas."""
    lignes = tp_conteneurs.derouler(SCENARIOS["tp5"].saisie)
    assert lignes[-3:] == ("PRESENT", "PRESENT", "ABSENT")
    assert "PROFS : 2/curie/physique 9/euler/maths 1/gauss/maths" in lignes
    assert "ETUDIANTS : 3/ali 5/bob 7/zoe" in lignes


def test_tp6_les_moyennes_du_doyen() -> None:
    lignes = tp_heritage.derouler(SCENARIOS["tp6"].saisie)
    assert "DOYEN : 1/noether/algebre/doyenne" in lignes
    assert "MOYENNES : zoe=14 ali=11.3333" in lignes
    assert lignes[-1] == "MOYENNES : ali=11.3333"


def test_une_saisie_qui_s_arrete_trop_tot_est_refusee() -> None:
    with pytest.raises(EOFError, match="plus rien a lire"):
        tp12.derouler("5\n1\n1 2\n")
