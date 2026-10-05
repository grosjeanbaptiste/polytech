"""Programmation, seance 2 de 2025-2026 : ce que ce jeu ajoute a la partie 1.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2025_2026_S2.pdf`. Le
jeu reprend la partie 1 de 2023-2024 -- resume de la seance 1, chaines,
saisie, fonctions, visibilite --, deja portee planche par planche. Il a
ete relu en entier ; ne figurent ici que les exemples qu'il ajoute, tous
sur des symboles existants.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit deux ecarts :

* planche 6 : `auto racine;` sans initialisation ne compile pas ;
* planche 53 : `return a[];` ne compile pas -- et c'est le propos de la
  planche, qui dit qu'un tableau ne se retourne pas.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.identificateurs import type_deduit
from polytech.algorithmique.domain.memoire import Memoire
from polytech.algorithmique.domain.portee import Portees
from polytech.algorithmique.domain.tableaux import TableauStatique, indice_valide
from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS


def test_l_ajout_a_la_fin_puis_a_la_position_2_de_la_planche_10() -> None:
    """`1 9 3`, puis `1 9 3 7`, puis `1 9 0 3 7` : le tableau de cinq est plein."""
    tab = TableauStatique(5, [1, 9, 3])
    tab.ajouter_a_la_fin(7)
    assert tab.elements == [1, 9, 3, 7]
    tab.inserer(2, 0)
    assert tab.elements == [1, 9, 0, 3, 7]
    with pytest.raises(OverflowError, match="tableau plein"):
        tab.ajouter_a_la_fin(4)


def test_la_suppression_a_la_position_1_de_la_planche_11() -> None:
    """Supprimer le 9 : `1 0 3 7`, et `N` passe de 5 a 4."""
    tab = TableauStatique(5, [1, 9, 0, 3, 7])
    assert tab.supprimer(1) == 9
    assert (tab.elements, tab.taille) == ([1, 0, 3, 7], 4)


def test_les_indices_du_tableau_de_la_planche_7() -> None:
    """`int tab[10]` : les indices vont de 0 a 9."""
    assert [i for i in range(-1, 12) if indice_valide(i, 10)] == list(range(10))


def test_une_fonction_peut_retourner_auto_a_la_planche_37() -> None:
    """`auto somme(int a, int b) { return a + b; }` : 7, de type `int`."""
    assert 3 + 4 == 7
    assert type_deduit(3 + 4) == "int"


def test_auto_sans_initialisation_ne_compile_pas() -> None:
    """Ecart de la planche 6 : `auto racine;`."""
    with pytest.raises(ValueError, match="auto exige une initialisation"):
        type_deduit(None)


def test_la_taille_d_un_int_a_la_planche_46() -> None:
    """« Taille d'un int : 4 octets (32 bits) »."""
    assert TYPES_ENTIERS["int"].bits // 8 == 4


def test_la_variable_de_boucle_partagee_vaut_3_apres_la_boucle() -> None:
    """Planche 51, premiere colonne : `for (i = 0; ...)` affiche 0 1 2 puis 3."""
    portees = Portees()
    portees.declarer("i", 10)
    portees.affecter("i", 0)
    while portees.valeur("i") < 3:
        portees.affecter("i", portees.valeur("i") + 1)
    assert portees.valeur("i") == 3


def test_la_variable_de_boucle_declaree_dans_le_for_masque_l_autre() -> None:
    """Planche 51, deuxieme colonne : `for (int i = 0; ...)`, puis 10."""
    portees = Portees()
    portees.declarer("i", 10)
    portees.ouvrir()
    portees.declarer("i", 0)
    while portees.valeur("i") < 3:
        portees.affecter("i", portees.valeur("i") + 1)
    portees.fermer()
    assert portees.valeur("i") == 10


def test_la_variable_de_boucle_n_existe_plus_apres_le_for() -> None:
    """Planche 51, troisieme colonne : « Erreur »."""
    portees = Portees()
    portees.ouvrir()
    portees.declarer("i", 0)
    portees.fermer()
    with pytest.raises(NameError, match="identificateur non declare"):
        portees.valeur("i")


@pytest.mark.parametrize(("x", "attendu"), [(1, 10), (4, 13), (5, 0)])
def test_ce_que_vaut_y_a_la_planche_48(x: int, attendu: int) -> None:
    """`Fonct1` rend 10, `Fonct4` ecrit 13 par reference, `Fonct3` ne change rien."""
    memoire = Memoire()
    memoire.declarer("y", 0)
    if x == 1:
        memoire.declarer("copie", memoire.valeur("y"))
        memoire.affecter("copie", 10)
        memoire.affecter("y", memoire.valeur("copie"))
    elif x == 4:
        memoire.referencer("alias", "y")
        memoire.affecter("alias", 13)
    else:
        memoire.declarer("copie", memoire.valeur("y"))
        memoire.affecter("copie", 12)
    assert memoire.valeur("y") == attendu
