"""Syllabus « Methodologie orientee objet » de 2018 : ce qu'il ajoute aux planches.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Programmation_orientee_objet_2018.pdf`, 73 pages, lues en entier.
Ce fichier porte les quatre notions que les planches n'avaient pas fait
naitre :

* 3.2, page 46 : le **mode de derivation** et son tableau de neuf cases ;
* exercice 1.7, page 35 : la classe `vecteur3d` et sa solution imprimee ;
* 4.4, pages 65 a 68 : `lower_bound`, `upper_bound`, `reverse`, `merge` et
  `splice`, avec l'execution du programme 4.2.

Chaque affichage a ete confirme en compilant le programme (clang++ 21).
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.algorithmes_stl import reverse
from polytech.algorithmique.domain.derivation import (
    INACCESSIBLE,
    mode_par_defaut,
    statut_dans_la_derivee,
)
from polytech.algorithmique.domain.encapsulation import PRIVE, PROTEGE, PUBLIC
from polytech.algorithmique.domain.operateurs import Vecteur3d
from polytech.algorithmique.domain.recherche_triee import (
    lower_bound,
    merge,
    upper_bound,
)
from polytech.algorithmique.domain.sequences import Sequence

VECT1_TRIE = [1, 3, 4, 5, 5, 9, 18]


@pytest.mark.parametrize(
    ("mode", "statut", "attendu"),
    [
        (PUBLIC, PUBLIC, PUBLIC),
        (PUBLIC, PROTEGE, PROTEGE),
        (PUBLIC, PRIVE, INACCESSIBLE),
        (PROTEGE, PUBLIC, PROTEGE),
        (PROTEGE, PROTEGE, PROTEGE),
        (PROTEGE, PRIVE, INACCESSIBLE),
        (PRIVE, PUBLIC, PRIVE),
        (PRIVE, PROTEGE, PRIVE),
        (PRIVE, PRIVE, INACCESSIBLE),
    ],
)
def test_les_neuf_cases_du_tableau_de_la_page_46(
    mode: str, statut: str, attendu: str
) -> None:
    """Mode de derivation, statut dans la classe de base, statut dans la derivee."""
    assert statut_dans_la_derivee(mode, statut) == attendu


def test_sans_mode_de_derivation_une_classe_herite_en_prive() -> None:
    """Page 45 : « le compilateur prend par defaut le mot-cle private »."""
    assert mode_par_defaut("class") == PRIVE
    assert mode_par_defaut("struct") == PUBLIC
    assert statut_dans_la_derivee(mode_par_defaut("class"), PUBLIC) == PRIVE


@pytest.mark.parametrize(("mode", "statut"), [("amie", PUBLIC), (PUBLIC, "amie")])
def test_un_mode_ou_un_statut_inconnu_est_refuse(mode: str, statut: str) -> None:
    """Trois mots cles, pour le mode comme pour le statut."""
    with pytest.raises(ValueError, match="ni public, ni protected, ni private"):
        statut_dans_la_derivee(mode, statut)


def test_le_vecteur_a_des_composantes_nulles_par_defaut() -> None:
    """Exercice 1.7 : `v1 (1,2,3)`, `v2 (4,5)` et `w` -- < 4, 5, 0> et < 0, 0, 0>."""
    assert str(Vecteur3d(1, 2, 3)) == "< 1, 2, 3>"
    assert str(Vecteur3d(4, 5)) == "< 4, 5, 0>"
    assert str(Vecteur3d()) == "< 0, 0, 0>"


def test_les_sommes_et_les_differences_de_l_exercice_1_7() -> None:
    """Page 35 : < 5, 7, 3>, < 2, 4, 6>, puis < -3, -3, 3> et < 0, 0, 0>."""
    v1, v2 = Vecteur3d(1, 2, 3), Vecteur3d(4, 5)
    assert str(v1 + v2) == "< 5, 7, 3>"
    assert str(v1 + v1) == "< 2, 4, 6>"
    assert str(v1 - v2) == "< -3, -3, 3>"
    assert v1 - v1 == Vecteur3d()


def test_la_coincidence_de_deux_vecteurs() -> None:
    """Page 35 : « Pas EGALITE » entre v1 et v2, « EGALITE » entre v1 et v1."""
    v1, v2 = Vecteur3d(1, 2, 3), Vecteur3d(4, 5)
    assert v1 != v2
    assert v1 == Vecteur3d(1, 2, 3)
    assert v1 != v1 + v1


def test_ou_inserer_un_5_dans_le_vecteur_trie_du_programme_4_2() -> None:
    """Page 68 : « Pour Vect1 Trie: 5 peut etre insere de 3 a 5 »."""
    assert (lower_bound(VECT1_TRIE, 5), upper_bound(VECT1_TRIE, 5)) == (3, 5)
    assert (lower_bound(VECT1_TRIE, 6), upper_bound(VECT1_TRIE, 6)) == (5, 5)
    assert (lower_bound(VECT1_TRIE, 0), upper_bound(VECT1_TRIE, 99)) == (0, 7)


def test_une_recherche_binaire_exige_une_sequence_triee() -> None:
    """Sur `3 5 9 1 4 18 5`, la recherche binaire n'a pas de sens."""
    with pytest.raises(ValueError, match="sequence non triee"):
        lower_bound([3, 5, 9, 1, 4, 18, 5], 5)
    with pytest.raises(ValueError, match="sequence non triee"):
        upper_bound([2, 1], 1)


def test_reverse_retourne_le_vecteur_trie_du_programme_4_2() -> None:
    """Page 68 : « Vect1 Trie Reverse : 18 9 5 5 4 3 1 »."""
    v1 = list(VECT1_TRIE)
    reverse(v1)
    assert v1 == [18, 9, 5, 5, 4, 3, 1]


def test_la_fusion_de_deux_sequences_triees() -> None:
    """Page 68 : L2 `2 3 4 5 7` et L1 `0 1 2 3 5 7 9 18`."""
    fusion = merge([2, 3, 4, 5, 7], [0, 1, 2, 3, 5, 7, 9, 18])
    assert fusion == [0, 1, 2, 2, 3, 3, 4, 5, 5, 7, 7, 9, 18]
    assert merge([], [1, 2]) == [1, 2]
    with pytest.raises(ValueError, match="sequence non triee"):
        merge([3, 7, 5, 4, 2], [0, 1])
    with pytest.raises(ValueError, match="sequence non triee"):
        merge([0, 1], [3, 7, 5, 4, 2])


def test_merge_vide_la_liste_fusionnee() -> None:
    """Page 68 : « L2 merge L1 », puis « L1 apres merge.... : » -- vide."""
    l1 = Sequence("list", (0, 1, 2, 3, 5, 7, 9, 18))
    l2 = Sequence("list", (3, 7, 5, 4, 2))
    l2.sort()
    l2.merge(l1)
    assert l2.elements == (0, 1, 2, 2, 3, 3, 4, 5, 5, 7, 7, 9, 18)
    assert l1.empty()
    l2.unique()
    assert l2.elements == (0, 1, 2, 3, 4, 5, 7, 9, 18)


def test_splice_deplace_une_liste_dans_une_autre() -> None:
    """Page 68 : « L3 splice L2 : 99 0 1 2 3 4 5 7 9 18 », et L2 est vide."""
    l2 = Sequence("list", (0, 1, 2, 3, 4, 5, 7, 9, 18))
    l3 = Sequence("list", (99,))
    l3.splice(l3.size(), l2)
    assert l3.elements == (99, 0, 1, 2, 3, 4, 5, 7, 9, 18)
    assert l2.empty()
    l4 = Sequence("list", (1, 4))
    l4.splice(1, Sequence("list", (2, 3)))
    assert l4.elements == (1, 2, 3, 4)


def test_reverse_est_aussi_une_fonction_membre_de_la_liste() -> None:
    """Note des seances sur la STL : « l.reverse () reverse order of elements »."""
    liste = Sequence("list", (1, 2, 3))
    liste.reverse()
    assert liste.elements == (3, 2, 1)


@pytest.mark.parametrize("geste", ["merge", "splice", "reverse"])
def test_le_vecteur_n_a_aucune_de_ces_trois_fonctions_membres(geste: str) -> None:
    """Le tableau des fonctions membres ne les donne qu'aux listes."""
    vecteur, autre = Sequence("vector", (1, 2)), Sequence("vector", (3,))
    arguments = {"merge": (autre,), "splice": (0, autre), "reverse": ()}[geste]
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        getattr(vecteur, geste)(*arguments)


def test_splice_refuse_une_position_hors_de_la_liste() -> None:
    """L'iterateur de destination va de `begin()` a `end()`."""
    liste = Sequence("list", (1, 2))
    with pytest.raises(IndexError, match="position hors du conteneur"):
        liste.splice(3, Sequence("list", (9,)))
