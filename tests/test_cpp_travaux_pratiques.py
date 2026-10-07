"""Programmation : les enonces des travaux pratiques 1, 2 et 4.

Sources : `ProgrammationAlgorithmique/TP/TP_01_a_06_2019_2020.pdf` (douze
pages, TP 1 a 6) et `TP_04_a_06_2016_2017.pdf` (quatre pages, TP 4 a 6, mot
pour mot ceux de 2019). Le TP 3 est porte par
`test_cpp_fichiers_et_structures`, les TP 5 et 6 par les tests de la STL
et de l'heritage.

Les enonces donnent peu de nombres ; ceux qu'ils donnent sont ici :

* page 5, l'echange de `i = 5` et `j = 8`, par valeur puis par reference ;
* page 6, « supprimer le milieu et ajouter son double au debut » :
  `1 2 3 4 5`, puis `6 1 2 4 5`, puis `4 6 1 4 5`.

Tous les tests portent sur des symboles existants : les TP composent les
gestes du cours, ils n'en ajoutent pas.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.algorithmes_stl import copy
from polytech.algorithmique.domain.enregistrements import trier_selon
from polytech.algorithmique.domain.memoire import Memoire
from polytech.algorithmique.domain.recherches import recherche_lineaire
from polytech.algorithmique.domain.tableaux import TableauStatique
from polytech.algorithmique.domain.tableaux_de_structures import extremes

NMAX = 10


def _echanger(memoire: Memoire, a: str, b: str) -> None:
    tmp = memoire.valeur(b)
    memoire.affecter(b, memoire.valeur(a))
    memoire.affecter(a, tmp)


def test_l_echange_par_valeur_du_tp_2_ne_change_rien() -> None:
    """Page 5 : « i = 5 j = 8 » avant, « i = 5 j = 8 » apres."""
    memoire = Memoire()
    memoire.declarer("i", 5)
    memoire.declarer("j", 8)
    memoire.declarer("a", memoire.valeur("i"))
    memoire.declarer("b", memoire.valeur("j"))
    _echanger(memoire, "a", "b")
    assert (memoire.valeur("a"), memoire.valeur("b")) == (8, 5)
    assert (memoire.valeur("i"), memoire.valeur("j")) == (5, 8)


def test_l_echange_par_reference_du_tp_2_echange_vraiment() -> None:
    """Page 5 : « i = 5 j = 8 » avant, « i = 8 j = 5 » apres."""
    memoire = Memoire()
    memoire.declarer("i", 5)
    memoire.declarer("j", 8)
    memoire.referencer("a", "i")
    memoire.referencer("b", "j")
    _echanger(memoire, "a", "b")
    assert (memoire.valeur("i"), memoire.valeur("j")) == (8, 5)


def _supprimer_le_milieu_et_doubler(tableau: TableauStatique) -> None:
    milieu = tableau.supprimer(tableau.taille // 2)
    tableau.inserer(0, 2 * milieu)


def test_le_point_7_du_tp_2() -> None:
    """Page 6 : `1 2 3 4 5`, puis `6 1 2 4 5`, puis `4 6 1 4 5`."""
    tab2 = TableauStatique(NMAX, [1, 2, 3, 4, 5])
    _supprimer_le_milieu_et_doubler(tab2)
    assert tab2.elements == [6, 1, 2, 4, 5]
    _supprimer_le_milieu_et_doubler(tab2)
    assert tab2.elements == [4, 6, 1, 4, 5]


def test_le_point_6_du_tp_2() -> None:
    """Transvaser TAB1 dans TAB2, puis son minimum, son maximum, sa moyenne."""
    tab1 = [4.0, 1.5, 9.0, 2.5]
    tab2 = [0.0] * len(tab1)
    assert copy(tab1, tab2) == len(tab1)
    plus, moins = extremes(tab2, lambda valeur: valeur)
    assert (moins, plus, sum(tab2) / len(tab2)) == (1.5, 9.0, 4.25)


def test_les_points_2_et_3_du_tp_1() -> None:
    """Supprimer le debut et la fin, puis ajouter un element au debut."""
    tab1 = TableauStatique(NMAX, [1.0, 2.0, 3.0, 4.0, 5.0])
    tab1.supprimer(0)
    tab1.supprimer(tab1.taille - 1)
    assert tab1.elements == [2.0, 3.0, 4.0]
    tab1.inserer(0, 9.0)
    assert tab1.elements == [9.0, 2.0, 3.0, 4.0]


def test_le_tableau_du_tp_1_ne_depasse_pas_nmax() -> None:
    """Page 3 : `N` dans l'intervalle ferme `[1, Nmax]`, `Nmax` valant 10."""
    tab1 = TableauStatique(NMAX, list(range(NMAX)))
    with pytest.raises(OverflowError, match="tableau plein"):
        tab1.inserer(0, 9)
    with pytest.raises(OverflowError, match="plus d'elements que de places"):
        TableauStatique(NMAX, list(range(NMAX + 1)))


def test_le_menu_du_tp_4() -> None:
    """Ajouter au debut, supprimer le debut, chercher selon l'Id, trier selon le nom."""
    etudiants = TableauStatique(NMAX, [(2, "Riri"), (3, "Fifi")])
    etudiants.inserer(0, (1, "Zoe"))
    assert etudiants.elements == [(1, "Zoe"), (2, "Riri"), (3, "Fifi")]
    identifiants = [identifiant for identifiant, _ in etudiants.elements]
    assert recherche_lineaire(identifiants, 3) == 2
    assert recherche_lineaire(identifiants, 7) == etudiants.taille
    tries = trier_selon(etudiants.elements, lambda etudiant: etudiant[1])
    assert [nom for _, nom in tries] == ["Fifi", "Riri", "Zoe"]
    assert etudiants.supprimer(0) == (1, "Zoe")
