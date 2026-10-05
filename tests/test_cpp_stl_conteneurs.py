"""Programmation objet, seances 5 et 6 : les conteneurs de la STL.

Sources, dans `ProgrammationAlgorithmique/SLIDES/` :
`Slides_2023_2024_S5.pdf` (planches 39 a 84),
`Slides_2025_2026_S5_S6.pdf` (planches 21 a 76, 105 et 106) et
`Slides_Note_Fonctions_membres.pdf`.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit trois ecarts :

* le tableau « Le template vector » (planche 45 de 2023-2024, 26 de
  2025-2026, page 3 de la note) dit de `v.insert(iterator, T)` « insert new
  element **after** iterator » : l'element est insere **devant**, comme
  l'ecrit la planche 48 ;
* planche 72 de 2023-2024 (54 de 2025-2026) :
  `L2.erase(L2.begin(), L2.begin() + L2.size()/2)` ne compile pas -- un
  iterateur de `list` n'a pas d'operateur `+`. Seule la colonne `vector`
  est juste ; la liste demande `advance`, que la meme planche donne ;
* planche 33 de 2023-2024 : `adr = new int[...]` est libere par
  `delete adr` au lieu de `delete [] adr`.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.associatifs import Associatif
from polytech.algorithmique.domain.conteneurs import (
    ASSOCIATIFS,
    MEMBRES,
    SEQUENTIELS,
    exiger,
    possede,
)
from polytech.algorithmique.domain.sequences import Sequence
from polytech.algorithmique.domain.tableaux import TableauStatique

NOMS = ("Obelix", "Tintin", "Asterix", "Titeuf", "Titeuf", "Obelix", "Asterix")


def test_les_cinq_conteneurs_sequentiels_et_les_quatre_associatifs() -> None:
    """Planche 43 et page 5 de la note : les en-tetes des deux tableaux."""
    assert SEQUENTIELS == ("array", "vector", "deque", "forward_list", "list")
    assert ASSOCIATIFS == ("set", "multiset", "map", "multimap")


@pytest.mark.parametrize("membre", ["push_front", "pop_front", "sort", "remove"])
def test_ce_que_la_liste_a_et_que_le_vecteur_n_a_pas(membre: str) -> None:
    """Planches 71 et 82 a 84 : « n'est pas membre de std::vector »."""
    assert possede("list", membre)
    assert not possede("vector", membre)


@pytest.mark.parametrize("membre", ["operator[]", "at", "capacity", "data"])
def test_ce_que_le_vecteur_a_et_que_la_liste_n_a_pas(membre: str) -> None:
    """Planche 58 : « l'operateur [i] n'est pas disponible sur les objets list »."""
    assert possede("vector", membre)
    assert not possede("list", membre)


@pytest.mark.parametrize(
    ("membre", "conteneurs"),
    [
        ("front", SEQUENTIELS),
        ("back", ("array", "vector", "deque", "list")),
        ("push_back", ("vector", "deque", "list")),
        ("push_front", ("deque", "forward_list", "list")),
        ("unique", ("forward_list", "list")),
        ("splice", ("list",)),
        ("resize", ("vector", "deque", "forward_list", "list")),
        ("size", ("array", "vector", "deque", "list")),
        ("empty", SEQUENTIELS),
    ],
)
def test_les_lignes_du_tableau_des_fonctions_membres(
    membre: str, conteneurs: tuple[str, ...]
) -> None:
    """Planches 43 et 44 : chaque ligne nomme les conteneurs qui ont le membre."""
    assert tuple(c for c in SEQUENTIELS if possede(c, membre)) == conteneurs


def test_seul_le_map_a_un_acces_par_la_cle() -> None:
    """Page 5 de la note : `at` et `operator[]` ne figurent que sous `map`."""
    for membre in ("at", "operator[]"):
        assert [c for c in ASSOCIATIFS if possede(c, membre)] == ["map"]
    assert all(possede(c, "lower_bound") for c in ASSOCIATIFS)
    assert not possede("set", "push_back")


def test_un_conteneur_inconnu_est_refuse() -> None:
    """`possede` ne repond que pour un conteneur du tableau."""
    with pytest.raises(ValueError, match="conteneur inconnu"):
        possede("stack", "push")


def test_l_appel_d_un_membre_absent_ne_compile_pas() -> None:
    """Planche 82 : `V.sort()` -- sort n'est pas membre de std::vector."""
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        exiger("vector", "sort")
    exiger("list", "sort")


def test_les_quatre_constructions_de_la_planche_52() -> None:
    """`V0` vide, `V1(6)` de zeros, `V2(6, 2)` de deux, `V3(V2)` sa copie."""
    assert Sequence("vector").elements == ()
    assert Sequence.de_taille("vector", 6).elements == (0,) * 6
    deux = Sequence.de_taille("vector", 6, 2)
    assert deux.elements == (2,) * 6
    assert Sequence("vector", deux.elements).elements == deux.elements
    assert Sequence.de_taille("list", 5, 77).elements == (77,) * 5


def test_seuls_le_vecteur_et_la_liste_sont_modelises() -> None:
    """Le cours ne manipule que ces deux conteneurs sequentiels."""
    with pytest.raises(ValueError, match="ni vector ni list"):
        Sequence("deque")


def test_push_back_et_pop_back_de_la_planche_53() -> None:
    """`Tab(6)` rempli de 0 a 5, deux `pop_back` : taille 5, puis 4."""
    tab = Sequence("vector", range(6))
    tab.pop_back()
    assert (tab.size(), tab.elements) == (5, (0, 1, 2, 3, 4))
    tab.pop_back()
    assert (tab.size(), tab.elements) == (4, (0, 1, 2, 3))
    tab.push_back(9)
    assert tab.back() == 9


def test_resize_de_la_planche_39() -> None:
    """`vector<int> v(10); v.resize(20);` -- 10 puis 20, completes de zeros."""
    v = Sequence.de_taille("vector", 10, 7)
    v.resize(20)
    assert v.size() == 20
    assert v.elements[9:11] == (7, 0)
    v.resize(3)
    assert v.elements == (7, 7, 7)


def test_la_liste_de_la_planche_59() -> None:
    """33 ; 33 15 ; 33 15 44 ; 99 33 15 44 ; 99 33 15 ; 33 15."""
    liste = Sequence("list")
    releves = []
    for geste, valeur, lu in (
        ("push_back", 33, "front"),
        ("push_back", 15, "back"),
        ("push_back", 44, "back"),
        ("push_front", 99, "front"),
    ):
        getattr(liste, geste)(valeur)
        releves.append(getattr(liste, lu)())
    liste.pop_back()
    releves.append(liste.back())
    liste.pop_front()
    releves.append(liste.front())
    assert releves == [33, 15, 44, 99, 15, 33]
    assert liste.elements == (33, 15)


def test_le_vecteur_n_a_ni_push_front_ni_pop_front() -> None:
    """Planche 71 : `V.push_front(i)` et `V.pop_front()` sont barres."""
    vecteur = Sequence("vector", (1, 2))
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        vecteur.push_front(0)
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        vecteur.pop_front()
    assert vecteur.elements == (1, 2)


def test_la_liste_n_a_pas_d_acces_par_indice() -> None:
    """Planche 58 : pas d'operateur `[i]`, donc pas de `at`."""
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        Sequence("list", (1, 2)).at(0)


def test_at_verifie_l_indice_du_vecteur() -> None:
    """Planche 45 : `v.at(i)` -- « subscript access (safe) »."""
    vecteur = Sequence("vector", (5, 6, 7))
    assert vecteur.at(2) == 7
    for indice in (3, -1):
        with pytest.raises(IndexError, match="indice hors du vecteur"):
            vecteur.at(indice)


@pytest.mark.parametrize("geste", ["front", "back", "pop_back", "pop_front"])
def test_un_conteneur_vide_n_a_ni_tete_ni_queue(geste: str) -> None:
    """En C++ le comportement est indefini : ici, une erreur."""
    with pytest.raises(IndexError, match="conteneur vide"):
        getattr(Sequence("list"), geste)()


def test_insert_et_remove_de_la_planche_69() -> None:
    """Liste `a b d d e`, puis `a A d d e`, `a A B B B d d e`, `a A B B B e`."""
    c1 = Sequence("list", "abdde")
    c1.erase(1)
    c1.insert(1, "A")
    assert "".join(c1.elements) == "aAdde"
    c1.insert(2, "B", 3)
    assert "".join(c1.elements) == "aABBBdde"
    c1.remove("d")
    assert (c1.size(), "".join(c1.elements)) == (6, "aABBBe")


def test_insert_place_l_element_devant_la_position() -> None:
    """Ecart du tableau « insert new element after iterator » : c'est devant."""
    vecteur = Sequence("vector", "abdde")
    vecteur.insert(2, "B", 3)
    assert "".join(vecteur.elements) == "abBBBdde"
    vecteur.insert(vecteur.size(), "z")
    assert vecteur.back() == "z"
    with pytest.raises(IndexError, match="position hors du conteneur"):
        vecteur.insert(vecteur.size() + 1, "z")


def test_remove_et_unique_de_la_liste_des_planches_83_et_84() -> None:
    """`a A B B B d d e` : sans B puis sans d, `a A e` ; `unique`, `a A B d e`."""
    liste = Sequence("list", "aABBBdde")
    liste.unique()
    assert "".join(liste.elements) == "aABde"
    liste = Sequence("list", "aABBBdde")
    liste.remove("B")
    liste.remove("d")
    assert "".join(liste.elements) == "aAe"


@pytest.mark.parametrize("geste", ["unique", "sort"])
def test_le_vecteur_n_a_ni_unique_ni_sort(geste: str) -> None:
    """Planches 82 et 84 : il faut les algorithmes de `<algorithm>`."""
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        getattr(Sequence("vector", (2, 1)), geste)()


def test_le_vecteur_n_a_pas_de_remove() -> None:
    """Planche 83, et la derniere ligne de la planche 70."""
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        Sequence("vector", "abdde").remove("d")


def test_les_tris_de_liste_de_la_planche_80() -> None:
    """`L1.sort()` : 1 2 3 4 8 9 ; `L2.sort(cmp)` : 18 16 8 6 4 2."""
    donnees = (9, 4, 2, 8, 1, 3)
    l1 = Sequence("list", donnees)
    l2 = Sequence("list", (2 * a for a in donnees))
    l1.sort()
    l2.sort(decroissant=True)
    assert l1.elements == (1, 2, 3, 4, 8, 9)
    assert l2.elements == (18, 16, 8, 6, 4, 2)


def test_les_effacements_de_la_planche_72() -> None:
    """`bcdefghij` et `2 3 4 5 6` ; puis la premiere moitie : `fghij` et `4 5 6`."""
    l2, v2 = Sequence("list", "abcdefghij"), Sequence("vector", range(1, 7))
    assert l2.erase(0) == 0
    v2.erase(0)
    assert ("".join(l2.elements), v2.elements) == ("bcdefghij", (2, 3, 4, 5, 6))
    l2, v2 = Sequence("list", "abcdefghij"), Sequence("vector", range(1, 7))
    l2.erase(0, l2.size() // 2)
    v2.erase(0, v2.size() // 2)
    assert ("".join(l2.elements), v2.elements) == ("fghij", (4, 5, 6))


def test_erase_rend_la_position_de_l_element_suivant() -> None:
    """2025-2026, planche 63 : apres le dernier, `erase` rend `end()`."""
    vecteur = Sequence("vector", (7, 3, 4))
    assert vecteur.erase(1) == 1
    assert vecteur.erase(1) == vecteur.size() == 1
    with pytest.raises(IndexError, match="plage hors du conteneur"):
        vecteur.erase(1)
    with pytest.raises(IndexError, match="plage hors du conteneur"):
        vecteur.erase(1, 0)


def test_la_boucle_d_effacement_de_la_planche_87() -> None:
    """2025-2026 : supprimer tous les 9 -- tailles 7, 6, 6, 5, 4, 4, 4, 3."""
    vecteur = Sequence("vector", (9, 9, 2, 9, 9, 2, 2, 9))
    position, tailles = 0, []
    while position != vecteur.size():
        if vecteur.elements[position] == 9:
            position = vecteur.erase(position)
        else:
            position += 1
        tailles.append(vecteur.size())
    assert tailles == [7, 6, 6, 5, 4, 4, 4, 3]
    assert vecteur.elements == (2, 2, 2)


def test_la_boucle_fautive_de_la_planche_62_saute_un_element_sur_deux() -> None:
    """`for (...; it++) { it = V.erase(it); }` sur 7 3 4 8 4 2 laisse 3 8 2."""
    vecteur = Sequence("vector", (7, 3, 4, 8, 4, 2))
    position = 0
    while position != vecteur.size():
        position = vecteur.erase(position) + 1
    assert vecteur.elements == (3, 8, 2)


def test_clear_et_empty() -> None:
    """2025-2026, planche 102 : « Tout Effacer »."""
    vecteur = Sequence("vector", "abbfef")
    assert not vecteur.empty()
    vecteur.clear()
    assert vecteur.empty()
    assert vecteur.size() == 0


def test_la_liste_videe_par_pop_back_de_la_planche_66() -> None:
    """`while (!L1.empty()) L1.pop_back();` : tailles 4, 3, 2, 1, 0."""
    liste = Sequence("list", (1, 2, 2, 3, 4))
    tailles = []
    while not liste.empty():
        liste.pop_back()
        tailles.append(liste.size())
    assert tailles == [4, 3, 2, 1, 0]


def test_le_set_ne_garde_qu_un_exemplaire() -> None:
    """Note, page 4 : trois `SET.insert("Asterix")`, un seul element."""
    ensemble, multiple = Associatif("set"), Associatif("multiset")
    for _ in range(3):
        ensemble.insert("Asterix")
        multiple.insert("Asterix")
    assert ensemble.size() == 1
    assert multiple.size() == 3
    assert (ensemble.count("Asterix"), multiple.count("Asterix")) == (1, 3)


def test_le_set_range_ses_cles_en_ordre_croissant() -> None:
    """2025-2026, planche 105 : Asterix, Obelix, Tintin, Titeuf."""
    ensemble, multiple = Associatif("set"), Associatif("multiset")
    for nom in NOMS:
        ensemble.insert(nom)
        multiple.insert(nom)
    assert ensemble.cles() == ("Asterix", "Obelix", "Tintin", "Titeuf")
    assert multiple.cles() == tuple(sorted(NOMS))


def test_le_map_de_la_planche_106_est_range_par_cle() -> None:
    """Insere dans l'ordre 1, 2, 5, 7, 6, 4, 3 ; parcouru de 1 a 7."""
    table = Associatif("map")
    for cle, nom in zip((1, 2, 5, 7, 6, 4, 3), NOMS, strict=True):
        table.insert(cle, nom)
    assert table.cles() == (1, 2, 3, 4, 5, 6, 7)
    assert [table.at(c) for c in table.cles()] == [
        "Obelix",
        "Tintin",
        "Asterix",
        "Obelix",
        "Asterix",
        "Titeuf",
        "Titeuf",
    ]


def test_le_map_refuse_une_cle_deja_presente_et_le_multimap_l_accepte() -> None:
    """Note, page 4 : « multimap : un map acceptant des duplications de cles »."""
    table, multiple = Associatif("map"), Associatif("multimap")
    assert table.insert(1, "Obelix")
    assert not table.insert(1, "Tintin")
    assert table.at(1) == "Obelix"
    assert multiple.insert(1, "Obelix")
    assert multiple.insert(1, "Tintin")
    assert multiple.count(1) == 2


def test_at_n_existe_que_pour_le_map() -> None:
    """Page 5 de la note : la colonne `map`, et elle seule."""
    multiple = Associatif("multimap")
    multiple.insert(1, "Obelix")
    with pytest.raises(AttributeError, match="n'est pas membre de std::"):
        multiple.at(1)
    with pytest.raises(KeyError, match="cle absente du map"):
        Associatif("map").at(1)


def test_les_bornes_d_une_cle() -> None:
    """Note, page 5 : `lower_bound`, premiere cle >= ; `upper_bound`, premiere >."""
    multiple = Associatif("multiset")
    for cle in (1, 3, 3, 3, 7):
        multiple.insert(cle)
    assert (multiple.lower_bound(3), multiple.upper_bound(3)) == (1, 4)
    assert (multiple.lower_bound(4), multiple.upper_bound(4)) == (4, 4)
    assert multiple.lower_bound(9) == multiple.size()


def test_un_conteneur_associatif_inconnu_est_refuse() -> None:
    """Quatre genres, pas un de plus."""
    with pytest.raises(ValueError, match="conteneur associatif inconnu"):
        Associatif("vector")


def test_effacer_une_plage_vide_ne_change_rien() -> None:
    """`erase(it, it)` est permis en C++ : rien n'est efface, `it` est rendu."""
    vecteur = Sequence("vector", (7, 3, 4))
    assert vecteur.erase(1, 1) == 1
    assert vecteur.erase(3, 3) == 3
    assert vecteur.elements == (7, 3, 4)


def test_unique_de_liste_ne_retire_que_les_voisins_egaux() -> None:
    """Planche 96 : `9 9 2 9 2 2 2 9 8` devient `9 2 9 2 9 8`, de taille 6."""
    liste = Sequence("list", (9, 9, 2, 9, 2, 2, 2, 9, 8))
    liste.unique()
    assert liste.elements == (9, 2, 9, 2, 9, 8)


def test_resize_complete_avec_la_valeur_donnee() -> None:
    """Planche 45 : `v.resize(unsigned, T)` -- « padding with value »."""
    vecteur = Sequence("vector", (1, 2))
    vecteur.resize(4, 5)
    assert vecteur.elements == (1, 2, 5, 5)


def test_une_position_negative_n_est_pas_un_iterateur() -> None:
    """Avant `begin()` il n'y a rien : ni a effacer, ni ou inserer."""
    vecteur = Sequence("vector", (7, 3, 4))
    with pytest.raises(IndexError, match="plage hors du conteneur"):
        vecteur.erase(-1)
    with pytest.raises(IndexError, match="position hors du conteneur"):
        vecteur.insert(-1, 0)
    assert vecteur.elements == (7, 3, 4)


def test_la_table_ne_nomme_que_des_conteneurs_du_cours() -> None:
    """Chaque ligne de `MEMBRES` cite des conteneurs des deux tableaux."""
    connus = {*SEQUENTIELS, *ASSOCIATIFS}
    assert all(conteneurs <= connus for conteneurs in MEMBRES.values())
    assert MEMBRES["sort"] == {"forward_list", "list"}
    assert MEMBRES["capacity"] == {"vector"}
    assert MEMBRES["swap"] == connus


def test_le_tableau_statique_est_borne_la_ou_le_vecteur_grandit() -> None:
    """Lien avec `tableaux` : « Nmax » reserve des cases, `vector` en ajoute."""
    statique = TableauStatique(3, [1, 2, 3])
    vecteur, liste = Sequence("vector", (1, 2, 3)), Sequence("list", (1, 2, 3))
    with pytest.raises(OverflowError, match="tableau plein"):
        statique.ajouter_a_la_fin(4)
    vecteur.push_back(4)
    liste.push_front(0)
    assert (statique.taille, vecteur.size(), liste.size()) == (3, 4, 4)
