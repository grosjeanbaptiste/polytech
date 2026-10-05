"""Syllabus « Methodologie orientee objet » de 2018 : ses programmes.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Programmation_orientee_objet_2018.pdf`. Chaque programme dont le
syllabus imprime la « Solution » est repris ici, sur les symboles nes des
planches des seances 4 a 6.

Chaque affichage a ete confirme en compilant le programme (clang++ 21,
C++17). La lecture et la compilation etablissent six ecarts :

* page 18, programme 1.6 : `static i=1;` n'a pas de type ;
* page 24, programme 5.9 : le code affiche « In Constructeur Param », la
  solution imprime « Dans Constructeur Param » ; et la surface du
  rectangle 2, jamais initialise, est donnee pour 0 -- elle est
  indeterminee ;
* page 22 : le destructeur est ecrit `~CRec()::CRec()` pour
  `CRec::~CRec()` ;
* page 27, programme 1.11 : la fin de chaine est posee en `Nom[i]`, hors de
  la boucle `for (int i ...)` qui declare `i` ;
* page 35, exercice 1.7 : « un operateur - afin d'additionner deux
  vecteurs » -- la solution imprimee soustrait ;
* pages 53 et 55 : `Pixel::Pixel (int abs=0, int ord=0, int cl=9)` donne
  les valeurs par defaut dans la definition hors de la classe -- clang le
  refuse, comme a la planche 22 de la seance 5.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.algorithmes_stl import (
    copy,
    count_if,
    max_element,
    min_element,
    replace,
    reverse,
    search,
    sort,
)
from polytech.algorithmique.domain.constructeurs import (
    Constructeur,
    est_constructible,
    tableau_constructible,
)
from polytech.algorithmique.domain.duree_de_vie import (
    CONSTRUCTION,
    COPIE,
    DESTRUCTION,
    Execution,
)
from polytech.algorithmique.domain.encapsulation import PRIVE, PUBLIC, Classe
from polytech.algorithmique.domain.heritage import Declaration, Hierarchie
from polytech.algorithmique.domain.liste_chainee import ListeChainee
from polytech.algorithmique.domain.portee import Portees
from polytech.algorithmique.domain.recherche_triee import upper_bound
from polytech.algorithmique.domain.sequences import Sequence
from polytech.algorithmique.domain.surcharge import arguments_completes
from polytech.algorithmique.domain.tableaux_de_structures import un_sur_deux
from polytech.algorithmique.domain.types_cpp import caractere_decale, tronquer

C, K, D = CONSTRUCTION, COPIE, DESTRUCTION
SAISIE = (1, 2, 55, 33, 4, 9, 7)
POULES = [("Poule4", 1000), ("Poule3", 1200), ("Poule2", 180), ("Poule1", 200)]


def _lus(conteneur: Sequence) -> tuple[object, ...]:
    """Une lecture fraiche : mypy garde sinon le type etroit de l'assertion d'avant."""
    return conteneur.elements


def test_les_identifiants_et_les_codes_du_programme_1_6() -> None:
    """Page 18 : trois etudiants -- 1 et A, 2 et B, 3 et C."""
    portees = Portees()
    crees = []
    for _ in range(3):
        portees.ouvrir()
        portees.declarer_statique("Etudiant", "i", 1)
        portees.declarer_statique("Etudiant", "K", "A")
        crees.append((portees.valeur("i"), portees.valeur("K")))
        portees.affecter("i", portees.valeur("i") + 1)
        portees.affecter("K", caractere_decale(portees.valeur("K"), 1))
        portees.fermer()
    assert crees == [(1, "A"), (2, "B"), (3, "C")]


def test_la_trace_du_programme_1_10() -> None:
    """Page 26 : la globale, b et c, trois tours de boucle, le reste a l'envers."""
    execution = Execution()
    execution.globale("a(1,1)")
    execution.ouvrir()
    execution.locale("b(10,10)")
    execution.locale("c(3,3)")
    for tour in (1, 2, 3):
        execution.ouvrir()
        execution.locale(f"b({tour},{2 * tour})")
        execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "a(1,1)"),
        (C, "b(10,10)"),
        (C, "c(3,3)"),
        (C, "b(1,2)"),
        (D, "b(1,2)"),
        (C, "b(2,4)"),
        (D, "b(2,4)"),
        (C, "b(3,6)"),
        (D, "b(3,6)"),
        (D, "c(3,3)"),
        (D, "b(10,10)"),
        (D, "a(1,1)"),
    ]


def test_les_declarations_fautives_de_la_page_23() -> None:
    """`CRec R2;`, `Tab T1, T2[10];`, `point Pt;` : aucun constructeur par defaut."""
    deux, un = (Constructeur(2),), (Constructeur(1),)
    assert est_constructible(deux, 2)
    assert not est_constructible(deux, 0)
    assert not est_constructible(un, 0)
    assert not tableau_constructible(un)
    assert tableau_constructible((Constructeur(0), Constructeur(2)))


def test_les_deux_copies_du_programme_1_11() -> None:
    """Page 28 : `Etudiant E2(E1);` et `Etudiant E3=E1;` appellent la copie."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("E1")
    execution.locale("E2", modele="E1")
    execution.locale("E3", modele="E1")
    assert execution.journal == [(C, "E1"), (K, "E2"), (K, "E3")]


def test_les_surfaces_du_programme_1_12() -> None:
    """Page 30 : 200 et 200 ; apres `Rc1 = CRec(5, 15)`, 75, 75 et 75."""
    rc1 = (10, 20)
    rc2 = rc1
    assert (rc1[0] * rc1[1], rc2[0] * rc2[1]) == (200, 200)
    rc1 = (5, 15)
    rc3, rc4 = rc1, rc1
    assert [a * b for a, b in (rc1, rc3, rc4)] == [75, 75, 75]
    assert rc2 == (10, 20)


def test_le_tableau_d_objets_du_programme_1_13() -> None:
    """Page 31 : `{ CRec(3,4), 9, n, 5*n+2 }` -- le cinquieme prend les defauts."""
    n = 2
    initialisations = ((3, 4), (9,), (n,), (5 * n + 2,), ())
    objets = [arguments_completes((0, 0), i) for i in initialisations]
    assert objets == [(3, 4), (9, 0), (2, 0), (12, 0), (0, 0)]
    execution = Execution()
    execution.ouvrir()
    execution.tableau("Tab", 5)
    execution.terminer()
    detruits = [nom for evenement, nom in execution.journal if evenement == D]
    assert detruits == [f"Tab[{rang}]" for rang in (4, 3, 2, 1, 0)]


def test_les_quatre_objets_de_l_exercice_1_5() -> None:
    """Page 34 : `CLS(int a = 1, float b = 2.2, char c = 'F')`."""
    defauts = (1, 2.2, "F")
    appels = ((), (20,), (30, 31.23), (40, 43.21, "D"))
    assert [arguments_completes(defauts, a) for a in appels] == [
        (1, 2.2, "F"),
        (20, 2.2, "F"),
        (30, 31.23, "F"),
        (40, 43.21, "D"),
    ]


def test_la_methode_amie_de_la_page_43() -> None:
    """`friend void Fenetre::afficher(Tableau t);` : cette methode, et elle seule."""
    tableau = Classe(
        "Tableau", {"tab": PRIVE, "nbr": PRIVE}, amies=frozenset({"Fenetre::afficher"})
    )
    assert tableau.accede("tab", "Fenetre::afficher")
    assert not tableau.accede("tab", "Fenetre::fermer")
    assert not tableau.accede("tab", "Fenetre")


def test_une_classe_ne_lit_pas_le_prive_d_une_autre() -> None:
    """Page 16 : dans `A::fonction_a`, `a0.a = 1` passe, `b0.b = 3` non."""
    a = Classe("A", {"a": PRIVE, "fonction_a": PUBLIC})
    b = Classe("B", {"b": PRIVE, "fonction_b": PUBLIC})
    assert a.accede("a", "A")
    assert not b.accede("b", "A")


def test_le_pixel_ne_lit_pas_les_coordonnees_privees_du_point() -> None:
    """Page 49 : `AfficheCol` ne peut pas ecrire `x` et `y` ; elle appelle `affiche`."""
    hierarchie = Hierarchie(
        {
            "point": Declaration(membres={"x": PRIVE, "y": PRIVE, "affiche": PUBLIC}),
            "Pixel": Declaration(
                meres=("point",), membres={"couleur": PRIVE, "affiche": PUBLIC}
            ),
        }
    )
    assert not hierarchie.accede("Pixel", "x", "Pixel")
    assert hierarchie.definit("Pixel", "affiche") == "Pixel"
    assert hierarchie.definit("point", "affiche") == "point"


def test_la_trace_du_programme_3_3() -> None:
    """Page 51 : `++++ A`, `++++ B`, puis `~~~~ B`, `~~~~ A`."""
    hierarchie = Hierarchie({"A": Declaration(), "B": Declaration(meres=("A",))})
    assert hierarchie.ordre_de_construction("B") == ("A", "B")
    assert hierarchie.ordre_de_destruction("B") == ("B", "A")


def test_les_arguments_transmis_au_point_par_le_pixel() -> None:
    """Pages 52 et 53 : sans liste d'initialisation, `point` recoit 0 et 0."""
    assert arguments_completes((0, 0), ()) == (0, 0)
    pixel = arguments_completes((0, 0, 9), (2,))
    assert (pixel, pixel[:2]) == ((2, 0, 9), (2, 0))
    dynamique = arguments_completes((0, 0, 9), (20, 30))
    assert (dynamique, dynamique[:2]) == ((20, 30, 9), (20, 30))


def test_la_solution_de_l_exercice_3_1() -> None:
    """Page 55 : quatre pixels -- (10,15,3), (2,3,9), (12,0,9), (12,25,9)."""
    appels = ((10, 15, 3), (2, 3), (12,), (12, 25))
    pixels = [arguments_completes((0, 0, 9), a) for a in appels]
    assert pixels == [(10, 15, 3), (2, 3, 9), (12, 0, 9), (12, 25, 9)]
    assert [couleur for _, _, couleur in reversed(pixels)] == [9, 9, 9, 3]


def test_les_valeurs_de_l_exercice_3_2() -> None:
    """Page 56 : `ClsB b(10)` transmet `2*a` a `ClsA` -- x = 20, y = 10."""
    a = 10
    (x,) = arguments_completes((3,), (2 * a,))
    assert (x, a) == (20, 10)
    hierarchie = Hierarchie(
        {
            "ClsA": Declaration(membres={"x": "protected", "f": "protected"}),
            "ClsB": Declaration(meres=("ClsA",), membres={"y": PRIVE, "g": PUBLIC}),
        }
    )
    assert hierarchie.accede("ClsB", "x", "ClsB")
    assert not hierarchie.accede("ClsB", "f", "main")


def test_le_pointeur_polymorphe_de_la_page_54() -> None:
    """Sans `virtual`, `ptr->Affiche()` rend CLA deux fois ; avec, CLA puis CLB."""
    for virtuelles, attendu in ((frozenset(), "CLA"), (frozenset({"Affiche"}), "CLB")):
        hierarchie = Hierarchie(
            {
                "CLA": Declaration(membres={"Affiche": PUBLIC}, virtuelles=virtuelles),
                "CLB": Declaration(meres=("CLA",), membres={"Affiche": PUBLIC}),
            }
        )
        assert hierarchie.methode_executee("CLA", "CLA", "Affiche") == "CLA"
        assert hierarchie.methode_executee("CLA", "CLB", "Affiche") == attendu


def test_les_oeufs_du_poulailler() -> None:
    """Exercice 3.3 : un oeuf pese 0,1 poule -- 100, 120, 18 et 20 grammes."""
    oeufs = [tronquer(0.1 * poids) for _, poids in POULES]
    assert oeufs == [100, 120, 18, 20]


def test_une_poule_sur_deux_pond_quand_le_coq_dit_coco() -> None:
    """Exercice 3.3, choix 4 : « pese : 100(gr) » puis « pese : 18(gr) »."""
    pondeuses = un_sur_deux(POULES)
    assert [tronquer(0.1 * poids) for _, poids in pondeuses] == [100, 18]


def test_la_hierarchie_du_poulailler() -> None:
    """Exercice 3.3 : Volaille, d'ou Poule, Coq et Oeuf ; `affiche` redefinie."""
    affiche = {"saisie": PUBLIC, "affiche": PUBLIC}
    hierarchie = Hierarchie(
        {
            "Volaille": Declaration(membres={"Nom": PRIVE, **affiche}),
            "Poule": Declaration(meres=("Volaille",), membres=affiche),
            "Coq": Declaration(meres=("Volaille",), membres={"cocorico": PUBLIC}),
            "Oeuf": Declaration(meres=("Volaille",), membres=affiche),
        }
    )
    assert hierarchie.definit("Poule", "affiche") == "Poule"
    assert hierarchie.definit("Coq", "affiche") == "Volaille"
    assert hierarchie.ordre_de_construction("Oeuf") == ("Volaille", "Oeuf")


def test_l_execution_du_programme_4_1() -> None:
    """Pages 64 : sept entiers saisis, puis chaque ligne des « Resultats »."""
    tab, liste1, liste2 = Sequence("vector"), Sequence("list"), Sequence("list")
    for entier in SAISIE:
        liste1.push_back(entier)
        liste2.push_front(entier)
        tab.push_back(entier)
    assert liste2.elements == (7, 9, 4, 33, 55, 2, 1) == liste1.elements[::-1]
    assert (tab.size(), tab.front(), tab.back(), tab.at(tab.size() // 2)) == (
        7,
        1,
        7,
        33,
    )
    liste1.pop_back()
    tab.pop_back()
    liste1.pop_front()
    assert liste1.elements == (2, 55, 33, 4, 9)
    tab.erase(2)
    assert tab.elements == (1, 2, 33, 4, 9)
    tab.insert(tab.size() - 1, 99)
    assert _lus(tab) == (1, 2, 33, 4, 99, 9)
    tab.erase(1, tab.size() - 1)
    assert _lus(tab) == (1, 9)
    tab.clear()
    assert tab.empty()


def test_les_recherches_et_les_tris_du_programme_4_2() -> None:
    """Page 68 : `{9,1}` en position 2 puis 0 ; minimum 1, maximum 18."""
    t1, t2 = [3, 5, 9, 1, 4, 18, 5], [9, 1, 7, 2, 5]
    assert (search(t1, (9, 1)), search(t2, (9, 1))) == (2, 0)
    sort(t1)
    sort(t2)
    assert (t1, t2) == ([1, 3, 4, 5, 5, 9, 18], [1, 2, 5, 7, 9])
    assert (t1[min_element(t1)], t1[max_element(t1)]) == (1, 18)


def test_les_copies_du_programme_4_2() -> None:
    """Page 68 : « Vect +V1+V2 », puis « Vect Trie »."""
    v1, v2, vv = [1, 3, 4, 5, 5, 9, 18], [1, 2, 5, 7, 9], [0] * 15
    reverse(v1)
    replace(v1, 5, 4)
    assert v1 == [18, 9, 4, 4, 4, 3, 1]
    assert copy(v1, vv) == 7
    assert copy(v2, vv, 7) == 12
    assert vv == [18, 9, 4, 4, 4, 3, 1, 1, 2, 5, 7, 9, 0, 0, 0]
    sort(vv)
    assert vv == [0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 5, 7, 9, 9, 18]


def test_les_listes_du_programme_4_2() -> None:
    """Page 68 : « L1 Unique », « L1 - 4 », « L2 », « L2 Trie »."""
    l1 = Sequence("list", (0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 5, 7, 9, 9, 18))
    l1.unique()
    assert l1.elements == (0, 1, 2, 3, 4, 5, 7, 9, 18)
    l1.remove(4)
    assert _lus(l1) == (0, 1, 2, 3, 5, 7, 9, 18)
    l2 = Sequence("list")
    for geste, valeur in (
        ("push_front", 5),
        ("push_front", 7),
        ("push_back", 4),
        ("push_back", 2),
        ("push_front", 3),
    ):
        getattr(l2, geste)(valeur)
    assert l2.elements == (3, 7, 5, 4, 2)
    l2.sort()
    assert l2.elements == (2, 3, 4, 5, 7)


def test_la_liste_ordonnee_de_l_exercice_4_3() -> None:
    """Sans `sort` : Toto, Mimi, Picsou, Aladin, Jasmine ranges a l'insertion."""
    noms = Sequence("list")
    for nom in ("Toto", "Mimi", "Picsou", "Aladin", "Jasmine"):
        noms.insert(upper_bound(noms.elements, nom), nom)
    assert noms.elements == ("Aladin", "Jasmine", "Mimi", "Picsou", "Toto")


def test_la_deuxieme_moitie_effacee_element_par_element() -> None:
    """Exercice 4.5 : dix reels de 0.0 a 0.9 ; il en reste cinq, de 0.0 a 0.4."""
    v2 = Sequence("vector", (rang / 10 for rang in range(10)))
    while v2.size() > 5:
        v2.erase(v2.size() - 1)
    assert v2.elements == (0.0, 0.1, 0.2, 0.3, 0.4)


def test_le_nombre_de_filles_de_l_exercice_4_9() -> None:
    """`count_if` sur le sexe de chaque etudiante de la liste."""
    etudiants = (("Ana", "F"), ("Bob", "M"), ("Cleo", "F"), ("Dan", "M"), ("Eve", "F"))
    assert count_if(etudiants, lambda etudiant: etudiant[1] == "F") == 3


def _passer_d_une_liste_a_l_autre(depart: Sequence, arrivee: Sequence) -> list[str]:
    etats = []
    while not depart.empty():
        arrivee.push_back(depart.back())
        depart.pop_back()
        etats.append(" ".join(depart.elements) + " | " + " ".join(arrivee.elements))
    return etats


def test_le_parcours_vita_de_l_exercice_4_11() -> None:
    """Page 72 : D, C, B, A inseres en tete ; puis un par un vers la liste 2."""
    liste1, liste2, liste3 = (Sequence("list") for _ in range(3))
    for participant in ("D", "C", "B", "A"):
        liste1.push_front(participant)
    assert liste1.elements == ("A", "B", "C", "D")
    etats = _passer_d_une_liste_a_l_autre(liste1, liste2)
    assert etats[0] == "A B C | D"
    assert (_lus(liste1), _lus(liste2)) == ((), ("D", "C", "B", "A"))
    _passer_d_une_liste_a_l_autre(liste2, liste3)
    assert liste3.elements == ("A", "B", "C", "D")
    with pytest.raises(IndexError, match="conteneur vide"):
        liste1.pop_back()


# --- Syllabus « Informatique II et Programmation Objet » de 2008-2009 --------
#
# `Syllabus_Programmation_objet_C_2009.pdf`, 73 pages. Ses chapitres 3 a 6
# sont ceux de 2018, renumerotes ; il a en propre les chapitres 1 (structures
# et fonctions membres) et 2 (listes simplement chainees, le programme que le
# syllabus d'introduction de 2015 reprend a son chapitre 9).


def test_les_points_du_programme_1_2_de_2009() -> None:
    """Page 6 : « Je suis en 3 7 », « 10 20 », puis 0 4, 1 5 et 2 6."""
    tab_point = [(i, i + 4) for i in range(3)]
    assert [(3, 7), (10, 20), *tab_point] == [(3, 7), (10, 20), (0, 4), (1, 5), (2, 6)]


def test_une_structure_laisse_ecrire_ses_champs_et_une_classe_non() -> None:
    """Pages 6 et 18 : `b.x = 10` passe sur la structure ; sur la classe, C2248."""
    assert Classe.de("struct", "point", ["x", "y"]).accede("x", "main")
    assert not Classe.de("class", "point", ["x", "y"]).accede("x", "main")


def test_la_liste_chainee_du_chapitre_2_de_2009() -> None:
    """Page 10 : nom3, second, premier ; « Nouveau » apres le debut ; le debut ote."""
    liste = ListeChainee()
    for nom in ("premier", "second", "nom3"):
        liste.inserer_en_tete(nom)
    assert liste.parcours() == ("nom3", "second", "premier")
    liste.inserer_apres_la_tete("Nouveau")
    assert liste.parcours() == ("nom3", "Nouveau", "second", "premier")
    assert liste.supprimer_la_tete() == "nom3"
    assert liste.parcours() == ("Nouveau", "second", "premier")
    assert liste.liberer() == 3


def test_le_parcours_vita_en_liste_chainee_de_l_exercice_2_2() -> None:
    """Page 11 : saisis D, C, B, A en tete, la liste se lit A, B, C, D."""
    liste = ListeChainee()
    for participant in ("D", "C", "B", "A"):
        liste.inserer_en_tete(participant)
    assert liste.parcours() == ("A", "B", "C", "D")


def test_la_trace_de_l_exercice_3_4_de_2009() -> None:
    """Page 34 : « Creation Objet », « Salut », « Destruction Objet »."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("x")
    assert execution.journal == [(C, "x")]
    execution.terminer()
    assert execution.journal == [(C, "x"), (D, "x")]
