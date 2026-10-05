"""Programmation objet, seance 4 : classes, operateurs, patrons, amies.

Sources, dans `ProgrammationAlgorithmique/SLIDES/` :
`Slides_2023_2024_S4_Surcharge_des_operateurs.pdf`,
`Slides_2024_2025_S4_Methodologie_orientee_objet.pdf` et
`Slides_2025_2026_S4.pdf`. Le numero de planche est donne avec son annee.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit deux ecarts :

* 2025-2026, planche 48 (et 64 de 2024-2025) : sous `using namespace std`,
  l'appel `min(2, 9)` est ambigu entre le patron de la planche et
  `std::min` ; le programme ne compile que sans la directive ;
* la meme planche annote `min(2.3, 5.2)` de `float min(float, float)` :
  `2.3` est un `double`, et c'est `double` que le patron deduit.
"""

from __future__ import annotations

import math

import pytest

from polytech.algorithmique.domain.encapsulation import (
    PRIVE,
    PROTEGE,
    PUBLIC,
    Classe,
    ligne_uml,
    visibilite_par_defaut,
)
from polytech.algorithmique.domain.flux_de_sortie import ecrire_un_reel
from polytech.algorithmique.domain.identificateurs import type_deduit
from polytech.algorithmique.domain.operateurs import Point
from polytech.algorithmique.domain.patrons import (
    minimum,
    somme_du_patron,
    types_deduits,
)
from polytech.algorithmique.domain.recherches import recherche_lineaire
from polytech.algorithmique.domain.types_cpp import diviser

MEMBRES_DU_POINT = ("x", "y", "z", "initialise", "affiche")
POINT_CLASSE = Classe(
    "point",
    {"x": PRIVE, "y": PRIVE, "z": PUBLIC, "initialise": PUBLIC, "affiche": PUBLIC},
)
POINT_STRUCTURE = Classe.de("struct", "point", MEMBRES_DU_POINT)
OBJET = Classe(
    "Objet", {"tab": PRIVE, "nbr": PRIVE}, amies=frozenset({"afficher", "OB"})
)
QUI = Classe("Qui", {"LeNom": PRIVE, "sonnom": PUBLIC}, amies=frozenset({"Copain"}))


def test_une_structure_est_publique_et_une_classe_privee() -> None:
    """2025-2026, planche 39 : sans mot cle, `class` est privee et `struct` publique."""
    assert visibilite_par_defaut("struct") == PUBLIC
    assert visibilite_par_defaut("class") == PRIVE
    with pytest.raises(ValueError, match="ni struct ni class"):
        visibilite_par_defaut("union")


def test_tout_membre_d_une_structure_se_lit_de_main() -> None:
    """Planche 7 : `c.x = 10; c.y = 30;` dans `main`, puis « Je suis en 10 30 »."""
    assert all(POINT_STRUCTURE.accede(membre, "main") for membre in MEMBRES_DU_POINT)
    assert Classe.de("class", "point", ("x", "y")).membres == {"x": PRIVE, "y": PRIVE}


def test_le_membre_prive_de_la_planche_40_est_refuse_a_main() -> None:
    """`a.z = 9` passe, `c.x = 10` rend l'erreur C2248."""
    assert POINT_CLASSE.accede("z", "main")
    assert not POINT_CLASSE.accede("x", "main")
    assert not POINT_CLASSE.accede("y", "main")


def test_une_fonction_membre_lit_les_membres_prives_de_sa_classe() -> None:
    """Planche 43 : `initialise` et `getX` touchent `x`, `main` passe par eux."""
    assert POINT_CLASSE.accede("x", "point")
    assert POINT_CLASSE.accede("initialise", "main")


def test_un_membre_protege_s_ouvre_aux_classes_derivees() -> None:
    """2024-2025, planche 50 : `protected`, entre `private` et `public`."""
    classe = Classe("X", {"secret": PRIVE, "herite": PROTEGE})
    assert classe.accede("herite", "Fille", derivee=True)
    assert not classe.accede("herite", "main")
    assert not classe.accede("secret", "Fille", derivee=True)


def test_un_membre_inconnu_n_a_pas_de_visibilite() -> None:
    """Ni accessible ni refuse : il n'existe pas."""
    with pytest.raises(AttributeError, match="n'est pas membre de point"):
        POINT_CLASSE.accede("w", "main")


def test_une_visibilite_inconnue_est_refusee_a_la_declaration() -> None:
    """Trois mots cles, pas un de plus."""
    with pytest.raises(ValueError, match="visibilite inconnue"):
        Classe("X", {"x": "secret"})


def test_la_fonction_amie_de_la_planche_61_lit_le_tableau_prive() -> None:
    """`friend void afficher(Objet t);` : « 0 1 2 3 4 »."""
    assert OBJET.accede("tab", "afficher")
    assert OBJET.accede("nbr", "afficher")


def test_la_fonction_non_amie_de_la_planche_63_est_refusee() -> None:
    """`afficher2` n'est pas declaree amie : erreur C2248 sur `Objet::tab`."""
    assert not OBJET.accede("tab", "afficher2")
    assert not OBJET.accede("nbr", "afficher2")


def test_la_classe_amie_de_la_planche_62_lit_le_membre_prive() -> None:
    """`friend class OB;` : `Y.fct(C)` affiche 9."""
    assert OBJET.accede("nbr", "OB")


def test_l_amitie_de_la_planche_75_n_est_pas_reciproque() -> None:
    """2024-2025 : `Copain` lit `Qui::LeNom`, mais `Qui` ne lit pas `Copain::nom`."""
    copain = Classe("Copain", {"nom": PRIVE, "monami": PUBLIC})
    assert QUI.accede("LeNom", "Copain")
    assert not QUI.accede("LeNom", "main")
    assert not copain.accede("nom", "Qui")


def test_la_notation_uml_de_la_planche_41() -> None:
    """2024-2025 : `-Marque : string`, `+Avancer() : void`."""
    assert ligne_uml("Marque", "string", PRIVE) == "-Marque : string"
    assert ligne_uml("Avancer()", "void", PUBLIC) == "+Avancer() : void"
    assert ligne_uml("Id", "int", PROTEGE) == "#Id : int"


def test_les_rectangles_de_la_planche_61_passent_par_les_accesseurs() -> None:
    """2024-2025 : `Rc2.initialise(R_Long() / 2, R_Larg() * 2)`, deux fois 720."""
    rectangle = Classe("CRec", {"Long": PRIVE, "Larg": PRIVE, "R_Long": PUBLIC})
    assert not rectangle.accede("Long", "main")
    assert rectangle.accede("R_Long", "main")
    longueur, largeur = 30, 24
    seconde = (diviser(longueur, 2), largeur * 2)
    attendue: tuple[float, int] = (15, 48)
    assert seconde == attendue
    assert longueur * largeur == seconde[0] * seconde[1] == 720


def test_l_egalite_de_deux_points_de_la_planche_25() -> None:
    """2025-2026 : `a(3, 7)`, `b(4, 4)` -- « Non EGALITE »."""
    assert Point(3, 7) != Point(4, 4)
    assert Point(3, 7) == Point(3, 7)
    assert Point(3, 7) != Point(3, 4)


def test_la_somme_et_la_difference_de_la_planche_27() -> None:
    """`(a + b).affiche()` rend 7, 11 ; `(a - b).affiche()` rend -1, 3."""
    a, b = Point(3, 7), Point(4, 4)
    assert a + b == Point(7, 11)
    assert a - b == Point(-1, 3)


def test_la_somme_et_la_difference_de_la_planche_15_de_2023() -> None:
    """`A(3, 4)`, `B(4, 4)` : 7, 8 puis -1, 0."""
    a, b = Point(3, 4), Point(4, 4)
    assert (a + b, a - b) == (Point(7, 8), Point(-1, 0))


def test_la_recherche_de_la_planche_26_repose_sur_l_operateur() -> None:
    """Lien avec `recherches` : `if (Tab[i] == P) return (i + 1);`."""
    tableau = (Point(1, 1), Point(3, 7), Point(4, 4))
    assert recherche_lineaire(tableau, Point(3, 7)) + 1 == 2
    assert recherche_lineaire(tableau, Point(9, 9)) == len(tableau)


def test_le_milieu_et_la_distance_de_la_planche_54() -> None:
    """2024-2025 : `A(4, 3)`, `B(5, 8)` -- milieu (4.5, 5.5), distance 5.09902."""
    a, b = Point(4, 3), Point(5, 8)
    assert a.milieu(b) == Point(4.5, 5.5)
    assert ecrire_un_reel(a.distance(b)) == "5.09902"
    assert a.distance(b) == pytest.approx(math.sqrt(26))
    assert a.distance(a) == 0


def test_le_patron_min_de_la_planche_48() -> None:
    """`min(2, 9) = 2`, `min(2.3, 5.2) = 2.3`, `min('z', 'c') = c`."""
    assert minimum(2, 9) == 2
    assert minimum(2.3, 5.2) == 2.3
    assert minimum("z", "c") == "c"
    assert minimum(9, 2) == 2


def test_le_minimum_de_trois_valeurs_de_la_planche_72() -> None:
    """2023-2024 : `min(5, min(10, 1))` vaut 1."""
    assert minimum(5, minimum(10, 1)) == 1
    assert minimum(minimum(5, 10), 1) == 1


def test_le_minimum_de_deux_chaines_de_la_planche_69() -> None:
    """2024-2025 : `point<string>("Salut", " A vous")`, l'espace precede `S`."""
    assert minimum("Salut", " A vous") == " A vous"


def test_le_type_que_le_patron_deduit_de_ses_arguments() -> None:
    """Lien avec `type_deduit` : `int`, `double` -- et non `float` --, `char`."""
    parametres = ("T", "T")
    for valeurs, guillemets, attendu in (
        ((2, 9), '"', "int"),
        ((2.3, 5.2), '"', "double"),
        (("z", "c"), "'", "char"),
    ):
        arguments = [type_deduit(v, guillemets) for v in valeurs]
        assert types_deduits(parametres, arguments) == {"T": attendu}


def test_deux_arguments_de_types_differents_contredisent_le_patron() -> None:
    """`min(2, 9.5)` : `T` ne peut etre a la fois `int` et `double`."""
    with pytest.raises(TypeError, match="deduction contradictoire pour "):
        types_deduits(("T", "T"), ("int", "double"))


def test_les_trois_parametres_de_type_de_la_planche_50() -> None:
    """`Fonct(2, 5.2, "Cool")` : chacun des trois types est deduit a part."""
    assert types_deduits(("T", "T1", "T2"), ("int", "double", "const char*")) == {
        "T": "int",
        "T1": "double",
        "T2": "const char*",
    }
    deduits = types_deduits(("T", "T1", "T2"), ("double", "double", "double"))
    assert set(deduits.values()) == {"double"}


def test_un_parametre_qui_n_est_pas_un_patron_ne_deduit_rien() -> None:
    """`void Saisie(T Tab, int N)` : seul `T` est a deduire."""
    assert types_deduits(("T", "int"), ("float*", "int"), fixes=("int",)) == {
        "T": "float*"
    }


def test_le_patron_recoit_autant_d_arguments_que_de_parametres() -> None:
    """`min(5, 10, 1)` ne convient pas au patron a deux parametres."""
    with pytest.raises(TypeError, match="nombre d'arguments inattendu"):
        types_deduits(("T", "T"), ("int", "int", "int"))


def test_les_sommes_de_l_exercice_6_2() -> None:
    """2024-2025, planche 67 : 15, 15.8, et `t` pour « H E L L O »."""
    assert somme_du_patron((1, 2, 3, 4, 5), "int") == 15
    reelle = somme_du_patron((1.2, 3.4, 4.5, 6.7), "float")
    assert ecrire_un_reel(reelle) == "15.8"
    lettres = somme_du_patron("HELLO", "char")
    assert sum(map(ord, "HELLO")) == 372
    assert lettres == 372 - 256
    assert chr(int(lettres)) == "t"


def test_la_somme_de_caracteres_peut_devenir_negative() -> None:
    """`'A' + 'A'` vaut 130 : dans un `char` signe, -126."""
    assert somme_du_patron("AA", "char") == -126
