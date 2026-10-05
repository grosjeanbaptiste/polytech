"""Programmation et algorithmique, partie 1 : pointeurs, references, fonctions.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 83 a 135. Adresse et contenu, pointeur, reference, les trois
passages de parametres, le passage d'un tableau, les regles de visibilite
(bloc, variable globale, variable statique), la recursivite, la surcharge
et les parametres par defaut.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit un ecart : planche 122,
l'appel `affiche(5.2, 6.3)` est ambigu et ne compile pas -- un `double`
se convertit aussi bien en `int` qu'en `float`.
"""

from __future__ import annotations

import math

import pytest

from polytech.algorithmique.domain.memoire import Memoire
from polytech.algorithmique.domain.portee import Portees
from polytech.algorithmique.domain.recursivite import (
    deplacements_de_hanoi,
    factorielle,
    fibonacci,
    pgcd,
    pgcd_recursif,
    puissance,
)
from polytech.algorithmique.domain.surcharge import (
    arguments_completes,
    surcharge_choisie,
)
from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS, tronquer

AFFICHE = (("int", "int"), ("float", "float"), ("int",))


def test_l_adresse_et_le_contenu_de_la_planche_83() -> None:
    """`i = 8` en `12FF80`, `j = 35` en `12FF84` ; `p = &j`, et `*p` vaut 35."""
    memoire = Memoire(0x12FF80)
    memoire.declarer("i", 8)
    memoire.declarer("j", 35)
    assert memoire.adresse_de("i") == 0x12FF80
    assert memoire.adresse_de("j") == 0x12FF84
    p = memoire.adresse_de("j")
    assert memoire.contenu(p) == 35
    assert memoire.adresse_de("j") - memoire.adresse_de("i") == 4


def test_le_pointeur_de_la_planche_84() -> None:
    """`p = &C` : `*p` vaut 4, et `*(&C)` aussi."""
    memoire = Memoire()
    memoire.declarer("C", 4)
    p = memoire.adresse_de("C")
    assert memoire.contenu(p) == 4
    assert memoire.contenu(memoire.adresse_de("C")) == 4
    memoire.ecrire(p, 9)
    assert memoire.valeur("C") == 9


def test_la_reference_de_la_planche_85() -> None:
    """`ref` est un autre nom de `i` : `ref = 100` change `i`, et `j` reste a 35."""
    memoire = Memoire()
    memoire.declarer("i", 35)
    memoire.declarer("j")
    memoire.referencer("ref", "i")
    memoire.affecter("j", memoire.valeur("ref"))
    memoire.affecter("ref", 100)
    assert (memoire.valeur("i"), memoire.valeur("j")) == (100, 35)
    memoire.affecter("i", memoire.valeur("j"))
    assert memoire.valeur("ref") == 35
    assert memoire.adresse_de("ref") == memoire.adresse_de("i")


def test_une_reference_doit_etre_initialisee() -> None:
    """Planche 85 : `int &ErRef;` est interdit."""
    with pytest.raises(KeyError, match="une reference doit designer une variable"):
        Memoire().referencer("ErRef", "inconnue")


def test_une_variable_non_initialisee_n_a_pas_de_valeur_definie() -> None:
    """Planches 117 et 148 : `-858993460`, ce que la memoire contenait avant."""
    memoire = Memoire()
    memoire.declarer("j")
    with pytest.raises(ValueError, match="variable non initialisee"):
        memoire.valeur("j")
    assert TYPES_ENTIERS["int"].ranger(0xCCCCCCCC) == -858_993_460


def _fonct(memoire: Memoire, mode: str) -> int:
    """Les quatre versions de `fonct` de la planche 110, appelees sur `var = 5`."""
    memoire.declarer("var", 5)
    if mode == "valeur":
        memoire.declarer("a", memoire.valeur("var"))
        memoire.affecter("a", 1)
    elif mode == "retour":
        memoire.declarer("a", memoire.valeur("var"))
        memoire.affecter("a", 1)
        memoire.affecter("var", memoire.valeur("a"))
    elif mode == "pointeur":
        memoire.ecrire(memoire.adresse_de("var"), 1)
    else:
        memoire.referencer("a", "var")
        memoire.affecter("a", 1)
    return int(memoire.valeur("var"))


def test_les_quatre_appels_de_la_planche_110() -> None:
    """Par valeur : 5 ; par retour, par pointeur, par reference : 1."""
    resultats = [
        _fonct(Memoire(), mode)
        for mode in ("valeur", "retour", "pointeur", "reference")
    ]
    assert resultats == [5, 1, 1, 1]


def test_modifier_par_valeur_pointeur_et_reference() -> None:
    """Planche 111 : dans la fonction 500 partout ; dans `main`, 5 puis 500 et 500."""
    par_valeur = Memoire()
    par_valeur.declarer("v", 5)
    par_valeur.declarer("copie", par_valeur.valeur("v"))
    par_valeur.affecter("copie", par_valeur.valeur("copie") * 100)
    assert (par_valeur.valeur("copie"), par_valeur.valeur("v")) == (500, 5)
    par_reference = Memoire()
    par_reference.declarer("v", 5)
    par_reference.referencer("alias", "v")
    par_reference.affecter("alias", par_reference.valeur("alias") * 100)
    assert par_reference.valeur("v") == 500


def test_le_programme_de_la_planche_113() -> None:
    """« Somme de a=2 et b=8 : 10 », puis « Modif : a=4 et b=12 : 5 »."""
    memoire = Memoire()
    for nom, valeur in (("a", 2), ("b", 8)):
        memoire.declarer(nom, valeur)
    memoire.declarer("c")
    memoire.declarer("y", memoire.valeur("b"))
    memoire.referencer("z", "c")
    memoire.affecter("z", memoire.valeur("a") + memoire.valeur("y"))
    memoire.affecter("y", 29)
    assert [memoire.valeur(n) for n in "abc"] == [2, 8, 10]
    memoire.declarer("x", memoire.valeur("a"))
    memoire.referencer("y2", "b")
    memoire.affecter("x", memoire.valeur("x") * 2)
    memoire.affecter("y2", memoire.valeur("x") + memoire.valeur("y2"))
    memoire.affecter("z", 5)
    memoire.affecter("a", memoire.valeur("x"))
    assert [memoire.valeur(n) for n in "abc"] == [4, 12, 5]


def test_un_tableau_est_passe_par_son_adresse() -> None:
    """Planche 125 : `Modif(p)` change `p` : « 1 ; 2 » devient « 5 ; 6 »."""
    memoire = Memoire()
    base = memoire.declarer_un_tableau("p", [1, 2])
    assert [memoire.contenu(base + 4 * rang) for rang in range(2)] == [1, 2]
    parametre = memoire.adresse_de("p")
    memoire.ecrire(parametre, 5)
    memoire.ecrire(parametre + 4, 6)
    assert memoire.tableau("p", 2) == [5, 6]


def test_une_variable_declaree_apres_un_tableau_ne_le_recouvre_pas() -> None:
    """Trois cases de quatre octets, puis la variable suivante, douze octets apres."""
    memoire = Memoire(0x2000)
    base = memoire.declarer_un_tableau("t", [7, 8, 9])
    suivante = memoire.declarer("n", 3)
    assert suivante == base + 12
    assert memoire.tableau("t", 3) == [7, 8, 9]
    assert memoire.valeur("n") == 3


def test_une_adresse_qui_ne_designe_rien_est_refusee() -> None:
    """Le C++ lirait ce qui s'y trouve ; ici la lecture est refusee."""
    memoire = Memoire()
    memoire.declarer("i", 1)
    with pytest.raises(KeyError, match="adresse hors des variables declarees"):
        memoire.contenu(0x10)
    with pytest.raises(KeyError, match="variable non declaree"):
        memoire.valeur("absente")


def test_la_visibilite_dans_un_bloc_a_la_planche_118() -> None:
    """`for (i = 0; ...)` laisse `i` a 3 ; `for (int i = 0; ...)` le laisse a 10."""
    meme = Portees()
    meme.declarer("i", 10)
    for valeur in range(4):
        meme.affecter("i", valeur)
    assert meme.valeur("i") == 3
    masquee = Portees()
    masquee.declarer("i", 10)
    masquee.ouvrir()
    masquee.declarer("i", 0)
    masquee.affecter("i", 3)
    assert masquee.valeur("i") == 3
    masquee.fermer()
    assert masquee.valeur("i") == 10


def test_une_variable_de_bloc_n_existe_plus_apres_lui() -> None:
    """Planche 118, troisieme programme : « Erreur »."""
    portees = Portees()
    portees.ouvrir()
    portees.declarer("i", 0)
    portees.fermer()
    with pytest.raises(NameError, match="identificateur non declare"):
        portees.valeur("i")
    with pytest.raises(RuntimeError, match="aucun bloc a fermer"):
        portees.fermer()


def test_la_variable_globale_de_la_planche_117() -> None:
    """`g` vaut 0, puis 2 apres `fonct` ; `i` et `j`, copies, restent 5 et 10."""
    portees = Portees()
    portees.declarer("g", 0)
    lignes = []
    portees.ouvrir()
    portees.declarer("un", 5)
    portees.declarer("deux", 10)
    lignes.append((portees.valeur("un"), portees.valeur("deux"), portees.valeur("g")))
    for nom in ("un", "deux", "g"):
        portees.affecter(nom, portees.valeur(nom) + 2)
    lignes.append((portees.valeur("un"), portees.valeur("deux"), portees.valeur("g")))
    portees.fermer()
    lignes.append((5, 10, portees.valeur("g")))
    assert lignes == [(5, 10, 0), (7, 12, 2), (5, 10, 2)]


def test_les_trois_versions_de_fct2_a_la_planche_119() -> None:
    """Une locale repart de 30 a chaque appel ; une `static` garde sa valeur."""
    portees = Portees()
    affichages = []
    for _ in range(2):
        portees.ouvrir()
        portees.declarer("I", 30)
        portees.affecter("I", portees.valeur("I") + 1)
        affichages.append(portees.valeur("I"))
        portees.fermer()
    assert affichages == [31, 31]
    statiques = []
    for _ in range(2):
        portees.ouvrir()
        portees.declarer_statique("Fct2", "I", 30)
        portees.affecter("I", portees.valeur("I") + 1)
        statiques.append(portees.valeur("I"))
        portees.fermer()
    assert statiques == [31, 32]


def test_la_factorielle_de_la_planche_120() -> None:
    """`n! = n (n-1)!`, `0! = 1`, et -1 pour code d'erreur si `n < 0`."""
    assert [factorielle(n) for n in range(6)] == [1, 1, 2, 6, 24, 120]
    assert factorielle(-3) == -1
    assert factorielle(10) == math.factorial(10)


def test_les_deux_pgcd_des_planches_95_et_120() -> None:
    """L'iteratif et le recursif s'accordent, et avec `math.gcd`."""
    for a, b in ((48, 18), (18, 48), (17, 5), (100, 75), (7, 7)):
        assert pgcd(a, b) == pgcd_recursif(a, b) == math.gcd(a, b)
    assert pgcd(48, 18) == 6
    with pytest.raises(ZeroDivisionError, match="division par zero"):
        pgcd_recursif(5, 0)
    assert pgcd(5, 0) == 5


def test_la_suite_de_fibonacci_de_la_planche_121() -> None:
    """`u_n = u_(n-1) + u_(n-2)`, avec `u_0 = u_1 = 1`."""
    assert [fibonacci(n) for n in range(10)] == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    with pytest.raises(ValueError, match="rang negatif"):
        fibonacci(-1)


def test_la_puissance_entiere_de_la_planche_121() -> None:
    """`a^n = 1` si `n = 0`, `a . a^(n-1)` sinon ; et les produits de la planche 102."""
    assert (puissance(2, 10), puissance(5, 3), puissance(7, 0)) == (1024, 125, 1)
    assert puissance(7, 9) == 40_353_607
    entiere = puissance(3, 21)
    assert isinstance(entiere, int)
    assert TYPES_ENTIERS["int"].ranger(entiere) == 1_870_418_611
    with pytest.raises(ValueError, match="exposant negatif"):
        puissance(2, -1)


def test_les_tours_de_hanoi() -> None:
    """Planche 121 : pour trois disques, sept deplacements ; pour `n`, `2^n - 1`."""
    trois = deplacements_de_hanoi(3, "A", "C", "B")
    assert ["".join(d) for d in trois] == ["AC", "AB", "CB", "AC", "BA", "BC", "AC"]
    for disques in (1, 4, 10):
        assert len(deplacements_de_hanoi(disques, "A", "C", "B")) == 2**disques - 1
    assert deplacements_de_hanoi(0, "A", "C", "B") == ()


def test_les_sommes_de_la_planche_97() -> None:
    """`Fnct_Som(5)`, `Fnct_Som(10)`, `Fnct_Som(7)` : 10, 45, 21."""
    assert [sum(range(n)) for n in (5, 10, 7)] == [10, 45, 21]
    assert tronquer((2.5 + 3) / 2 + (5 + 7.2) / 2) == 8


def test_la_surcharge_choisit_selon_les_arguments() -> None:
    """Planche 122 : `affiche(1, 2)` et `affiche(100)`."""
    assert surcharge_choisie(AFFICHE, ("int", "int")) == 0
    assert surcharge_choisie(AFFICHE, ("int",)) == 2
    assert surcharge_choisie(AFFICHE, ("float", "float")) == 1


def test_l_appel_de_la_planche_122_est_ambigu() -> None:
    """`affiche(5.2, 6.3)` : deux `double`, ni `int` ni `float` -- refuse."""
    with pytest.raises(TypeError, match="appel ambigu"):
        surcharge_choisie(AFFICHE, ("double", "double"))
    with pytest.raises(TypeError, match="appel ambigu"):
        surcharge_choisie(AFFICHE, ("float", "int"))
    with pytest.raises(TypeError, match="appel ambigu"):
        surcharge_choisie(AFFICHE, ("long", "long"))


def test_une_promotion_l_emporte_sur_une_conversion() -> None:
    """`affiche('a', 'b')` appelle la version `int` : un `char` se promeut en `int`."""
    assert surcharge_choisie(AFFICHE, ("char", "char")) == 0
    assert surcharge_choisie(AFFICHE, ("bool",)) == 2
    assert surcharge_choisie(AFFICHE, ("int", "double")) == 0
    assert surcharge_choisie(AFFICHE, ("double", "int")) == 0
    assert surcharge_choisie(AFFICHE, ("double",)) == 2


def test_aucune_surcharge_ne_prend_trois_arguments() -> None:
    with pytest.raises(TypeError, match="aucune fonction ne correspond"):
        surcharge_choisie(AFFICHE, ("int", "int", "int"))


def test_les_parametres_par_defaut_de_la_planche_123() -> None:
    """`affiche(100, 200)` affiche `100 200 3` ; `affiche()` affiche `1 2 3`."""
    defauts = (1, 2, 3)
    appels = [(1, 2, 3), (5, 6, 7), (100, 200), (1000,), ()]
    assert [arguments_completes(defauts, appel) for appel in appels] == [
        (1, 2, 3),
        (5, 6, 7),
        (100, 200, 3),
        (1000, 2, 3),
        (1, 2, 3),
    ]
    with pytest.raises(TypeError, match="trop d'arguments"):
        arguments_completes(defauts, (1, 2, 3, 4))
