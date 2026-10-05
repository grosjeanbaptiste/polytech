"""Syllabus « Introduction a la programmation en C++ » : exercices et erreurs.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf` -- les exercices des
chapitres 3, 4, 5 et 8, et le chapitre 10, « Quelques erreurs a eviter ».

Le syllabus ne donne pas les solutions de ses exercices. Les oracles sont
donc de deux sortes : les exemples chiffres que l'enonce imprime (tableaux
des exercices 4.4, 4.10, 8.5, 8.6 et 8.8, distances du 4.12, produit
scalaire du 5.4), et, pour les exercices sans exemple, une valeur calculee
par un second chemin, independant de la fonction eprouvee.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import sympy as sp

from polytech.algorithmique.domain.algorithmes_stl import count
from polytech.algorithmique.domain.calculs_iteratifs import (
    dernier_fibonacci_au_plus,
    dernier_fibonacci_representable,
    nombres_premiers,
    sommes_partielles,
)
from polytech.algorithmique.domain.chaines import longueur, tableau_de_char
from polytech.algorithmique.domain.controle import (
    BoucleInfinie,
    Cas,
    boucle_for,
    selon,
)
from polytech.algorithmique.domain.enregistrements import trier_selon
from polytech.algorithmique.domain.second_degre import (
    Nature,
    discuter_le_second_degre,
)
from polytech.algorithmique.domain.tableaux import TableauStatique
from polytech.algorithmique.domain.tableaux_de_structures import (
    exiger_strictement_entre,
    extremes,
    sans_ceux,
)
from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS, diviser
from polytech.maths.domain.algebre.metrique import produit_scalaire
from polytech.maths.domain.algebre.racines import racines_reelles_du_second_degre

DISTANCES = {
    ("Bruxelles", "Namur"): 63,
    ("Bruxelles", "Liege"): 97,
    ("Bruxelles", "Charleroi"): 62,
    ("Bruxelles", "Mons"): 68,
    ("Namur", "Charleroi"): 37,
    ("Namur", "Liege"): 65,
    ("Namur", "Mons"): 74,
    ("Charleroi", "Liege"): 94,
    ("Charleroi", "Mons"): 50,
    ("Liege", "Mons"): 130,
}
ELEMENTS = [
    ("van cool", "P", (45, 54)),
    ("mon_nom", "M", (2, 1)),
    ("son_nom", "T", (29, 2)),
    ("van cool", "A", (40, 41)),
    ("aladdin", "Y", (40, 42)),
]
LIGNES = [
    "ceci est un test",
    "qui compte le nombre",
    "de e et de i ainsi",
    "la longueur de",
    "la ligne",
]
PERSONNES = [("Riri", 20, 70), ("Fifi", 40, 60), ("Lou Lou", 30, 75)]


def _seuil_de_lipolyse(age: int, repos: int) -> int:
    return int(diviser(220 - age - repos, 2)) + repos


def test_une_boucle_sans_accolades_ne_s_arrete_pas() -> None:
    """Exercice 3.0 : `i++` est hors du `while`, la condition reste vraie."""
    with pytest.raises(BoucleInfinie):
        boucle_for(1, lambda i: i < 5, lambda i: i)


def test_sans_break_toutes_les_branches_s_enchainent() -> None:
    """Exercice 3.2 et page 104 : `print_chiffre(1)` affiche « undeuxtrois »."""
    cas = [
        Cas(i, mot, avec_break=False)
        for i, mot in ((1, "un"), (2, "deux"), (3, "trois"))
    ]
    assert "".join(selon(1, cas, None)) == "undeuxtrois"
    assert "".join(selon(2, cas, "1,2 ou 3 ?")) == "deuxtrois1,2 ou 3 ?"


def test_les_cent_premiers_nombres_premiers_de_l_exercice_3_5() -> None:
    """Le centieme vaut 541 ; sympy, par un autre chemin, donne les memes."""
    premiers = nombres_premiers(100)
    assert (premiers[:5], premiers[-1]) == ((2, 3, 5, 7, 11), 541)
    assert premiers == tuple(sp.prime(rang) for rang in range(1, 101))


def test_aucun_nombre_premier_n_est_demande_a_moins_de_zero() -> None:
    """Zero nombre premier est une liste vide ; un nombre negatif, une erreur."""
    assert nombres_premiers(0) == ()
    with pytest.raises(ValueError, match="nombre negatif de nombres premiers"):
        nombres_premiers(-1)


def test_le_dernier_terme_de_fibonacci_qui_tient_dans_un_int() -> None:
    """Exercice 3.12 : avec `U1 = U2 = 1`, le rang 46 vaut 1 836 311 903."""
    rang, valeur = dernier_fibonacci_representable("int")
    assert (rang, valeur) == (46, 1836311903)
    assert TYPES_ENTIERS["int"].contient(valeur)
    phi = (1 + math.sqrt(5)) / 2
    assert round(phi**47 / math.sqrt(5)) > TYPES_ENTIERS["int"].maximum


def test_le_dernier_terme_depend_du_type() -> None:
    """Un `char` s'arrete au rang 11, a 89 ; un `unsigned int` va un rang plus loin."""
    assert dernier_fibonacci_representable("char") == (11, 89)
    assert dernier_fibonacci_representable("unsigned char") == (13, 233)
    assert dernier_fibonacci_representable("unsigned int") == (47, 2971215073)
    with pytest.raises(ValueError, match="type entier inconnu"):
        dernier_fibonacci_representable("double")


def test_la_serie_de_l_exercice_3_13() -> None:
    """`S = 1 - X.3!/(1.2) + X^2.5!/(2.4) - X^3.7!/(3.8) + X^4.9!/(4.16) ...`."""
    assert sommes_partielles(1, 5) == (1, -2, 13, -197, 5473)


def test_chaque_terme_de_la_serie_se_deduit_du_precedent() -> None:
    """Second chemin : le terme general `(-1)^n X^n (2n+1)! / (n 2^n)`, en fractions."""
    x = Fraction(1, 3)
    attendu, somme = [], Fraction(1)
    attendu.append(somme)
    for n in range(1, 7):
        somme += (-1) ** n * x**n * math.factorial(2 * n + 1) / Fraction(n * 2**n)
        attendu.append(somme)
    assert sommes_partielles(x, 7) == tuple(attendu)


def test_une_serie_a_au_moins_un_terme() -> None:
    """« Le nombre de termes a prendre en compte » ne peut etre nul."""
    with pytest.raises(ValueError, match="au moins un terme"):
        sommes_partielles(1, 0)


@pytest.mark.parametrize(
    ("coefficients", "nature", "racines"),
    [
        ((1, -3, 2), Nature.DEUX_RACINES_REELLES, (1.0, 2.0)),
        ((1, -2, 1), Nature.RACINE_DOUBLE, (1.0,)),
        ((1, 0, 1), Nature.DEUX_RACINES_COMPLEXES, (-1j, 1j)),
        ((1, 2, 5), Nature.DEUX_RACINES_COMPLEXES, (-1 - 2j, -1 + 2j)),
        ((0, 2, -4), Nature.PREMIER_DEGRE, (2.0,)),
        ((0, 0, 3), Nature.AUCUNE_SOLUTION, ()),
        ((0, 0, 0), Nature.INFINITE_DE_SOLUTIONS, ()),
    ],
)
def test_toutes_les_possibilites_de_l_exercice_3_4(
    coefficients: tuple[float, float, float],
    nature: Nature,
    racines: tuple[complex, ...],
) -> None:
    """Racines reelles, complexes, double, infinite ou inexistence de solutions."""
    discussion = discuter_le_second_degre(*coefficients)
    assert discussion.nature is nature
    assert discussion.racines == pytest.approx(racines)


def test_la_discussion_concorde_avec_le_cours_de_mathematique() -> None:
    """Lien : les racines reelles sont celles du syllabus d'algebre."""
    for a, b, c in ((1, -3, 2), (2, 4, -6), (1, -2, 1)):
        reelles = discuter_le_second_degre(a, b, c).racines
        assert sorted(reelles) == pytest.approx(
            sorted(set(racines_reelles_du_second_degre(a, b, c)))
        )
    assert racines_reelles_du_second_degre(1, 0, 1) == ()


def test_supprimer_les_bords_de_l_exercice_4_4() -> None:
    """`0 1 2 3 4 5 6`, puis `1 2 3 4 5`, puis `2 3 4`."""
    tab = TableauStatique(20, list(range(7)))
    for attendu in ([1, 2, 3, 4, 5], [2, 3, 4]):
        tab.supprimer(0)
        tab.supprimer(tab.taille - 1)
        assert tab.elements == attendu


def test_supprimer_les_valeurs_superieures_a_7_de_l_exercice_4_10() -> None:
    """`8 2 7 9 1 9` devient `2 7 1`."""
    assert sans_ceux([8, 2, 7, 9, 1, 9], lambda v: v > 7) == (2, 7, 1)


def test_les_deux_villes_les_plus_proches_de_l_exercice_4_12() -> None:
    """Dix distances entre cinq villes : Namur et Charleroi, 37 km."""
    _, plus_proches = extremes(list(DISTANCES.items()), lambda paire: paire[1])
    assert plus_proches == (("Namur", "Charleroi"), 37)
    villes = {ville for paire in DISTANCES for ville in paire}
    assert len(DISTANCES) == math.comb(len(villes), 2) == 10


def test_le_produit_scalaire_de_l_exercice_5_4() -> None:
    """Lien avec l'algebre : `(5 2 -4) . (1 -3 6) = -25`."""
    assert produit_scalaire((5, 2, -4), (1, -3, 6)) == -25


def test_le_tri_selon_le_nom_et_le_code_de_l_exercice_8_5() -> None:
    """Dans l'ordre : aladdin Y, mon_nom M, son_nom T, van cool A, van cool P."""
    tries = trier_selon(ELEMENTS, lambda e: (e[0], e[1]))
    assert [(nom, code) for nom, code, _ in tries] == [
        ("aladdin", "Y"),
        ("mon_nom", "M"),
        ("son_nom", "T"),
        ("van cool", "A"),
        ("van cool", "P"),
    ]


def test_la_suppression_puis_la_sauvegarde_de_l_exercice_8_5() -> None:
    """Sans « van cool A », seuil 80 : aladdin (82) et van cool P (99) sont gardes."""
    restants = sans_ceux(
        trier_selon(ELEMENTS, lambda e: (e[0], e[1])),
        lambda e: (e[0], e[1]) == ("van cool", "A"),
    )
    exiger_strictement_entre(80, 0, 100)
    sauves = sans_ceux(restants, lambda e: sum(e[2]) <= 80)
    assert [(nom, code) for nom, code, _ in sauves] == [
        ("aladdin", "Y"),
        ("van cool", "P"),
    ]


def test_le_compte_des_e_et_des_i_de_l_exercice_8_6() -> None:
    """« ceci est un test » : 3, 1, 4 ; « de e et de i ainsi » : 4, 3, 7."""
    comptes = [(count(ligne, "e"), count(ligne, "i")) for ligne in LIGNES]
    assert comptes == [(3, 1), (3, 1), (4, 3), (2, 0), (1, 1)]
    sommes = [e + i for e, i in comptes]
    maximum, _ = extremes(list(enumerate(sommes)), lambda paire: paire[1])
    assert maximum == (2, 7)


def test_le_tri_decroissant_selon_la_longueur_de_l_exercice_8_6() -> None:
    """Longueurs 20, 18, 16, 14 et 8."""
    tries = trier_selon(LIGNES, lambda ligne: -len(ligne))
    assert [len(ligne) for ligne in tries] == [20, 18, 16, 14, 8]
    assert tries[0] == "qui compte le nombre"


def test_le_seuil_de_lipolyse_de_l_exercice_8_8() -> None:
    """25 ans, 70 au repos : 125, puis 125/2 = 62 en entiers, puis 132."""
    assert diviser(220 - 25 - 70, 2) == 62
    assert _seuil_de_lipolyse(25, 70) == 132
    assert [_seuil_de_lipolyse(age, repos) for _, age, repos in PERSONNES] == [
        135,
        120,
        132,
    ]


def test_les_seuils_par_tranche_d_age_de_l_exercice_8_8() -> None:
    """F = 70 : pas de 5, « 137 135 132 ... » ; pas de 10, « 137 132 ... »."""
    par_cinq = [_seuil_de_lipolyse(age, 70) for age in range(15, 46, 5)]
    par_dix = [_seuil_de_lipolyse(age, 70) for age in range(15, 46, 10)]
    assert par_cinq == [137, 135, 132, 130, 127, 125, 122]
    assert par_dix == [137, 132, 127, 122]


def test_effacer_les_plus_jeunes_de_l_exercice_8_8() -> None:
    """V = 35, entre 10 et 50 : il ne reste que Fifi."""
    exiger_strictement_entre(35, 10, 50)
    assert sans_ceux(PERSONNES, lambda p: p[1] < 35) == (("Fifi", 40, 60),)
    with pytest.raises(ValueError, match="trop grande"):
        exiger_strictement_entre(50, 10, 50)


def test_la_division_entiere_de_la_page_102() -> None:
    """Chapitre 10 : `F = B/A` vaut 2 ; `F = (float)B/A` vaut 2.5."""
    assert (diviser(25, 10), diviser(25.0, 10)) == (2, 2.5)


def test_sizeof_n_est_pas_la_longueur_de_la_chaine() -> None:
    """Page 103 : `char nom[30]` occupe 30 octets, quoi que l'on y ait tape."""
    nom = tableau_de_char("abc", 30)
    assert (len(nom), longueur(nom)) == (30, 3)


def test_la_boucle_dont_le_test_est_une_egalite_ne_tourne_pas() -> None:
    """Page 103 : `for (i = 0; i == N; i++)` ne s'execute que si N vaut 0."""
    assert boucle_for(0, lambda i: i == 5, lambda i: i + 1).valeurs == ()
    assert boucle_for(0, lambda i: i < 5, lambda i: i + 1).valeurs == (0, 1, 2, 3, 4)


def test_un_terme_egal_a_la_borne_est_encore_representable() -> None:
    """89 tient dans une borne de 89 ; dans une borne de 88, on s'arrete a 55."""
    assert dernier_fibonacci_au_plus(89) == (11, 89)
    assert dernier_fibonacci_au_plus(88) == (10, 55)
    assert dernier_fibonacci_au_plus(1) == (2, 1)
    with pytest.raises(ValueError, match="borne inferieure a 1"):
        dernier_fibonacci_au_plus(0)


def test_les_racines_restent_en_ordre_croissant_quand_a_est_negatif() -> None:
    """`-x^2 + 3x - 2` a pour racines 1 et 2 ; `-x^2 - 1`, -i puis i."""
    reelles = discuter_le_second_degre(-1, 3, -2)
    assert reelles.racines == pytest.approx((1.0, 2.0))
    complexes = discuter_le_second_degre(-1, 0, -1)
    assert complexes.racines == pytest.approx((-1j, 1j))
