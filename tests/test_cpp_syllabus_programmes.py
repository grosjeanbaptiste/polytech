"""Syllabus « Introduction a la programmation en C++ » : ses programmes.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf`, chapitres 2 a 8. Chaque
« Programme Chap.Num » dont le syllabus imprime l'execution est repris ici,
sur les symboles que les planches avaient deja fait naitre.

Chaque affichage a ete confirme en compilant le programme (clang++ 21,
C++17). La compilation etablit quatre ecarts :

* pages 15 a 19 : la constante est ecrite `PI = 3.141596` ; pi vaut
  3.141593. Les resultats imprimes (12.5664, 78.5399, 31.416) sont bien
  ceux de la constante fautive ;
* page 10 : le tableau des caracteres speciaux donne `0x2C` pour
  l'apostrophe ; `0x2C` est la virgule, l'apostrophe vaut `0x27` -- ce que
  dit la table ASCII de l'annexe du meme syllabus ;
* page 35 : l'exemple du `goto` imprime `i/j=0.00` ; `(float)0 / -2` vaut
  `-0`, et c'est `-0` que le programme affiche ;
* page 60, programme 5.6 : l'execution imprimee place « Dans la fonction
  int 5 » avant « Retour de la fonction int ». Depuis C++17 les operandes
  de `<<` s'evaluent de gauche a droite, et l'ordre est l'inverse.

Un cinquieme soupcon a ete leve en regardant la page : le fichier de la
page 80 semblait porter `12345` ; l'image montre bien `1 2 3 4 5`, les
blancs s'etaient perdus a l'extraction du texte.
"""

from __future__ import annotations

import math

import pytest

from polytech.algorithmique.domain.caracteres import caractere_de_code, code_ascii
from polytech.algorithmique.domain.cercle import disque_du_syllabus
from polytech.algorithmique.domain.controle import (
    boucle_for,
    jusqu_au_break,
    sans_les_continue,
)
from polytech.algorithmique.domain.fichiers import Disque, tours_de_la_boucle_eof
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.flux_de_sortie import Ecran, ecrire_un_reel
from polytech.algorithmique.domain.identificateurs import est_un_identificateur
from polytech.algorithmique.domain.memoire import Memoire
from polytech.algorithmique.domain.operateurs import Point
from polytech.algorithmique.domain.portee import Portees
from polytech.algorithmique.domain.surcharge import (
    arguments_completes,
    surcharge_choisie,
)
from polytech.algorithmique.domain.types_cpp import TYPES_ENTIERS, diviser

PI_DU_SYLLABUS = 3.141596
"""Programmes 2.3 a 2.8, pages 15 a 19 : la constante telle que le syllabus l'ecrit."""
PI_A_SEPT_CHIFFRES = float(ecrire_un_reel(math.pi, 7))
"""3.141593 : pi arrondi a sept chiffres, comme `cout` le ferait (bibliotheque)."""
FICHIER_DE_LA_PAGE_80 = "1 2 3 4 5\n6 7 8 9\n10 11 12\n13 14\n15"


def test_la_surface_et_la_circonference_du_programme_2_3() -> None:
    """Page 17 : rayon 2 -- « vaut 12.5664 » pour l'une comme pour l'autre."""
    disque = disque_du_syllabus(PI_DU_SYLLABUS, 2)
    assert (ecrire_un_reel(disque.surface), ecrire_un_reel(disque.circonference)) == (
        "12.5664",
    ) * 2


def test_la_surface_et_la_circonference_du_programme_2_8() -> None:
    """Page 19 : rayon 5 -- surface 78.5399, circonference 31.416."""
    disque = disque_du_syllabus(PI_DU_SYLLABUS, 5)
    assert (ecrire_un_reel(disque.surface), ecrire_un_reel(disque.circonference)) == (
        "78.5399",
        "31.416",
    )


def test_la_constante_pi_du_syllabus_est_fautive() -> None:
    """Ecart : `3.141596` pour 3.141593 ; avec pi, la surface serait 78.5398."""
    assert ecrire_un_reel(math.pi, 7) == "3.141593"
    assert pytest.approx(3.141593, abs=1e-9) == PI_A_SEPT_CHIFFRES
    assert ecrire_un_reel(disque_du_syllabus(math.pi, 5).surface) == "78.5398"
    assert ecrire_un_reel(disque_du_syllabus(math.pi, 5).circonference) == "31.4159"
    assert (
        ecrire_un_reel(disque_du_syllabus(PI_A_SEPT_CHIFFRES, 5).surface) == "78.5398"
    )
    with pytest.raises(ValueError, match="rayon negatif"):
        disque_du_syllabus(math.pi, -1)


def test_les_precisions_du_programme_2_5() -> None:
    """Page 18 : 5.12346, puis 5.1, 5.123, 5.12346 et 5.1234567."""
    v = 5.1234567
    assert [ecrire_un_reel(v, p) for p in (6, 2, 4, 6, 8)] == [
        "5.12346",
        "5.1",
        "5.123",
        "5.12346",
        "5.1234567",
    ]


def test_les_largeurs_et_le_remplissage_du_programme_2_6() -> None:
    """Page 18 : `width(5)`, `width(i + 3)`, puis les memes apres `fill('.')`."""
    ecran = Ecran()
    ecran.setw(5).ecrire(0).endl().setw(3).ecrire(0).endl().setw(4).ecrire(1).endl()
    ecran.setfill(".").setw(3).ecrire(0).endl().setw(4).ecrire(1).endl()
    assert ecran.lignes() == ["    0", "  0", "   1", "..0", "...1"]


def test_les_tailles_du_programme_2_4() -> None:
    """Page 17 : char 1, int 4, short 2, long 4 -- signes ou non."""
    octets = {nom: type_entier.bits // 8 for nom, type_entier in TYPES_ENTIERS.items()}
    assert (octets["char"], octets["unsigned char"]) == (1, 1)
    assert (octets["int"], octets["unsigned int"]) == (4, 4)
    assert (octets["short int"], octets["long int"]) == (2, 4)


def test_le_code_de_l_apostrophe_n_est_pas_celui_du_tableau() -> None:
    """Ecart de la page 10 : `0x2C` est la virgule ; l'apostrophe vaut `0x27`."""
    assert (code_ascii("'"), caractere_de_code(0x2C)) == (0x27, ",")
    assert code_ascii(caractere_de_code(0x2C)) == 0x2C
    assert (code_ascii("\\"), code_ascii("\a"), code_ascii("\n")) == (0x5C, 7, 0x0A)
    with pytest.raises(ValueError, match="hors de la table ASCII"):
        code_ascii("é")


def test_les_identificateurs_de_l_exercice_2_1() -> None:
    """Page 20 : `long` et `double` sont reserves, `2x` commence par un chiffre."""
    candidats = (
        "record",
        "long",
        "%interet",
        "étudiant",
        "MaVariable3",
        "ma_var_4",
        "2x",
        "Long",
        "double",
        "ma-var-5",
    )
    valides = [nom for nom in candidats if est_un_identificateur(nom)]
    assert valides == ["record", "MaVariable3", "ma_var_4", "Long"]


def test_les_neuf_tours_du_programme_3_1() -> None:
    """Page 27 : `i` monte, `k` descend de 9 a 1 ; `F = (float)i/k + r`."""
    tours = boucle_for((0, 5, 9), lambda e: e[0] < e[1] and e[2] != 0, _suivant_du_3_1)
    assert len(tours.valeurs) == 9
    affiches = [ecrire_un_reel(i / k + 0.2 * i) for i, _, k in tours.valeurs]
    assert affiches == [
        "0",
        "0.325",
        "0.685714",
        "1.1",
        "1.6",
        "2.25",
        "3.2",
        "4.9",
        "9.6",
    ]


def _suivant_du_3_1(etat: tuple[int, int, int]) -> tuple[int, int, int]:
    i, j, k = etat
    return i + 1, j + 1, k - 1


def test_le_while_de_la_page_29_fait_trois_tours() -> None:
    """`int i = 3; while (i < 6) i++;` : valeurs 3, 4, 5 ; a la sortie, 6."""
    parcours = boucle_for(3, lambda i: i < 6, lambda i: i + 1)
    assert (parcours.valeurs, parcours.sortie) == ((3, 4, 5), 6)


def test_le_break_de_la_page_34_arrete_la_boucle_interieure() -> None:
    """Pour chaque i : « j= 0 ; j= 1 ; », puis le saut de ligne et la sortie."""
    for _ in range(3):
        assert jusqu_au_break(range(10), lambda j: j == 2) == (0, 1, 2)


def test_le_continue_de_la_page_34_ne_saute_qu_un_tour() -> None:
    """Pour chaque i : j vaut 0, 1, puis 3 -- le tour ou j vaut 2 est saute."""
    assert sans_les_continue(range(4), lambda j: j == 2) == (0, 1, 3)


def test_le_quotient_de_l_exemple_du_goto_est_un_zero_negatif() -> None:
    """Ecart de la page 35 : `(float)0 / -2` s'affiche `-0`, non `0.00`."""
    assert ecrire_un_reel(diviser(0.0, -2)) == "-0"
    assert ecrire_un_reel(diviser(0.0, -1)) == "-0"


def test_les_moyennes_du_programme_5_1() -> None:
    """Page 52 : `moyenne(i, i*i)` rend 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55."""
    moyennes = [diviser(i + i * i, 2.0) for i in range(11)]
    assert moyennes == [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55]


def test_la_variable_statique_et_la_globale_du_programme_5_5() -> None:
    """Page 59 : globale -1 a -5 ; la statique, initialisee une fois, 0 a 8."""
    portees = Portees()
    portees.declarer("globale", 0)
    releves = []
    for _ in range(5):
        portees.affecter("globale", portees.valeur("globale") - 1)
        portees.ouvrir()
        portees.declarer_statique("fonc_Static", "i", 0)
        releves.append((portees.valeur("globale"), portees.valeur("i")))
        portees.affecter("i", portees.valeur("i") + 2)
        portees.fermer()
    assert releves == [(-1, 0), (-2, 2), (-3, 4), (-4, 6), (-5, 8)]


def test_les_trois_surcharges_du_programme_5_6() -> None:
    """Page 60 : `Somme(i, j)` entiere, `Somme(k, z)` reelle, `Somme(1, 2, 3)`."""
    signatures = (("int", "int"), ("float", "float"), ("int", "int", "int"))
    assert surcharge_choisie(signatures, ("int", "int")) == 0
    assert surcharge_choisie(signatures, ("float", "float")) == 1
    assert surcharge_choisie(signatures, ("int", "int", "int")) == 2
    assert (2 + 3, ecrire_un_reel(2.3 + 2.2), 1 + 2 + 3) == (5, "4.5", 6)


def test_les_trois_volumes_du_programme_5_8() -> None:
    """Page 61 : 100, puis 50 (hauteur 1), puis 30 (largeur 3, hauteur 1)."""
    defauts = (0, 3, 1)
    volumes = [
        math.prod(arguments_completes(defauts, arguments))
        for arguments in ((10, 5, 2), (10, 5), (10,))
    ]
    assert volumes == [100, 50, 30]


def test_les_pointeurs_de_l_exemple_4() -> None:
    """Page 65 : j = 35 ; `*p = 10` ; `(*p)++` rend 21 ; `*(q+1) = 29` epargne i."""
    memoire = Memoire()
    p = memoire.declarer("i", 35)
    memoire.declarer("j", memoire.contenu(p))
    assert memoire.valeur("j") == 35
    memoire.ecrire(p, 10)
    assert memoire.valeur("i") == 10
    memoire.affecter("i", 20)
    memoire.ecrire(p, memoire.contenu(p) + 1)
    assert memoire.valeur("i") == 21
    q = p
    memoire.affecter("i", 50)
    memoire.ecrire(q + 4, 29)
    assert (memoire.contenu(p), memoire.contenu(q), memoire.valeur("i")) == (50,) * 3


def test_la_reference_de_l_exemple_5() -> None:
    """Page 66 : `ref = 100` ecrit dans i ; `i = j` remet 35 sous les deux noms."""
    memoire = Memoire()
    memoire.declarer("i", 35)
    memoire.referencer("ref", "i")
    memoire.declarer("j", memoire.valeur("ref"))
    memoire.affecter("ref", 100)
    assert (memoire.valeur("i"), memoire.valeur("j")) == (100, 35)
    memoire.affecter("i", memoire.valeur("j"))
    assert memoire.valeur("ref") == 35


def test_la_saisie_du_programme_6_3() -> None:
    """Page 69 : l'entier et le caractere par reference changent, le reel non."""
    memoire = Memoire()
    for nom, valeur in (("N", 5), ("f", 6.7), ("C", "A")):
        memoire.declarer(nom, valeur)
    memoire.referencer("i", "N")
    memoire.referencer("cc", "C")
    memoire.declarer("r", memoire.valeur("f"))
    for nom, saisi in (("i", 12), ("r", 12.13), ("cc", "Z")):
        memoire.affecter(nom, saisi)
    assert [memoire.valeur(nom) for nom in ("N", "f", "C")] == [12, 6.7, "Z"]


def test_le_pointeur_qui_avance_dans_le_tableau_du_programme_6_4() -> None:
    """Page 71 : `*p` vaut 2 ; apres `p++`, 6 ; apres `p += 4`, 8."""
    memoire = Memoire(1000)
    p = memoire.declarer_un_tableau("T", [2, 6, 4, 5, 10, 8])
    lus = [memoire.contenu(p), memoire.contenu(p + 4), memoire.contenu(p + 4 + 4 * 4)]
    assert (p, lus) == (1000, [2, 6, 8])


def test_l_affectation_puis_l_avance_du_programme_6_5() -> None:
    """Page 72 : `*p++ = 15` ecrit T[0], p designe T[1] ; `p[5]` sort du tableau."""
    memoire = Memoire(0x0012FF68)
    p = memoire.declarer_un_tableau("T", [2, 4, 6, 8, 10, 12])
    memoire.ecrire(p, 15)
    p += 4
    assert memoire.tableau("T", 6) == [15, 4, 6, 8, 10, 12]
    assert (p, memoire.contenu(p), memoire.contenu(p + 16)) == (0x0012FF6C, 4, 12)
    with pytest.raises(KeyError, match="adresse hors des variables declarees"):
        memoire.contenu(p + 20)


def test_l_exercice_6_2_pas_a_pas() -> None:
    """Page 74 : x = 10, y = 9 ; puis 11, 12, 13 ; `*p++` deplace p sans toucher x."""
    memoire = Memoire()
    p = memoire.declarer("x", 10)
    memoire.referencer("r", "x")
    y = memoire.contenu(p) - 1
    etapes = []
    memoire.ecrire(p, memoire.contenu(p) + 1)
    etapes.append(memoire.valeur("x"))
    memoire.ecrire(p, memoire.contenu(p) + 1)
    etapes.append(memoire.valeur("x"))
    memoire.affecter("r", memoire.valeur("r") + 1)
    etapes.append(memoire.valeur("x"))
    p += 4
    assert (y, etapes, memoire.valeur("x")) == (9, [11, 12, 13], 13)
    assert p != memoire.adresse_de("x")


def test_les_programmes_7_1_et_7_2_relisent_ce_qu_ils_ont_ecrit() -> None:
    """Page 79 : « Car = A Entier = 10 Reel = 3.14 Double = 7.12345 »."""
    disque = Disque()
    for separateur in ("\n", " "):
        disque.ecrire("fichier.txt", separateur.join(("A", "10", "3.14", "7.12345")))
        lecture = disque.lire("fichier.txt")
        assert [lecture.mot() for _ in range(4)] == ["A", "10", "3.14", "7.12345"]


def test_l_extraction_s_arrete_au_blanc_et_getline_prend_la_ligne() -> None:
    """Page 80 : `input >> TXT` ne rend que « Bonjour »."""
    message = "Bonjour : Debut du fichier ...\n"
    assert Clavier(message).mot() == "Bonjour"
    assert Clavier(message).ligne() == "Bonjour : Debut du fichier ..."


def test_les_quinze_entiers_du_programme_7_3() -> None:
    """Page 81 : un titre, puis « 1 ; 2 ; ... ; 15 ; » -- quinze tours de boucle."""
    disque = Disque()
    disque.ecrire("fichier.txt", "Les donnees sont :\n" + FICHIER_DE_LA_PAGE_80)
    lecture = disque.lire("fichier.txt")
    assert lecture.ligne() == "Les donnees sont :"
    assert [int(lecture.mot()) for _ in range(15)] == list(range(1, 16))
    assert tours_de_la_boucle_eof(FICHIER_DE_LA_PAGE_80, champs=1) == 15


def test_un_retour_a_la_ligne_final_ajoute_un_tour_a_la_boucle_eof() -> None:
    """Le piege du programme 7.3 : le dernier entier serait affiche deux fois."""
    assert tours_de_la_boucle_eof(FICHIER_DE_LA_PAGE_80 + "\n", champs=1) == 16


def test_un_nom_a_espace_occupe_deux_lectures_du_programme_7_4() -> None:
    """Page 82 : « Bianca Castafiore » devient Nom 2 et Nom 3."""
    clavier = Clavier("Tintin\nBianca Castafiore\nMilou\nHaddock\nFIN\n")
    noms = []
    while (nom := clavier.mot()) != "FIN":
        noms.append(nom)
    assert noms == ["Tintin", "Bianca", "Castafiore", "Milou", "Haddock"]


def test_le_module_et_la_somme_de_deux_points_de_la_page_87() -> None:
    """`z1 = {1, -4}`, `z2 = {5, 2}` : module de z1 > 1, donc z3 = (6, -2)."""
    z1, z2 = Point(1, -4), Point(5, 2)
    assert z1.distance(Point(0, 0)) == pytest.approx(math.sqrt(17))
    assert z1.distance(Point(0, 0)) > 1.0
    assert z1 + z2 == Point(6, -2)


def test_la_structure_par_valeur_par_pointeur_et_par_reference() -> None:
    """Programme 8.2, page 89 : (0, 0) par valeur, (10, 20) par les deux autres."""
    memoire = Memoire()
    adresse = memoire.declarer("mon_pt", Point(0, 0))
    memoire.declarer("copie", memoire.valeur("mon_pt"))
    memoire.affecter("copie", Point(10, 20))
    assert memoire.valeur("mon_pt") == Point(0, 0)
    memoire.ecrire(adresse, Point(10, 20))
    assert memoire.valeur("mon_pt") == Point(10, 20)
    memoire.affecter("mon_pt", Point(0, 0))
    memoire.referencer("le_pt", "mon_pt")
    memoire.affecter("le_pt", Point(10, 20))
    assert memoire.valeur("mon_pt") == Point(10, 20)
