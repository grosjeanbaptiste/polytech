"""Programmation et algorithmique, partie 1 : chaines de caracteres et saisie.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 15 et 71 a 80. Le tableau de `char` et son caractere nul, le type
`string`, ses constructeurs, sa comparaison, `stoi` et `stof`, puis la
lecture au clavier : `cin >>` qui s'arrete au premier blanc, `getline`
qui lit la ligne, et le `cin.ignore()` qu'il faut entre les deux.

Chaque resultat a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit un ecart : planches 71 et 73,
`char H[5] = "Hello";` ne compile pas en C++ -- la chaine demande six
cases, caractere nul compris.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.chaines import (
    NUL,
    longueur,
    sous_chaine,
    stof,
    stoi,
    tableau_de_char,
)
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.tris import tri_a_bulles


def test_un_tableau_de_char_se_termine_par_le_caractere_nul() -> None:
    """Planche 71 : `"Hello"` occupe six cases, `"Le main"` huit."""
    assert tableau_de_char("Hello") == ("H", "e", "l", "l", "o", NUL)
    assert len(tableau_de_char("Le main")) == 8
    assert tableau_de_char("Hello", 6) == tableau_de_char("Hello")
    assert ord(NUL) == 0


def test_la_longueur_s_arrete_au_caractere_nul() -> None:
    """`strlen` compte jusqu'au nul, `sizeof` compte les cases."""
    hello = tableau_de_char("Hello", 10)
    assert longueur(hello) == 5
    assert len(hello) == 10
    assert longueur(tableau_de_char("")) == 0


def test_un_tableau_trop_petit_pour_le_caractere_nul_est_refuse() -> None:
    """Planches 71 et 73 : `char H[5] = "Hello";` ne compile pas en C++."""
    with pytest.raises(ValueError, match="pas de place pour le caractere nul"):
        tableau_de_char("Hello", 5)
    with pytest.raises(ValueError, match="aucun caractere nul"):
        longueur(("H", "i"))


def test_les_constructeurs_de_string_de_la_planche_72() -> None:
    """`string s3(4, 'x')` vaut `xxxx` ; `string s4(s1, 4, 8)` vaut `our a vo`."""
    s1 = "Bonjour a vous"
    assert len(s1) == 14
    assert 4 * "x" == "xxxx"
    assert sous_chaine(s1, 4, 8) == "our a vo"
    assert sous_chaine(s1, 8, 100) == "a vous"
    with pytest.raises(IndexError, match="position hors de la chaine"):
        sous_chaine(s1, 15, 2)


def test_les_operations_permises_sur_un_string() -> None:
    """Planche 74 : affectation, concatenation, comparaison -- refusees a `char[]`."""
    s1, s2 = "Salut", "Hello"
    s1 = s2
    assert s1 == "Hello"
    s1 = "Hooo"
    assert s1 + s2 == "HoooHello"
    egales = s1 == s2
    assert not egales


def test_le_programme_de_la_planche_75() -> None:
    """`csou` et `pi` : longueurs 4 et 2, `pi csou`, et `nom1 < nom2`."""
    nom1, nom2 = "csou", "pi"
    assert (len(nom1), len(nom2)) == (4, 2)
    assert nom2 + " " + nom1 == "pi csou"
    assert nom1 < nom2


def test_stoi_et_stof_a_la_planche_76() -> None:
    """`stoi("3.45")` vaut 3, `stoi("67 Et 8 ")` vaut 67, `stof("3.45")` vaut 3.45."""
    assert stoi("12") == 12
    assert stoi("3.45") == 3
    assert stoi("67 Et 8 ") == 67
    assert stof("12") == 12.0
    assert stof("3.45") == 3.45


def test_stoi_lit_un_prefixe_apres_les_blancs_et_le_signe() -> None:
    """`"  42abc"`, `"-17"`, `"+8"` ; `stof("  2.5e2x")` vaut 250."""
    assert (stoi("  42abc"), stoi("-17"), stoi("+8")) == (42, -17, 8)
    assert stof("  2.5e2x") == 250.0
    assert stof(".5") == 0.5


def test_une_chaine_sans_nombre_en_tete_leve_invalid_argument() -> None:
    """Planche 76 : `stoi("Ceci est un 9")` -- `error: 'invalid_argument'`."""
    with pytest.raises(ValueError, match="invalid_argument"):
        stoi("Ceci est un 9")
    with pytest.raises(ValueError, match="invalid_argument"):
        stoi("")
    with pytest.raises(ValueError, match="invalid_argument"):
        stof("x1.5")


def test_un_entier_trop_grand_pour_un_int_leve_out_of_range() -> None:
    with pytest.raises(OverflowError, match="out_of_range"):
        stoi("99999999999")


def test_le_classement_de_la_planche_77() -> None:
    """Par code ASCII : les majuscules avant les minuscules, `NOMbre` avant `Nom1`."""
    noms = ("Nom1", "Nom2", "NOMbre", "NOMzr")
    assert tri_a_bulles(noms).valeurs == ("NOMbre", "NOMzr", "Nom1", "Nom2")
    assert ord("O") < ord("o")
    derniere_majuscule, premiere_minuscule = "Z", "a"
    assert derniere_majuscule < premiere_minuscule


def test_cin_s_arrete_au_premier_blanc_et_getline_lit_la_ligne() -> None:
    """Planche 78 : `getline` rend « Van Cool », `cin >>` ne rend que « De »."""
    assert Clavier("Van Cool\n").ligne() == "Van Cool"
    assert Clavier("De Vlam\n").mot() == "De"


def test_sans_cin_ignore_le_premier_getline_lit_une_ligne_vide() -> None:
    """Planche 79 : apres `cin >> N`, le saut de ligne reste dans le tampon."""
    sans = Clavier("5\nVan Cool\nDe Vlam\n")
    assert sans.mot() == "5"
    assert (sans.ligne(), sans.ligne()) == ("", "Van Cool")
    avec = Clavier("5\nVan Cool\nDe Vlam\n")
    assert avec.mot() == "5"
    avec.ignorer()
    assert (avec.ligne(), avec.ligne()) == ("Van Cool", "De Vlam")


def test_un_cin_ignore_de_trop_mange_la_premiere_lettre() -> None:
    """Planche 80 : les `cin.ignore()` barres font lire `rof2`, `rof3`, `rof4`."""
    clavier = Clavier("5\nProf1\nProf2\nProf3\nProf4\n")
    clavier.mot()
    clavier.ignorer()
    noms = [clavier.ligne()]
    for _ in range(3):
        clavier.ignorer()
        noms.append(clavier.ligne())
    assert noms == ["Prof1", "rof2", "rof3", "rof4"]


def test_plusieurs_valeurs_se_lisent_a_la_suite() -> None:
    """Planche 15 : le separateur est l'espace ou le retour a la ligne."""
    clavier = Clavier("12 3.5 abc\nligne deux mots\n\nfin")
    assert not clavier.epuise()
    assert (clavier.mot(), clavier.mot(), clavier.mot()) == ("12", "3.5", "abc")
    assert (clavier.ligne(), clavier.ligne(), clavier.ligne()) == (
        "",
        "ligne deux mots",
        "",
    )
    assert clavier.mot() == "fin"
    assert clavier.epuise()


def test_ignorer_jusqu_a_un_caractere_et_lire_jusqu_a_un_delimiteur() -> None:
    """Planche 142 : `cin.ignore(100, ';')`, et `getline(flux, nom, ';')`."""
    clavier = Clavier("abcdef;ghi\nxyz")
    clavier.ignorer(100, ";")
    assert clavier.ligne() == "ghi"
    champs = Clavier("un;deux;trois\n")
    assert (champs.ligne(";"), champs.ligne(";"), champs.ligne()) == (
        "un",
        "deux",
        "trois",
    )


def test_lire_au_dela_de_la_saisie_est_refuse() -> None:
    clavier = Clavier("seul")
    assert clavier.mot() == "seul"
    with pytest.raises(EOFError, match="plus rien a lire"):
        clavier.mot()
    with pytest.raises(EOFError, match="plus rien a lire"):
        clavier.ligne()
