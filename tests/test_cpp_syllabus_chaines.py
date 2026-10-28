"""Syllabus « Introduction a la programmation en C++ » : chaines et caracteres.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf`, pages 31 a 33 (lecture
caractere par caractere), 42 a 45 (4.2, les chaines) et 74 (exercice 6.5).

Chaque affichage a ete confirme en compilant le programme (clang++ 21).
La compilation etablit deux ecarts :

* page 45 : `find` rendrait « une valeur quelconque » quand le motif est
  absent ; elle rend `string::npos`, la plus grande valeur de `size_t` ;
* page 74, exercice 6.5 : la phrase est lue par `cin >> Tab`, qui s'arrete
  au premier blanc. Le programme ne chiffrerait que « Les » ; l'exemple
  donne suppose une lecture de la ligne entiere.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.caracteres import decaler_les_lettres
from polytech.algorithmique.domain.chaines import (
    NPOS,
    find,
    insert,
    precede,
    sous_chaine,
)
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.iterateurs import plage

PHRASE = "abracadabra magie\n"
PLUS_GRANDE_VALEUR_DE_SIZE_T = 18446744073709551615
"""Page 45 : ce que `cout << s.find(...)` affiche pour un motif absent
(2^64 - 1, rejoue par le compilateur dans `test_cpp_preuves_par_le_compilateur.py`)."""


def test_la_recherche_du_programme_4_4() -> None:
    """Page 45 : « Position de la premiere occurrence de DEF dans ST12 = 3 »."""
    assert find("ABCDEF", "DEF") == 3
    assert find("ABCDEF", "A") == 0


def test_find_rend_npos_quand_le_motif_est_absent() -> None:
    """Ecart de la page 45 : non « une valeur quelconque », mais `string::npos`."""
    assert find("ABCDEF", "XYZ") == NPOS == PLUS_GRANDE_VALEUR_DE_SIZE_T


def test_les_affectations_partielles_du_programme_4_4() -> None:
    """`ST3.assign(&ST12[1], &ST12[5])` rend BCDE ; `ST4.assign(ST3, 0, 2)`, BC."""
    st12 = "ABCDEF PLUS"
    st3 = "".join(plage(st12, 1, 5))
    assert st3 == "BCDE"
    assert sous_chaine(st3, 0, 2) == "BC"


def test_les_insertions_du_programme_4_4() -> None:
    """BBBBAA, puis BbcdBBBAA, puis BbcdBCCDDBFFBAA, puis ..._FIN."""
    st12 = 4 * "B" + "AA"
    assert st12 == "BBBBAA"
    st12 = insert(st12, 1, "bcd")
    assert st12 == "BbcdBBBAA"
    st12 = insert(insert(insert(st12, 5, "CC"), 7, "DD"), 10, "FF")
    assert st12 == "BbcdBCCDDBFFBAA"
    assert st12 + "_FIN" == "BbcdBCCDDBFFBAA_FIN"


def test_insert_accepte_la_fin_de_la_chaine_et_pas_au_dela() -> None:
    """Inserer en `size()` ajoute a la fin ; plus loin, `out_of_range`."""
    assert insert("abc", 3, "d") == "abcd"
    assert insert("abc", 0, "z") == "zabc"
    for position in (4, -1):
        with pytest.raises(IndexError, match="position hors de la chaine"):
            insert("abc", position, "d")


@pytest.mark.parametrize(
    ("avant", "apres"),
    [
        ("ABC", "BCA"),
        ("ABC", "B"),
        ("Abc", "abc"),
        ("ab", "abcdZ"),
        (" ab", "ab"),
        ("", "a"),
    ],
)
def test_la_precedence_lexicographique_de_l_exemple_8(avant: str, apres: str) -> None:
    """Page 43 : `"ABC" < "BCA"`, `"Abc" < "abc"`, `"ab" < "abcdZ"`, `" ab" < "ab"`."""
    assert precede(avant, apres)
    assert not precede(apres, avant)
    assert (avant < apres) is True


def test_une_chaine_ne_se_precede_pas_elle_meme() -> None:
    """La relation est stricte : ni `"abc" < "abc"`, ni `"" < ""`."""
    assert not precede("abc", "abc")
    assert not precede("", "")


def test_le_plus_grand_des_deux_strings_du_programme_4_4() -> None:
    """Page 45 : « Le plus grand est : DEF »."""
    assert precede("ABC", "DEF")


def test_zero_precede_z_par_son_code_ascii() -> None:
    """Page 43 : `'0' < 'Z'` car 48 < 90."""
    assert (ord("0"), ord("Z")) == (48, 90)
    assert precede("0", "Z")


def test_cin_get_lit_les_blancs_que_l_extraction_saute() -> None:
    """Page 30 : `cin.get()` saisit un caractere, quel qu'il soit."""
    clavier = Clavier("a b\n")
    assert [clavier.caractere() for _ in range(4)] == ["a", " ", "b", "\n"]
    with pytest.raises(EOFError, match="plus rien a lire"):
        clavier.caractere()


def _compter(*, tester_avant: bool) -> tuple[int, int, int]:
    clavier = Clavier(PHRASE)
    a = b = total = 0
    if tester_avant:
        while (c := clavier.caractere()) != "\n":
            a, b, total = a + (c == "a"), b + (c == "b"), total + 1
        return a, b, total
    c = ""
    while c != "\n":
        c = clavier.caractere()
        a, b, total = a + (c == "a"), b + (c == "b"), total + 1
    return a, b, total


def test_le_programme_3_6_compte_dix_sept_caracteres() -> None:
    """Page 31, boucle `while` : « NOMBRE DE a: 6, de b :2 et le total = 17 »."""
    assert _compter(tester_avant=True) == (6, 2, 17)


def test_le_programme_3_8_en_compte_un_de_plus() -> None:
    """Page 33, boucle `do ... while` : 18, le retour a la ligne compris."""
    assert _compter(tester_avant=False) == (6, 2, 18)


def test_le_chiffrement_de_l_exercice_6_5() -> None:
    """Page 74 : chaque lettre decalee de 5 -- « a » devient « f »."""
    clair = "Les sanglots longs des violons de l'automne"
    assert (
        decaler_les_lettres(clair, 5) == "Qjx xfslqtyx qtslx ijx antqtsx ij q'fzytrsj"
    )
    assert decaler_les_lettres("ab", 5) == "fg"


def test_le_decalage_boucle_sur_l_alphabet_et_se_defait() -> None:
    """`z` decale de 5 revient en `e` ; le decalage oppose rend le texte clair."""
    assert decaler_les_lettres("xyz XYZ", 5) == "cde CDE"
    clair = "Les sanglots longs, 1866 !"
    assert decaler_les_lettres(decaler_les_lettres(clair, 5), -5) == clair


def test_cin_ne_lirait_que_le_premier_mot_de_la_phrase() -> None:
    """Ecart de l'exercice 6.5 : `cin >> Tab` s'arrete au premier blanc."""
    assert Clavier("Les sanglots longs\n").mot() == "Les"
