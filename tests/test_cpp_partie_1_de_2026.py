"""Programmation, partie 1 de 2026-2027 : ce que ce jeu ajoute.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2026_2027_Partie_1.pdf`,
179 planches. Le jeu reprend la partie 1 de 2023-2024 et les seances 2 et
3 de 2025-2026, deja portees planche par planche ; il a ete relu en
entier. Ne figurent ici que les exemples qu'il ajoute :

* planches 69 et 70 : la boucle `for (auto elem : tab)` travaille sur une
  copie, `for (auto& elem : tab)` sur l'element lui-meme ;
* planche 24 : `auto [A, B, C] = myTab;` copie les trois cases ;
* planches 76 et 77 (et 16 de la seance 2 de 2025-2026) : les predicats
  de `<cctype>` sur un caractere, et le mot de passe masque.

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17).
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.caracteres import (
    isalnum,
    isalpha,
    isblank,
    isdigit,
    islower,
    ispunct,
    isspace,
    isupper,
    isxdigit,
    masquer,
    toupper,
)
from polytech.algorithmique.domain.tableaux import boucle_sur_les_elements

PREDICATS = (
    isalpha,
    islower,
    isupper,
    isdigit,
    isxdigit,
    isalnum,
    isspace,
    isblank,
    ispunct,
)
"""L'ordre dans lequel le programme compile les a affiches."""

COMPILE = {
    "a": "110011000",
    "Z": "101001000",
    "5": "000111000",
    "f": "110011000",
    " ": "000000110",
    "\t": "000000110",
    "\n": "000000100",
    "_": "000000001",
    "G": "101001000",
    "@": "000000001",
}


@pytest.mark.parametrize(("caractere", "attendu"), list(COMPILE.items()))
def test_les_neuf_predicats_sur_un_caractere(caractere: str, attendu: str) -> None:
    """Planche 76 : lettre, minuscule, majuscule, chiffre, hexadecimal, ..."""
    rendus = "".join(str(int(predicat(caractere))) for predicat in PREDICATS)
    assert rendus == attendu


def test_une_lettre_accentuee_n_est_pas_une_lettre_pour_cctype() -> None:
    """Hors de l'ASCII, `isalpha` rend faux -- la ou Python dirait vrai."""
    assert "é".isalpha()
    assert not isalpha("é")
    assert not isdigit("²")
    assert toupper("é") == "é"


def test_un_predicat_ne_recoit_qu_un_caractere() -> None:
    """`isalpha("ab")` n'a pas de sens : `c` est un `char`."""
    with pytest.raises(ValueError, match="un seul caractere attendu"):
        isalpha("ab")
    with pytest.raises(ValueError, match="un seul caractere attendu"):
        toupper("")


def test_toupper_ne_touche_que_les_minuscules() -> None:
    """Planche 77 : `ch = toupper(ch)`."""
    assert [toupper(c) for c in "aZ5@"] == ["A", "Z", "5", "@"]


def test_le_mot_de_passe_masque_de_la_planche_77() -> None:
    """`R35 a7600@beL#y 54t` devient `R** A*****BEL*Y **T`."""
    assert masquer("R35 a7600@beL#y 54t") == "R** A*****BEL*Y **T"
    assert masquer("a\tb") == "A B"


def test_la_boucle_par_copie_de_la_planche_69() -> None:
    """`for (auto elem : myTab) { ...; elem = 5 * elem; }` : le tableau reste 1 2 3."""
    tableau = [1.0, 2.0, 3.0]
    vus = boucle_sur_les_elements(tableau, lambda e: 5 * e, par_reference=False)
    assert vus == (1.0, 2.0, 3.0)
    assert tableau == [1.0, 2.0, 3.0]


def test_la_boucle_par_reference_de_la_planche_70() -> None:
    """`for (auto& elem : myTab)` : affiche 1 2 3, et le tableau devient 5 10 15."""
    tableau = [1.0, 2.0, 3.0]
    vus = boucle_sur_les_elements(tableau, lambda e: 5 * e, par_reference=True)
    assert vus == (1.0, 2.0, 3.0)
    assert tableau == [5.0, 10.0, 15.0]


def test_la_liaison_structuree_de_la_planche_24_copie_les_cases() -> None:
    """`auto [A, B, C] = myTab; A = 0;` laisse `myTab[0]` intact."""
    tableau = [5.0, 10.0, 15.0]
    a, b, c = boucle_sur_les_elements(tableau, lambda _: 0.0, par_reference=False)
    a = 0.0
    assert (a, b, c) == (0.0, 10.0, 15.0)
    assert tableau[0] == 5.0
