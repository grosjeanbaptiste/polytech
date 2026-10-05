"""Programmation objet, seances 5 et 6 : iterateurs et algorithmes de la STL.

Sources, dans `ProgrammationAlgorithmique/SLIDES/` :
`Slides_2023_2024_S5.pdf` (planches 60 a 96) et
`Slides_2025_2026_S5_S6.pdf` (planches 44 a 104).

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit trois ecarts :

* planche 60 de 2023-2024 : le parcours de `"Un message "` est annote
  `U,n, ,m,e,s,s,a,g,e,` ; la chaine finit par un blanc, et l'affichage
  reel se termine par `e, ,` ;
* planche 61 : le programme ecrit `crypte +=` pour `crypto +=`, et ne
  compile pas tel quel ;
* planche 87 de 2025-2026 : `auto it;` sans initialisation ne compile pas
  -- `auto` n'a rien dont deduire le type.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.algorithmes_stl import (
    all_of,
    copy,
    count,
    count_if,
    equal,
    find,
    max_element,
    min_element,
    remove,
    remove_if,
    replace,
    search,
    sort,
    unique,
)
from polytech.algorithmique.domain.iterateurs import (
    avancer,
    chiffrer,
    parcours_inverse,
    plage,
)

T = (32, 71, 12, 45, 26, 80, 53, 33)
TEXTE = "UN EXEMPLE D'UTILISATION d'iterateur !!"
RESTE = "-5 '0 te9 -8 te12 9  !!"
SERIES = [
    ("Dark", 1, 1, 2017),
    ("Dark", 1, 3, 2017),
    ("Dark", 1, 1, 2017),
    ("Dark", 2, 1, 2019),
    ("Dark", 3, 1, 2020),
    ("Dark", 0, 1, 2017),
    ("Dark", 1, -1, 2017),
    ("Game of Thrones", 1, 1, 2011),
    ("Game of Thrones", 1, 3, 2011),
    ("Game of Thrones", 1, 1, 2011),
    ("Game of Thrones", 8, 5, 2019),
    ("Game of Thrones", 8, 6, 2019),
    ("Stranger Things", 1, 1, 2016),
    ("Stranger Things", 2, 1, 2017),
    ("Stranger Things", 2, 2, 2017),
    ("Stranger Things", 2, 2, 2017),
    ("Stranger Things", 3, 1, 2019),
    ("Stranger Things", 4, 1, 2022),
    ("Squid Game", 1, 1, 2021),
    ("Squid Game", 1, 2, 2021),
    ("Squid Game", 2, 1, 2024),
]


def test_les_trois_plages_de_la_planche_64() -> None:
    """`V1(T, T + 4)`, `V2(T, T + 6)`, `V3(T + 2, T + 5)`."""
    assert plage(T, 0, 4) == (32, 71, 12, 45)
    assert plage(T, 0, 6) == (32, 71, 12, 45, 26, 80)
    assert plage(T, 2, 5) == (12, 45, 26)
    assert plage(T, 8, 8) == ()


@pytest.mark.parametrize(("debut", "fin"), [(5, 2), (-1, 3), (0, 9)])
def test_une_plage_reste_dans_le_tableau(debut: int, fin: int) -> None:
    """`[first, last)` : `first` avant `last`, tous deux dans le tableau."""
    with pytest.raises(IndexError, match="plage hors du tableau"):
        plage(T, debut, fin)


def test_le_parcours_inverse_de_la_planche_62() -> None:
    """`for (rp = V.rbegin(); rp != V.rend(); rp++)` sur 1 2 3 : 3 ; 2 ; 1."""
    assert parcours_inverse((1, 2, 3)) == (3, 2, 1)
    assert parcours_inverse(()) == ()


def test_advance_de_la_planche_72() -> None:
    """`advance(itL, L2.size() / 2)` depuis `begin()` : la position 5 sur 10."""
    assert avancer(0, 10 // 2, 10) == 5
    assert avancer(5, 5, 10) == 10
    with pytest.raises(IndexError, match="au-dela de end"):
        avancer(5, 6, 10)


def test_le_parcours_d_une_chaine_caractere_par_caractere() -> None:
    """Ecart de la planche 60 : le blanc final de la chaine sort lui aussi."""
    affiche = "".join(f"{c}," for c in plage("Un message ", 0, 11))
    assert affiche == "U,n, ,m,e,s,s,a,g,e, ,"


@pytest.mark.parametrize(
    ("decalage", "majuscules"),
    [
        (0, "UN EXEMPLE D'UTILISATION"),
        (1, "VO FYFNQMF E'VUJMJTBUJPO"),
        (2, "WP GZGORNG F'WVKNKUCVKQP"),
        (3, "XQ HAHPSOH G'XWLOLVDWLRQ"),
        (4, "YR IBIQTPI H'YXMPMWEXMSR"),
    ],
)
def test_les_cinq_lignes_chiffrees_de_la_planche_61(
    decalage: int, majuscules: str
) -> None:
    """Les majuscules decalees ; les minuscules en code, sauf `e` et `t`."""
    assert chiffrer(TEXTE, decalage) == f"{majuscules} {RESTE}"


def test_le_decalage_boucle_apres_z() -> None:
    """`(c + decalage - 'A') % 26 + 'A'` : Z decale de 1 donne A."""
    assert chiffrer("XYZ", 3) == "ABC"
    assert chiffrer("d", 0) == "-5 "
    assert chiffrer("e t'!", 9) == "e t'!"


def test_remove_ne_change_pas_la_taille_du_vecteur() -> None:
    """Planche 94 : `9 4 2 8 1 8 3` devient `9 4 2 1 3 8 3`, fin logique en 5."""
    v1 = [9, 4, 2, 8, 1, 8, 3]
    fin = remove(v1, 8)
    assert (fin, v1) == (5, [9, 4, 2, 1, 3, 8, 3])


def test_erase_apres_remove_retire_vraiment() -> None:
    """Planche 94 : `V2.erase(remove(...), V2.end())` -- taille 5, `9 4 2 1 3`."""
    v2 = [9, 4, 2, 8, 1, 8, 3]
    del v2[remove(v2, 8) :]
    assert v2 == [9, 4, 2, 1, 3]


def test_remove_sur_la_chaine_de_la_planche_95() -> None:
    """« Qu donnra c txt si on nlv ls '' ? les 'e' ? » puis la chaine raccourcie."""
    texte = list("Que donnera ce texte si on enleve les 'e' ? ")
    fin = remove(texte, "e")
    assert "".join(texte) == "Qu donnra c txt si on nlv ls '' ? les 'e' ? "
    assert "".join(texte[:fin]) == "Qu donnra c txt si on nlv ls '' ? "


def test_remove_d_une_valeur_absente_ne_touche_a_rien() -> None:
    """La fin logique reste la fin."""
    valeurs = [1, 2, 3]
    assert remove(valeurs, 9) == 3
    assert valeurs == [1, 2, 3]


def test_unique_ne_change_pas_non_plus_la_taille() -> None:
    """Planche 96 : `9 9 2 9 2 2 2 9 8` devient `9 2 9 2 9 8 2 9 8`."""
    v1 = [9, 9, 2, 9, 2, 2, 2, 9, 8]
    fin = unique(v1)
    assert (fin, v1) == (6, [9, 2, 9, 2, 9, 8, 2, 9, 8])
    assert unique([]) == 0


def test_remove_if_generalise_remove() -> None:
    """Planche 46 : `remove_if(predicate)` -- tout ce qui satisfait la condition."""
    valeurs = [5, 15, 25, 10, 20, 40, 7]
    fin = remove_if(valeurs, lambda v: v > 10)
    assert valeurs[:fin] == [5, 10, 7]


def test_les_trois_tris_de_la_planche_88() -> None:
    """`2 4 9 8 1 3`, puis `1 2 3 4 8 9`, puis `9 8 4 3 2 1`."""
    v = [9, 4, 2, 8, 1, 3]
    sort(v, 0, 3)
    assert v == [2, 4, 9, 8, 1, 3]
    sort(v)
    assert v == [1, 2, 3, 4, 8, 9]
    sort(v, decroissant=True)
    assert v == [9, 8, 4, 3, 2, 1]


def test_un_tri_ne_sort_pas_de_sa_plage() -> None:
    """`sort(V.begin() + 2, V.begin() + 5)` laisse les bords en place."""
    v = [9, 4, 2, 8, 1, 3]
    sort(v, 2, 5)
    assert v == [9, 4, 1, 2, 8, 3]
    with pytest.raises(IndexError, match="plage hors du tableau"):
        sort(v, 2, 7)


def test_find_rend_end_quand_la_valeur_est_absente() -> None:
    """Planche 90 : `if (It != V.end())` -- « V1 est dans V », « V5 n'est pas... »."""
    v = ["V0", "V1", "V2"]
    assert find(v, "V1") == 1
    assert find(v, "V5") == len(v)


def test_l_exercice_8_7_il_fait_tres_beau() -> None:
    """Planche 91 : (1) il fait beau (2) il fait tres beau (3) il tres beau."""
    phrase = ["il", "fait", "beau"]
    phrase.insert(find(phrase, "beau"), "tres")
    assert phrase == ["il", "fait", "tres", "beau"]
    del phrase[find(phrase, "fait")]
    assert phrase == ["il", "tres", "beau"]


def test_l_exercice_8_8_la_fonction_test() -> None:
    """Planche 92 : V1 est dans Vecteur, V5 non ; 10 n'est pas dans List, 2 si."""
    vecteur, liste = ["V0", "V1", "V2"], list(range(5))
    constats = [
        find(vecteur, "V1") != len(vecteur),
        find(vecteur, "V5") != len(vecteur),
        find(liste, 10) != len(liste),
        find(liste, 2) != len(liste),
    ]
    assert constats == [True, False, False, True]


def test_search_rend_la_position_d_une_sous_suite() -> None:
    """2025-2026, planche 96 : `search(debut1, fin1, debut2, fin2)`."""
    assert search(T, (12, 45, 26)) == 2
    assert search(T, (12, 26)) == len(T)
    assert search(T, ()) == 0
    assert search((1, 2), (1, 2, 3)) == 2


def test_count_de_la_planche_93() -> None:
    """« (V) : 3 de 4 » et « (L) : 2 de 1 »."""
    valeurs = (0, 1, 1, 3, 4, 4, 4, 9, 8, 9)
    assert (count(valeurs, 4), count(valeurs, 1)) == (3, 2)


def test_count_if_de_la_planche_94() -> None:
    """« Le nombre d'elements de v1 > 10 est: 4. »."""
    assert count_if((5, 15, 25, 10, 20, 40, 7), lambda valeur: valeur > 10) == 4


def test_equal_de_la_planche_95() -> None:
    """Pas Egalite, Pas Egalite, Egalite, puis Pas Egalite apres `push_back(7)`."""
    tab1, tab2, tab3 = (1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 3, 9)
    tab4 = list(tab1)
    constats = [equal(tab1, tab2), equal(tab1, tab3), equal(tab1, tab4)]
    tab4.append(7)
    assert [*constats, equal(tab1, tab4)] == [False, False, True, False]


def test_all_of_de_la_planche_91() -> None:
    """« OK, numero de telephone correct. » pour 32474123456."""
    assert all_of("32474123456", str.isdigit)
    assert not all_of("3247 4123456", str.isdigit)


def test_min_max_et_replace_de_la_planche_102() -> None:
    """`a b b d e d` : count de b = 2, min a, max e ; puis `a b b f e f`."""
    v = list("abbded")
    assert count(v, "b") == 2
    assert (v[min_element(v)], v[max_element(v)]) == ("a", "e")
    replace(v, "d", "f")
    assert "".join(v) == "abbfef"


def test_min_et_max_rendent_la_premiere_position() -> None:
    """Sur des egaux, c'est le premier que l'iterateur designe."""
    assert (min_element((3, 1, 2, 1)), max_element((3, 1, 3, 2))) == (1, 0)
    with pytest.raises(ValueError, match="plage vide"):
        min_element(())
    with pytest.raises(ValueError, match="plage vide"):
        max_element(())


def test_copy_de_la_planche_89() -> None:
    """`vecteur.resize(liste.size())` d'abord, puis la copie de 1 a 9."""
    liste = list(range(1, 10))
    vecteur = [0] * len(liste)
    assert copy(liste, vecteur) == 9
    assert vecteur == liste


def test_copy_exige_une_destination_assez_grande() -> None:
    """Sans le `resize`, la copie ecrit hors du vecteur."""
    with pytest.raises(IndexError, match="destination trop petite"):
        copy([1, 2, 3], [0, 0])
    destination = [0, 0, 0, 0]
    assert copy([7, 8], destination, 1) == 3
    assert destination == [0, 7, 8, 0]
    with pytest.raises(IndexError, match="destination trop petite"):
        copy([7, 8], destination, 3)


def test_l_exercice_des_copies_de_la_planche_92() -> None:
    """2025-2026 : 4 de 9, 4 de 2 ; apres « 2 par 9 », 8 de 9 et plus d'egalite."""
    l1 = [9, 9, 2, 9, 2, 2, 2, 9, 8]
    v1, v2, v3 = [0] * 9, [0] * 9, [0] * 9
    for destination in (v1, v2, v3):
        copy(l1, destination)
    assert (count(v1, 9), count(l1, 2)) == (4, 4)
    sort(v2)
    assert [equal(v1, l1), equal(v1, v2), equal(v1, v3)] == [True, False, True]
    replace(v3, 2, 9)
    assert v3 == [9, 9, 9, 9, 9, 9, 9, 9, 8]
    assert (equal(v1, v3), count(v3, 9)) == (False, 8)


def test_les_noms_tries_puis_dedoublonnes_de_la_planche_105() -> None:
    """`sort` puis `unique` : Asterix, Obelix, Tintin, Titeuf."""
    noms = ["Obelix", "Tintin", "Asterix", "Titeuf", "Titeuf", "Obelix", "Asterix"]
    sort(noms)
    del noms[unique(noms) :]
    assert noms == ["Asterix", "Obelix", "Tintin", "Titeuf"]


def test_l_historique_des_series_de_la_planche_103() -> None:
    """Sans entree invalide, trie par nom, saison, episode, sans doublon."""
    series = list(SERIES)
    del series[remove_if(series, lambda s: s[1] <= 0 or s[2] <= 0) :]
    assert len(SERIES) - len(series) == 2
    sort(series)
    del series[unique(series) :]
    assert len(series) == 16
    assert series[:4] == [
        ("Dark", 1, 1, 2017),
        ("Dark", 1, 3, 2017),
        ("Dark", 2, 1, 2019),
        ("Dark", 3, 1, 2020),
    ]
    assert [s[0] for s in series[4::4]] == [
        "Game of Thrones",
        "Squid Game",
        "Stranger Things",
    ]
    assert series == sorted(set(series))


def test_unique_sans_tri_laisse_les_doublons_eloignes() -> None:
    """Planche 104, « Pas de doublon » : le tri vient d'abord."""
    series = list(SERIES[:3])
    assert unique(series) == 3
