"""Programmation et algorithmique, partie 1 : tableaux, tris et recherches.

Source : `ProgrammationAlgorithmique/SLIDES/Slides_2023_2024_Partie_1.pdf`,
planches 54 a 70. Declaration et acces, suppression et ajout dans un
tableau statique, tri a bulles, tri par selection, tableau des
complexites, recherche lineaire, recherche dichotomique, exercices 4.4 et
4.5.

Chaque resultat a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit un ecart : planche 62, la
boucle d'insertion `Tab[i+1] = Tab[i]` pour `i` de `N` a `P+1` ouvre la
place en `P+1` et non en `P`, et recopie une case de trop, au-dela du
dernier element.
"""

from __future__ import annotations

import math

import pytest

from polytech.algorithmique.domain.planches_fautives import insertion_de_la_planche_62
from polytech.algorithmique.domain.recherches import (
    recherche_dichotomique,
    recherche_lineaire,
)
from polytech.algorithmique.domain.tableaux import TableauStatique, indice_valide
from polytech.algorithmique.domain.tris import (
    COMPLEXITES,
    tri_a_bulles,
    tri_par_insertion,
    tri_par_selection,
)
from polytech.algorithmique.domain.tris_efficaces import (
    tri_de_shell,
    tri_fusion,
    tri_par_tas,
    tri_rapide,
)
from polytech.maths.domain.algebre.matrices import (
    addition,
    multiplication_par_un_scalaire,
)

PLANCHE = (1, 9, 5, 2, 3, 7)
DESORDRE = (5, 1, 4, 2, 8, 0)
QUADRATIQUES = (tri_a_bulles, tri_par_selection, tri_par_insertion)
EFFICACES = (tri_de_shell, tri_fusion, tri_rapide, tri_par_tas)


def _melange(taille: int) -> tuple[int, ...]:
    """Une suite fixe et sans ordre : les multiples de 37, modulo 101."""
    return tuple((37 * rang + 11) % 101 for rang in range(taille))


def _tableau(valeurs: tuple[int, ...], capacite: int = 11) -> TableauStatique:
    return TableauStatique(capacite, list(valeurs))


def test_les_indices_vont_de_zero_a_dim_moins_un() -> None:
    """Planche 56 : `a[6]` et `a[-1]` sortent d'un tableau de 6, sans avertissement."""
    assert all(indice_valide(i, 6) for i in range(6))
    assert not indice_valide(6, 6)
    assert not indice_valide(-1, 6)
    assert indice_valide(7 // 2, 6)


def test_le_tableau_de_la_planche_58() -> None:
    """`tab[i] = 2*i` pour `i` de 0 a 4 : 0, 2, 4, 6, 8."""
    tableau = TableauStatique(5, [])
    for i in range(5):
        tableau.ajouter_a_la_fin(2 * i)
    assert tableau.elements == [0, 2, 4, 6, 8]
    assert tableau.taille == tableau.capacite == 5


def test_supprimer_au_debut_decale_tout_vers_la_gauche() -> None:
    """Planche 60 : `1 9 5 2 3 7` devient `9 5 2 3 7`, et `N` diminue."""
    tableau = _tableau(PLANCHE)
    assert tableau.supprimer(0) == 1
    assert tableau.elements == [9, 5, 2, 3, 7]
    assert tableau.taille == 5


def test_supprimer_a_la_fin_et_au_milieu() -> None:
    """Planche 61."""
    fin = _tableau(PLANCHE)
    assert fin.supprimer(fin.taille - 1) == 7
    assert fin.elements == [1, 9, 5, 2, 3]
    milieu = _tableau(PLANCHE)
    assert milieu.supprimer(milieu.taille // 2) == 2
    assert milieu.elements == [1, 9, 5, 3, 7]


def test_ajouter_a_la_fin_au_debut_et_a_une_position() -> None:
    """Planche 62 : `N` augmente, et les suivants se decalent vers la droite."""
    tableau = _tableau(PLANCHE)
    tableau.ajouter_a_la_fin(4)
    tableau.inserer(0, 8)
    tableau.inserer(2, 6)
    assert tableau.elements == [8, 1, 6, 9, 5, 2, 3, 7, 4]
    assert tableau.taille == 9


def test_la_boucle_de_la_planche_62_recopie_une_case_de_trop() -> None:
    """`for (i = N; i > P; i--) Tab[i+1] = Tab[i];` avec `N = 6`, `P = 2`."""
    memoire = [*PLANCHE, -1, -1, -1, -1, -1]
    n, p = 6, 2
    memoire = insertion_de_la_planche_62(memoire, n, p)
    assert memoire[:8] == [1, 9, 5, 2, 2, 3, 7, -1]
    correcte = _tableau(PLANCHE)
    correcte.inserer(p, 0)
    assert correcte.elements == [1, 9, 0, 5, 2, 3, 7]
    assert memoire[p] == PLANCHE[p]
    assert memoire[n + 1] == -1


def test_un_tableau_statique_plein_ne_recoit_plus_rien() -> None:
    plein = TableauStatique(3, [1, 2, 3])
    with pytest.raises(OverflowError, match="tableau plein"):
        plein.ajouter_a_la_fin(4)
    with pytest.raises(OverflowError, match="tableau plein"):
        plein.inserer(0, 4)
    with pytest.raises(OverflowError, match="plus d'elements que de places"):
        TableauStatique(2, [1, 2, 3])


def test_une_position_hors_du_tableau_est_refusee() -> None:
    tableau = _tableau(PLANCHE)
    with pytest.raises(IndexError, match="position hors du tableau"):
        tableau.supprimer(6)
    with pytest.raises(IndexError, match="position hors du tableau"):
        tableau.inserer(7, 0)
    with pytest.raises(IndexError, match="position hors du tableau"):
        TableauStatique(4, []).supprimer(0)


def test_l_exercice_4_4_supprime_les_deux_bords() -> None:
    """`0 1 2 3 4 5 6` devient `1 2 3 4 5`, puis `2 3 4`."""
    tableau = _tableau(tuple(range(7)))
    for attendu in ([1, 2, 3, 4, 5], [2, 3, 4]):
        tableau.supprimer(0)
        tableau.supprimer(tableau.taille - 1)
        assert tableau.elements == attendu


def test_l_exercice_4_5_supprime_le_milieu_et_ajoute_son_double() -> None:
    """`1 2 3 4 5` devient `6 1 2 4 5`, puis `4 6 1 4 5`."""
    tableau = _tableau((1, 2, 3, 4, 5))
    for attendu in ([6, 1, 2, 4, 5], [4, 6, 1, 4, 5]):
        milieu = tableau.supprimer(tableau.taille // 2)
        tableau.inserer(0, 2 * milieu)
        assert tableau.elements == attendu


def test_le_tri_a_bulles_de_la_planche_65() -> None:
    """Sur `5 1 4 2 8 0` : six passes, trente comparaisons, neuf permutations."""
    resultat = tri_a_bulles(DESORDRE)
    assert resultat.valeurs == (0, 1, 2, 4, 5, 8)
    assert (resultat.passes, resultat.comparaisons, resultat.permutations) == (6, 30, 9)


def test_le_tri_par_selection_de_la_planche_67() -> None:
    """Sur le meme tableau : quinze comparaisons, trois permutations seulement."""
    resultat = tri_par_selection(DESORDRE)
    assert resultat.valeurs == (0, 1, 2, 4, 5, 8)
    assert (resultat.comparaisons, resultat.permutations) == (15, 3)


@pytest.mark.parametrize("tri", [*QUADRATIQUES, *EFFICACES])
def test_chaque_algorithme_de_la_planche_64_trie(tri: object) -> None:
    """Les sept tris rendent les memes valeurs, en ordre croissant."""
    for taille in (0, 1, 2, 7, 50):
        valeurs = _melange(taille)
        assert tri(valeurs).valeurs == tuple(sorted(valeurs))  # type: ignore[operator]
    assert tri(DESORDRE).valeurs == (0, 1, 2, 4, 5, 8)  # type: ignore[operator]


@pytest.mark.parametrize("tri", [*QUADRATIQUES, *EFFICACES])
def test_des_valeurs_egales_restent_toutes_la(tri: object) -> None:
    """Un tableau a doublons garde chacun de ses elements."""
    doublons = (3, 1, 3, 2, 1, 3)
    assert tri(doublons).valeurs == (1, 1, 2, 3, 3, 3)  # type: ignore[operator]


def test_le_tri_par_insertion_ne_deplace_pas_des_elements_egaux() -> None:
    """Trois valeurs egales : deux comparaisons, aucune permutation."""
    egaux = tri_par_insertion((3, 3, 3))
    assert (egaux.comparaisons, egaux.permutations) == (2, 0)


def test_le_tri_fusion_compte_les_comparaisons_de_chaque_moitie() -> None:
    """`4 3 2 1` : une par moitie, puis deux pour fusionner `3 4` et `1 2`."""
    assert tri_fusion((2, 1)).comparaisons == 1
    assert tri_fusion((4, 3, 2, 1)).comparaisons == 4


def test_un_tri_ne_modifie_pas_ce_qu_on_lui_donne() -> None:
    """Le resultat est une nouvelle suite : l'entree reste telle quelle."""
    entree = list(DESORDRE)
    tri_a_bulles(entree)
    assert entree == list(DESORDRE)


def test_le_tri_par_selection_fait_toujours_les_memes_comparaisons() -> None:
    """`N (N - 1) / 2`, que le tableau soit trie, inverse ou quelconque."""
    for taille in (2, 10, 40):
        attendues = taille * (taille - 1) // 2
        for valeurs in (range(taille), range(taille, 0, -1)):
            assert tri_par_selection(tuple(valeurs)).comparaisons == attendues


def test_le_tri_par_insertion_profite_d_un_tableau_deja_trie() -> None:
    """`N - 1` comparaisons au mieux, `N (N - 1) / 2` au pire."""
    taille = 30
    assert tri_par_insertion(tuple(range(taille))).comparaisons == taille - 1
    inverse = tuple(range(taille, 0, -1))
    assert tri_par_insertion(inverse).comparaisons == taille * (taille - 1) // 2


@pytest.mark.parametrize("tri", QUADRATIQUES)
def test_doubler_n_quadruple_le_travail_d_un_tri_en_n_carre(tri: object) -> None:
    """Planche 64, `O(N^2)` : au pire, deux fois plus d'elements, quatre fois plus."""
    petit = tri(tuple(range(200, 0, -1))).comparaisons  # type: ignore[operator]
    grand = tri(tuple(range(400, 0, -1))).comparaisons  # type: ignore[operator]
    assert grand / petit == pytest.approx(4, rel=0.03)


@pytest.mark.parametrize("tri", [tri_fusion, tri_par_tas])
def test_un_tri_en_n_log_n_reste_sous_sa_borne(tri: object) -> None:
    """Planche 64, `O(N log N)` : meme au pire, bien moins que `N^2`."""
    for taille in (64, 512):
        inverse = tuple(range(taille, 0, -1))
        comparaisons = tri(inverse).comparaisons  # type: ignore[operator]
        assert comparaisons <= 2 * taille * math.ceil(math.log2(taille))
        assert comparaisons < taille * (taille - 1) // 2


def test_le_tri_rapide_n_est_en_n_log_n_qu_en_moyenne() -> None:
    """Sur un tableau deja trie, pivot en tete, il retombe en `N^2`."""
    taille = 200
    moyen = tri_rapide(_melange(taille)).comparaisons
    pire = tri_rapide(tuple(range(taille))).comparaisons
    assert moyen < 3 * taille * math.log2(taille)
    assert pire == taille * (taille - 1) // 2


def test_le_tri_de_shell_ameliore_le_tri_par_insertion() -> None:
    """Planche 64 : sur un tableau inverse, dix fois moins de comparaisons au moins."""
    inverse = tuple(range(500, 0, -1))
    assert (
        tri_de_shell(inverse).comparaisons
        < tri_par_insertion(inverse).comparaisons / 10
    )


def test_le_tableau_des_complexites_de_la_planche_64() -> None:
    """Sept algorithmes, trois en `N^2`, un en `N log^2 N`, trois en `N log N`."""
    assert COMPLEXITES == {
        "tri a bulles": "O(N^2)",
        "tri par selection": "O(N^2)",
        "tri par insertion": "O(N^2)",
        "tri de Shell": "O(N log^2 N)",
        "tri fusion": "O(N log N)",
        "tri rapide": "O(N log N)",
        "tri par tas": "O(N log N)",
    }


def test_la_recherche_lineaire_de_la_planche_68() -> None:
    """Dans `9 3 6 2 1 12 0` : 2 est en 3, 0 en 6, et 5 est absent (`i` finit a `N`)."""
    tableau = (9, 3, 6, 2, 1, 12, 0)
    assert recherche_lineaire(tableau, 2) == 3
    assert recherche_lineaire(tableau, 0) == 6
    assert recherche_lineaire(tableau, 5) == len(tableau)


def test_la_recherche_dichotomique_de_la_planche_69() -> None:
    """Chercher 6 dans `2 4 6 8 10 12 14` : milieux 3, puis 1, puis 2."""
    tableau = (2, 4, 6, 8, 10, 12, 14)
    trouve = recherche_dichotomique(tableau, 6)
    assert trouve.milieux == (3, 1, 2)
    assert trouve.position == 2
    assert recherche_dichotomique(tableau, 14).milieux == (3, 5, 6)
    assert recherche_dichotomique(tableau, 2).milieux == (3, 1, 0)
    absent = recherche_dichotomique(tableau, 7)
    assert absent.milieux == (3, 1, 2)
    assert absent.position is None


def test_la_dichotomie_divise_l_intervalle_par_deux() -> None:
    """Au plus `log2(N) + 1` etapes, la ou la recherche lineaire en fait jusqu'a `N`."""
    tableau = tuple(range(0, 2048, 2))
    for cherche in (0, 2046, 777, 1024):
        etapes = len(recherche_dichotomique(tableau, cherche).milieux)
        assert etapes <= math.floor(math.log2(len(tableau))) + 1


def test_la_dichotomie_exige_un_tableau_trie() -> None:
    with pytest.raises(ValueError, match="tableau non trie"):
        recherche_dichotomique((3, 1, 2), 1)


def test_la_matrice_de_la_planche_81() -> None:
    """`tab[i][j] = i` sur 5 lignes et 4 colonnes : la ligne `i` est faite de `i`."""
    matrice = [[i for _ in range(4)] for i in range(5)]
    lignes = [" ".join(str(valeur) for valeur in ligne) for ligne in matrice]
    assert lignes == ["0 0 0 0", "1 1 1 1", "2 2 2 2", "3 3 3 3", "4 4 4 4"]
    assert indice_valide(4, 5)
    assert not indice_valide(4, 4)


def test_le_programme_de_la_planche_82_soustrait_deux_matrices() -> None:
    """« Que fait ce programme ? » : `c[i][j] = a[i][j] - b[i][j]`, case par case.

    Lien avec le cours de mathematique : c'est `A + (-1) B`, par l'addition
    et le produit par un scalaire du chapitre II du syllabus d'algebre.
    """
    a = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    b = ((9, 8, 7), (6, 5, 4), (3, 2, 1))
    programme = tuple(tuple(a[i][j] - b[i][j] for j in range(3)) for i in range(3))
    assert programme == ((-8, -6, -4), (-2, 0, 2), (4, 6, 8))
    assert addition(a, multiplication_par_un_scalaire(-1, b)) == programme


def test_la_boucle_de_la_planche_62_sans_place_ecrirait_hors_du_tableau() -> None:
    with pytest.raises(IndexError, match="hors de la memoire"):
        insertion_de_la_planche_62(list(PLANCHE), 6, 2)
