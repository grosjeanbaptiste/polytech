"""Syllabus d'algebre, chapitre VI : le rang d'une matrice.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 64 a 70. Les
exemples du calcul d'un rang, du lien avec les systemes lineaires et du
lien avec les determinants, puis les « Solutions du 6eme chapitre ».

Une coquille est etablie : page 65, la matrice echelonnee de l'exemple 2
porte `2 a a` en premiere ligne ; le chapitre III, d'ou elle vient, ecrit
`2 a 2`. Le rang n'en depend pas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.algebre.rang import rang
from polytech.maths.domain.algebre.rang_exact import (
    rang_par_echelonnement,
    rang_par_les_mineurs,
    rangs_selon_le_parametre,
)
from polytech.maths.domain.algebre.solutions import (
    PARAMETRES,
    Nature,
    cramer_generalise,
    nature_des_solutions,
    nombre_de_parametres,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

M = sp.Matrix
R = sp.Rational
A, T = sp.symbols("a t")
ALPHA, BETA = PARAMETRES[:2]
EXEMPLE_1 = M([[1, 2, -1, 3], [2, 4, 1, -2], [3, 6, 3, -7]])
PARAMETRIQUE = M([[A - 2, 2, -1], [2, A, 2], [2 * A, 2 * (A + 1), A + 1]])
QUATRE = M([[1, -1, 2, -1], [2, 1, -2, -2], [-1, 2, -4, 1], [4, -1, 2, -4]])
SECOND = M([-1, -2, 1, -4])


def test_le_rang_de_l_exemple_1_par_echelonnement() -> None:
    """`l2 - 2 l1`, `l3 - 3 l1`, `l3 - 2 l2` : deux lignes non nulles."""
    resultat = echelonner(EXEMPLE_1)
    assert resultat.operations == (
        "l2 -> l2 - 2 l1",
        "l3 -> l3 - 3 l1",
        "l3 -> l3 - 2 l2",
    )
    assert resultat.matrice == M([[1, 2, -1, 3], [0, 0, 3, -8], [0, 0, 0, 0]])
    assert rang_par_echelonnement(EXEMPLE_1) == 2


def test_le_rang_concorde_avec_le_calcul_flottant_existant() -> None:
    """Lien : `rang.rang` mene l'elimination en flottants, a pivot partiel."""
    lignes = tuple(tuple(float(x) for x in EXEMPLE_1.row(i)) for i in range(3))
    assert rang(lignes) == rang_par_echelonnement(EXEMPLE_1) == EXEMPLE_1.rank()


def test_le_rang_de_l_exemple_2_selon_le_parametre() -> None:
    """Trois hors de 0, 1 et 2 ; deux en chacune de ces valeurs."""
    generique, particuliers = rangs_selon_le_parametre(PARAMETRIQUE, A)
    assert generique == 3
    assert particuliers == {0: 2, 1: 2, 2: 2}


def test_la_coquille_de_la_page_65_ne_change_pas_le_rang() -> None:
    """`2 a a` imprime pour `2 a 2` : les deux matrices ont les memes rangs."""
    derniere = [0, 0, -A * (A - 1) * (A - 2)]
    imprimee = M([[2, A, A], [0, 2, -(A - 1)], derniere])
    correcte = M([[2, A, 2], [0, 2, -(A - 1)], derniere])
    assert imprimee != correcte
    for matrice in (imprimee, correcte):
        assert rangs_selon_le_parametre(matrice, A) == (3, {0: 2, 1: 2, 2: 2})


def test_le_rang_donne_l_ordre_d_indetermination_avant_de_resoudre() -> None:
    """VI.2, exemples 1 et 2 : `3 - 2 = 1`, puis `4 - 2 = 2`."""
    premier = SystemeLineaire(M([[3, -2, 7], [-6, 4, 3]]), M([1, 2]))
    complete = premier.matrice.row_join(premier.second_membre)
    assert rang_par_echelonnement(premier.matrice) == 2
    assert rang_par_echelonnement(complete) == 2
    assert nombre_de_parametres(premier) == 1
    second = SystemeLineaire(QUATRE, SECOND)
    assert rang_par_echelonnement(QUATRE) == 2
    assert rang_par_echelonnement(QUATRE.row_join(SECOND)) == 2
    assert nombre_de_parametres(second) == 2


def test_les_rangs_de_l_exemple_3_decident_de_la_discussion() -> None:
    """`a = 0` : `rg A = rg(A|b) = 2` ; `a = 1` ou `2` : `rg A = 2`, `rg(A|b) = 3`."""
    second = M([1, 1, 2])
    complete = PARAMETRIQUE.row_join(second)
    assert rang_par_echelonnement(PARAMETRIQUE) == 3
    assert rang_par_echelonnement(complete.subs(A, 0)) == 2
    for valeur in (1, 2):
        assert rang_par_echelonnement(PARAMETRIQUE.subs(A, valeur)) == 2
        assert rang_par_echelonnement(complete.subs(A, valeur)) == 3
        systeme = SystemeLineaire(PARAMETRIQUE.subs(A, valeur), second)
        assert nature_des_solutions(systeme) is Nature.IMPOSSIBLE
    nul = SystemeLineaire(PARAMETRIQUE.subs(A, 0), second)
    assert nature_des_solutions(nul) is Nature.INDETERMINE


def test_le_rang_est_l_ordre_de_la_plus_grande_sous_matrice_reguliere() -> None:
    """VI.3 : `rg [[3, -2, 7], [-6, 4, 3]] = 2` car `det [[3, 7], [-6, 3]] != 0`."""
    matrice = M([[3, -2, 7], [-6, 4, 3]])
    assert rang_par_les_mineurs(matrice) == 2
    assert M([[3, 7], [-6, 3]]).det() == 51
    assert M([[3, -2], [-6, 4]]).det() == 0


def test_les_mineurs_d_ordre_trois_de_la_page_67_sont_tous_nuls() -> None:
    """Rang 2 : aucune sous-matrice d'ordre 3 reguliere, au moins une d'ordre 2."""
    assert rang_par_les_mineurs(QUATRE) == 2
    assert rang_par_echelonnement(QUATRE) == 2
    assert all(
        QUATRE.minor_submatrix(i, j).det() == 0 for i in range(4) for j in range(4)
    )
    assert M([[1, -1], [2, 1]]).det() == 3


def test_le_rang_d_une_matrice_nulle_est_zero() -> None:
    """Aucune sous-matrice reguliere, aucune ligne non nulle."""
    assert rang_par_les_mineurs(sp.zeros(2, 3)) == 0
    assert rang_par_echelonnement(sp.zeros(2, 3)) == 0


def test_la_resolution_par_la_sous_matrice_de_la_page_67() -> None:
    """`[[1, -1], [2, 1]]`, d'inverse `1/3 [[1, 1], [-2, 1]]`, puis la solution."""
    reduction = cramer_generalise(SystemeLineaire(QUATRE, SECOND))
    assert (reduction.lignes, reduction.colonnes) == ((0, 1), (0, 1))
    assert reduction.sous_matrice == M([[1, -1], [2, 1]])
    assert reduction.sous_matrice.inv() == M([[1, 1], [-2, 1]]) / 3
    assert reduction.solution == (-1 + BETA, 2 * ALPHA, ALPHA, BETA)


def test_les_reponses_a_et_b() -> None:
    """a) 2 ; b) 3 -- par les deux chemins."""
    a = M(
        [
            [-1, 2, 0, 4, 5],
            [3, -7, 2, 0, 1],
            [2, -5, 2, 4, 6],
            [4, -9, 2, -4, -4],
        ]
    )
    b = M([[1, -3, 2, 2], [0, 3, 6, 0], [2, -3, -2, 4], [3, -6, 0, 6]])
    assert rang_par_echelonnement(a) == rang_par_les_mineurs(a) == 2
    assert rang_par_echelonnement(b) == rang_par_les_mineurs(b) == 3


@pytest.mark.parametrize(
    ("matrice", "parametre", "particuliers"),
    [
        ([[1, 2 * A, 1], [A, 2, 1], [1, 2, A]], A, {1: 1, -2: 2}),
        ([[1, T, 1], [T, 1, T], [1, T, -T]], T, {-1: 1, 1: 2}),
        ([[T, 3, -1], [3, 6, -2], [-1, -3, T]], T, {1: 2, R(3, 2): 2}),
    ],
    ids=["c", "d", "e"],
)
def test_les_reponses_c_d_e(
    matrice: list[list[sp.Expr]], parametre: sp.Symbol, particuliers: dict[sp.Expr, int]
) -> None:
    """Rang 3 en general, et le rang officiel en chaque valeur particuliere."""
    assert rangs_selon_le_parametre(M(matrice), parametre) == (3, particuliers)


def test_une_matrice_sans_parametre_n_a_pas_de_valeur_particuliere() -> None:
    """Le rang generique est le rang, et la table est vide."""
    assert rangs_selon_le_parametre(EXEMPLE_1, A) == (2, {})


def test_le_rang_ne_chute_pas_en_une_valeur_complexe() -> None:
    """`[[a, 1], [-1, a]]` : `det = a^2 + 1` ne s'annule pas pour `a` reel."""
    assert rangs_selon_le_parametre(M([[A, 1], [-1, A]]), A) == (2, {})


def test_le_rang_ne_chute_que_si_tous_les_mineurs_s_annulent() -> None:
    """Un mineur nul en `a = 0` ou `1` ne suffit pas : un autre vaut 1 partout."""
    assert rangs_selon_le_parametre(M([[A, 1, 0], [0, A - 1, 1]]), A) == (2, {})
    assert rangs_selon_le_parametre(M([[1, 0, A], [0, 1, A - 1]]), A) == (2, {})


def test_le_rang_chute_ou_tous_les_mineurs_s_annulent_ensemble() -> None:
    """`[[a, a^2, 0], [1, a, a]]` : mineurs `0`, `a^2`, `a^3`, rang 1 en `a = 0`."""
    assert rangs_selon_le_parametre(M([[A, A**2, 0], [1, A, A]]), A) == (2, {0: 1})
