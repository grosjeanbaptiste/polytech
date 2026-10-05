"""Syllabus d'algebre, chapitre II : les matrices.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 17 a 30. Les
exemples des paragraphes II.2 a II.4, l'echelonnement de II.7.3, l'inverse
de II.8.2, puis les « Solutions du 2eme chapitre » : chaque reponse
officielle des exercices 1 a 10 est recalculee.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.echelonnement import (
    echelonner,
    echelonner_simplement,
    est_echelonnee,
    est_echelonnee_simple,
    inverse_par_la_matrice_compagnon,
)
from polytech.maths.domain.algebre.matrices import (
    Matrice,
    addition,
    identite,
    multiplication_par_un_scalaire,
    produit,
    transposee,
)
from polytech.maths.domain.algebre.matrices_particulieres import (
    commutant,
    est_antisymetrique,
    est_diagonale,
    est_symetrique,
    est_triangulaire_inferieure,
    est_une_matrice_orthogonale,
    parties_symetrique_et_antisymetrique,
    trace,
)
from polytech.maths.domain.algebre.triangularisation import (
    est_triangulaire_superieure,
)
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
EXEMPLE = M([[2, 4, 3, 1], [-1, 2, 1, 1], [2, 3, 2, 1]])
A_INVERSER = M([[2, 4, 3], [-1, 2, 1], [2, 3, 2]])


def test_l_addition_et_le_produit_par_un_scalaire_des_pages_18() -> None:
    """II.2.1 et II.2.2 : terme a terme."""
    somme = addition(((2, 1), (-1, 0), (4, -2)), ((0, 3), (2, 4), (7, 0)))
    attendue: Matrice = ((2, 4), (1, 4), (11, -2))
    assert somme == attendue
    triple = multiplication_par_un_scalaire(-3, ((2, 1), (6, -7)))
    assert triple == ((-6, -3), (-18, 21))


def test_le_produit_de_la_page_19() -> None:
    """Une `2 x 3` par une `3 x 2` : `[[17, 20], [20, 16]]`."""
    gauche, droite = ((2, 3, 4), (5, 1, 2)), ((3, 2), (1, 4), (2, 1))
    assert produit(gauche, droite) == ((17, 20), (20, 16))
    assert len(produit(droite, gauche)) == 3


def test_l_identite_est_neutre_pour_le_produit() -> None:
    """II.2.3, « Neutre » : `A I = I A = A`."""
    a = ((1, 2), (3, 4))
    assert produit(a, identite(2)) == a
    assert produit(identite(2), a) == a


def test_les_trois_mises_en_garde_de_la_page_20() -> None:
    """Pas commutatif ; `AB = O` sans facteur nul ; `AB = AC` sans `B = C`."""
    a, b = ((1, 2), (3, 4)), ((5, 6), (7, 8))
    assert produit(a, b) != produit(b, a)
    a, b = ((1, 1), (2, 2)), ((1, 3), (-1, -3))
    nulle = ((0, 0), (0, 0))
    assert produit(a, b) == nulle
    assert produit(a, b) == produit(a, nulle)
    assert b != nulle


def test_la_transposee_et_ses_proprietes() -> None:
    """II.3 : les lignes en colonnes, et `(AB)^t = B^t A^t`."""
    a = ((2, 3), (1, 4), (5, 6))
    assert transposee(a) == ((2, 1, 5), (3, 4, 6))
    assert transposee(transposee(a)) == a
    b = ((1, 2), (3, 4))
    assert transposee(produit(a, b)) == produit(transposee(b), transposee(a))


def test_la_trace_est_la_somme_des_elements_diagonaux() -> None:
    """II.1."""
    assert trace(A_INVERSER) == 6
    with pytest.raises(ValueError, match="trace d'une matrice non carree"):
        trace(EXEMPLE)


def test_les_matrices_particulieres_de_la_page_21() -> None:
    """II.4 : chaque exemple, et chaque contre-exemple, du syllabus."""
    assert est_symetrique(M([[1, 4, 6], [4, 2, 0], [6, 0, 5]]))
    assert not est_symetrique(M([[1, 4, 6], [-4, 2, 0], [5, 0, 5]]))
    assert est_antisymetrique(M([[0, 1, 3], [-1, 0, 6], [-3, -6, 0]]))
    assert not est_antisymetrique(M([[0, 4], [-4, 1]]))
    assert est_diagonale(sp.diag(1, 3, 0, 4))
    assert not est_diagonale(M([[1, 2], [0, 3]]))
    assert est_triangulaire_superieure(M([[7, 0, 10], [0, 4, 2], [0, 0, 5]]))
    inferieure = M([[2, 0, 0, 0], [1, 2, 0, 0], [0, 1, 0, 0], [4, 1, 1, 5]])
    assert est_triangulaire_inferieure(inferieure)
    assert not est_triangulaire_inferieure(inferieure.T)


def test_une_matrice_non_carree_n_est_ni_symetrique_ni_diagonale() -> None:
    """Les quatre definitions de II.4 disent « matrice carree »."""
    rectangle = sp.zeros(2, 3)
    assert not est_symetrique(rectangle)
    assert not est_antisymetrique(rectangle)
    assert not est_diagonale(rectangle)
    assert not est_triangulaire_inferieure(rectangle)


def test_la_reponse_de_l_exercice_1() -> None:
    """`-3X + P = Q` : `X = (P - Q)/3`."""
    p = M([[2, 3, 4], [5, 1, 2], [0, 3, 5]]) * M([[2, 1], [3, 2], [4, 0]])
    q = M([[0, 2], [1, 4], [2, 1]])
    attendue = M([[R(29, 3), 2], [R(20, 3), 1], [9, R(5, 3)]])
    assert (p - q) / 3 == attendue
    assert -3 * attendue + p == q


def test_la_reponse_de_l_exercice_2() -> None:
    """`A(X - B) = XC` : `X = [[-3, 2], [0, -1]]`, et c'est la seule."""
    a, b, c = M([[3, -2], [0, 1]]), M([[-3, 1], [2, 0]]), M([[0, -1], [2, 1]])
    inconnues = sp.symbols("x1:5")
    x = M(2, 2, inconnues)
    solution = sp.solve(list(a * (x - b) - x * c), inconnues, dict=True)
    assert len(solution) == 1
    assert x.subs(solution[0]) == M([[-3, 2], [0, -1]])


def test_la_reponse_de_l_exercice_3() -> None:
    """Le commutant de `[[3,1,0],[0,3,1],[0,0,3]]` : `[[a,b,c],[0,a,b],[0,0,a]]`."""
    a = M([[3, 1, 0], [0, 3, 1], [0, 0, 3]])
    generale = commutant(a)
    assert len(generale.free_symbols) == 3
    assert generale * a == a * generale
    assert generale[0, 0] == generale[1, 1] == generale[2, 2]
    assert generale[0, 1] == generale[1, 2]
    assert generale[1, 0] == generale[2, 0] == generale[2, 1] == 0
    assert len({generale[0, 0], generale[0, 1], generale[0, 2]}) == 3


def test_la_reponse_de_l_exercice_4() -> None:
    """`[[x, y], [0, z]]` commute avec toutes les `2 x 2` : `y = 0` et `z = x`."""
    x, y, z = sp.symbols("x y z")
    candidate = M([[x, y], [0, z]])
    conditions: list[sp.Expr] = []
    for i in range(2):
        for j in range(2):
            unite = sp.zeros(2, 2)
            unite[i, j] = 1
            conditions.extend(candidate * unite - unite * candidate)
    assert sp.solve(conditions, [y, z], dict=True) == [{y: 0, z: x}]


def test_la_reponse_de_l_exercice_5() -> None:
    """`X^t A X` diagonale pour `x = (2 +- racine(13))/3`."""
    x = M([[1, -X], [X, 1]])
    transformee = sp.expand(x.T * M([[0, 3], [3, 4]]) * x)
    racines = sp.solve(transformee[0, 1], X)
    assert set(racines) == {(2 + sp.sqrt(13)) / 3, (2 - sp.sqrt(13)) / 3}
    for racine in racines:
        assert est_diagonale(sp.simplify(transformee.subs(X, racine)))


def test_la_reponse_de_l_exercice_6() -> None:
    """`[[1, 2], [3, 4]] = [[1, 5/2], [5/2, 4]] + [[0, -1/2], [1/2, 0]]`."""
    symetrique, antisymetrique = parties_symetrique_et_antisymetrique(
        M([[1, 2], [3, 4]])
    )
    assert symetrique == M([[1, R(5, 2)], [R(5, 2), 4]])
    assert antisymetrique == M([[0, -R(1, 2)], [R(1, 2), 0]])
    assert est_symetrique(symetrique)
    assert est_antisymetrique(antisymetrique)


def test_la_reponse_de_l_exercice_7() -> None:
    """La matrice de rotation est orthogonale ; `[[1, 2], [3, 4]]` ne l'est pas."""
    angle = sp.Symbol("theta", real=True)
    rotation = M([[sp.cos(angle), -sp.sin(angle)], [sp.sin(angle), sp.cos(angle)]])
    assert est_une_matrice_orthogonale(rotation)
    assert not est_une_matrice_orthogonale(M([[1, 2], [3, 4]]))
    assert not est_une_matrice_orthogonale(M([[1, 0, 0], [0, 1, 0]]))


def test_la_reponse_de_l_exercice_8() -> None:
    """`det [[x, x/2 - 1/3], [3, x - 1]] = 0` : `x = 2` ou `x = 1/2`."""
    determinant = Determinant(M([[X, X / 2 - R(1, 3)], [3, X - 1]]))
    assert sp.expand(determinant.par_cofacteurs(0) - (X**2 - R(5, 2) * X + 1)) == 0
    assert determinant.valeurs_singulieres(X) == (R(1, 2), 2)


def test_les_trois_exemples_de_la_definition_ii_7_1() -> None:
    """Echelonnee sans etre simple ; non echelonnee ; echelonnee simple."""
    premiere = M(
        [[2, 0, 0, 0, 4], [0, 2, 0, 5, 2], [0, 0, 0, 1, 7], [0, 0, 0, 0, 5]]
    ).col_join(sp.zeros(2, 5))
    assert est_echelonnee(premiere)
    assert not est_echelonnee_simple(premiere)
    deuxieme = M(
        [
            [2, 0, 0, 0, 4],
            [0, 2, 0, 5, 2],
            [0, 3, 1, 1, 7],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
        ]
    )
    assert not est_echelonnee(deuxieme)
    assert not est_echelonnee_simple(deuxieme)
    troisieme = M(
        [[1, 0, 9, 0, 0], [0, 1, 4, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    ).col_join(sp.zeros(2, 5))
    assert est_echelonnee(troisieme)
    assert est_echelonnee_simple(troisieme)


def test_une_ligne_nulle_suivie_d_une_ligne_non_nulle_n_est_pas_echelonnee() -> None:
    """Premiere clause de la definition."""
    assert not est_echelonnee(M([[1, 2], [0, 0], [0, 3]]))
    assert est_echelonnee(sp.zeros(2, 3))


def test_deux_pivots_dans_la_meme_colonne_ne_sont_pas_echelonnes() -> None:
    """La colonne du pivot croit « strictement »."""
    assert not est_echelonnee(M([[1, 2], [3, 4]]))
    assert not est_echelonnee(M([[0, 1, 2], [0, 5, 0]]))


def test_des_pivots_egaux_a_un_ne_suffisent_pas_sans_l_echelonnement() -> None:
    """`[[0, 1], [1, 0]]` : pivots 1, seuls dans leur colonne, mais mal ordonnes."""
    assert not est_echelonnee_simple(M([[0, 1], [1, 0]]))


def test_un_pivot_different_de_un_ou_surmonte_n_est_pas_simple() -> None:
    """Les deux clauses de l'echelonnement simple, une a une."""
    assert not est_echelonnee_simple(M([[2, 0], [0, 1]]))
    assert not est_echelonnee_simple(M([[1, 3], [0, 1]]))
    assert est_echelonnee_simple(M([[1, 3, 0], [0, 0, 1]]))


def test_l_echelonnement_de_l_exemple_ii_7_3() -> None:
    """`l2 -> l2 + 1/2 l1`, `l3 -> l3 - l1`, puis `l3 -> l3 + 1/4 l2`."""
    resultat = echelonner(EXEMPLE)
    assert resultat.operations == (
        "l2 -> l2 + 1/2 l1",
        "l3 -> l3 - l1",
        "l3 -> l3 + 1/4 l2",
    )
    attendue = M([[2, 4, 3, 1], [0, 4, R(5, 2), R(3, 2)], [0, 0, -R(3, 8), R(3, 8)]])
    assert resultat.matrice == attendue
    assert est_echelonnee(resultat.matrice)


def test_l_echelonnement_simple_de_l_exemple_ii_7_3() -> None:
    """`[[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, -1]]`."""
    simple = echelonner_simplement(EXEMPLE)
    assert simple == M([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, -1]])
    assert est_echelonnee_simple(simple)


def test_un_pivot_nul_fait_permuter_deux_lignes() -> None:
    """II.7.2 : « si cet element n'est pas a la premiere ligne, on permute »."""
    resultat = echelonner(M([[0, 1, 2], [3, 4, 5]]))
    assert resultat.operations == ("l1 <-> l2",)
    assert resultat.matrice == M([[3, 4, 5], [0, 1, 2]])


def test_une_colonne_nulle_est_sautee() -> None:
    """II.7.2 : « on passe tout de suite a la colonne suivante »."""
    resultat = echelonner(M([[0, 2, 4], [0, 1, 5]]))
    assert resultat.matrice == M([[0, 2, 4], [0, 0, 3]])
    assert echelonner_simplement(M([[0, 2, 4], [0, 1, 5]])) == M([[0, 1, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    ("matrice", "attendue"),
    [
        (
            [[1, -2, 3, -1], [2, -1, 2, 2], [-3, -5, 1, -5], [3, 1, 2, 3]],
            [
                [1, 0, 0, R(15, 7)],
                [0, 1, 0, -R(4, 7)],
                [0, 0, 1, -R(10, 7)],
                [0, 0, 0, 0],
            ],
        ),
        (
            [
                [1, 2, 1, 2, -3],
                [3, 6, 4, -1, -1],
                [4, 8, 5, 1, -1],
                [-2, -4, -3, 3, -5],
            ],
            [[1, 2, 0, 9, 0], [0, 0, 1, -7, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
        ),
        (
            [[6, 3, -4], [-4, 1, -6], [1, 2, -5]],
            [[1, 0, R(7, 9)], [0, 1, -R(26, 9)], [0, 0, 0]],
        ),
        (
            [
                [1, 3, 1, -2, -3],
                [1, 4, 3, -1, -4],
                [2, 3, -4, -7, -3],
                [3, 8, 1, -7, -8],
            ],
            [[1, 0, -5, -5, 0], [0, 1, 2, 1, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ),
    ],
    ids=["A", "B", "C", "D"],
)
def test_la_reponse_de_l_exercice_9(
    matrice: list[list[int]], attendue: list[list[sp.Expr]]
) -> None:
    """Les quatre formes echelonnees simples des solutions."""
    simple = echelonner_simplement(M(matrice))
    assert simple == M(attendue)
    assert simple == M(matrice).rref()[0]


def test_l_inverse_de_l_exemple_ii_8_2() -> None:
    """Par la matrice-compagnon : `[A | I]` echelonnee jusqu'a `[I | A^-1]`."""
    inverse = inverse_par_la_matrice_compagnon(A_INVERSER)
    attendue = M(
        [
            [-R(1, 3), -R(1, 3), R(2, 3)],
            [-R(4, 3), R(2, 3), R(5, 3)],
            [R(7, 3), -R(2, 3), -R(8, 3)],
        ]
    )
    assert inverse == attendue
    assert A_INVERSER * inverse == sp.eye(3)


def test_une_matrice_non_inversible_ne_s_echelonne_pas_en_l_identite() -> None:
    """II.8.1 : « si cela n'est pas le cas, A n'est pas inversible »."""
    with pytest.raises(ValueError, match="n'est pas l'identite"):
        inverse_par_la_matrice_compagnon(M([[1, 2], [2, 4]]))
    with pytest.raises(ValueError, match="matrice non carree"):
        inverse_par_la_matrice_compagnon(EXEMPLE)


def test_les_reponses_a_a_d_de_l_exercice_10() -> None:
    """Quatre inverses numeriques."""
    a = inverse_par_la_matrice_compagnon(M([[1, 2, -1], [0, 3, 0], [-2, 1, 0]]))
    assert a == M([[0, R(1, 6), -R(1, 2)], [0, R(1, 3), 0], [-1, R(5, 6), -R(1, 2)]])
    b = inverse_par_la_matrice_compagnon(M([[1, 2, 3], [4, 5, 6], [7, 8, 10]]))
    assert b == M([[-R(2, 3), -R(4, 3), 1], [-R(2, 3), R(11, 3), -2], [1, -2, 1]])
    c = inverse_par_la_matrice_compagnon(
        M([[1, -2, 3, -1], [2, -1, 2, 2], [-3, -5, 1, -5], [3, 1, 2, 4]])
    )
    assert c == M(
        [
            [R(3, 7), 2, -1, -R(15, 7)],
            [R(2, 7), -1, 0, R(4, 7)],
            [R(3, 14), -R(3, 2), R(1, 2), R(10, 7)],
            [-R(1, 2), -R(1, 2), R(1, 2), 1],
        ]
    )
    d = M([[1, 1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]])
    assert inverse_par_la_matrice_compagnon(d) == d / 4


@pytest.mark.parametrize("ordre", [2, 4, 5])
def test_les_reponses_e_et_f_de_l_exercice_10(ordre: int) -> None:
    """e) la bidiagonale `1, -1` s'inverse en triangulaire de 1 ; f) l'antidiagonale."""
    bidiagonale = sp.eye(ordre)
    for i in range(ordre - 1):
        bidiagonale[i, i + 1] = -1
    pleine = M(ordre, ordre, lambda i, j: 1 if j >= i else 0)
    assert inverse_par_la_matrice_compagnon(bidiagonale) == pleine
    lambdas = sp.symbols(f"lambda1:{ordre + 1}", nonzero=True)
    anti = M(ordre, ordre, lambda i, j: lambdas[i] if i + j == ordre - 1 else 0)
    attendue = M(ordre, ordre, lambda i, j: 1 / lambdas[j] if i + j == ordre - 1 else 0)
    assert inverse_par_la_matrice_compagnon(anti) == attendue


def test_une_matrice_immuable_s_echelonne_comme_une_autre() -> None:
    """Regression : `sp.expand` rend une matrice immuable, et il fallait permuter."""
    immuable = sp.ImmutableMatrix([[0, 1, 2], [3, 4, 5]])
    resultat = echelonner(immuable)
    assert resultat.operations == ("l1 <-> l2",)
    assert resultat.matrice == M([[3, 4, 5], [0, 1, 2]])
    assert echelonner_simplement(immuable) == M([[1, 0, -1], [0, 1, 2]])
    assert immuable == sp.ImmutableMatrix([[0, 1, 2], [3, 4, 5]])
