"""Syllabus d'algebre, chapitre V : les determinants.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 55 a 63. Les
deux exemples resolus, operation par operation et par chacune de leurs
methodes, puis les « Solutions du 5eme chapitre » : chaque reponse
officielle est recalculee, et les trois enonces a demontrer (Vandermonde,
exercices 4 et 6) sont verifies.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.operations_elementaires import (
    combiner,
    du_triangulaire,
    multiplier,
    par_echelonnement,
    par_sarrus,
    permuter,
    selon_la_colonne,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

M = sp.Matrix
A, B, C, D, LAMBDA, X = sp.symbols("a b c d lambda x")
EXEMPLE_1 = M([[2, 1, -1, 1], [-1, 2, -1, -1], [1, 1, 2, -1], [-1, 1, 1, 2]])
ETOILE = M([[-1, 2, -1, -1], [0, -1, 2, 3], [0, 0, 7, 7], [0, 0, 7, 14]])
EXEMPLE_2 = M([[A - 2, 2, -1], [2, A, 2], [2 * A, 2 * (A + 1), A + 1]])
PRODUIT = A * (A - 1) * (A - 2)


def test_les_trois_proprietes_du_v_2() -> None:
    """Permuter change le signe ; multiplier multiplie ; combiner conserve."""
    assert permuter(EXEMPLE_1, 0, 1).facteur == -1
    assert multiplier(EXEMPLE_1, 2, 7).facteur == 7
    assert combiner(EXEMPLE_1, 1, 2, 0).facteur == 1
    for transformee in (
        permuter(EXEMPLE_1, 0, 1),
        multiplier(EXEMPLE_1, 2, 7),
        combiner(EXEMPLE_1, 1, 2, 0),
    ):
        assert transformee.matrice.det() == transformee.facteur * EXEMPLE_1.det()


def test_une_ligne_ne_se_combine_pas_avec_elle_meme() -> None:
    """Propriete 3 : `l_i -> l_i + beta l_k` demande `i != k`."""
    with pytest.raises(ValueError, match="ne se combine pas avec elle-meme"):
        combiner(EXEMPLE_1, 1, 2, 1)


def test_les_operations_de_l_exemple_1_menent_a_l_etoile() -> None:
    """Page 56 : deux permutations, cinq combinaisons, la matrice a l'etoile."""
    etape = permuter(EXEMPLE_1, 0, 1)
    facteur = etape.facteur
    for ligne, beta in ((1, 2), (2, 1), (3, -1)):
        etape = combiner(etape.matrice, ligne, beta, 0)
    attendue = M([[-1, 2, -1, -1], [0, 5, -3, -1], [0, 3, 1, -2], [0, -1, 2, 3]])
    assert etape.matrice == attendue
    etape = permuter(etape.matrice, 1, 3)
    facteur *= etape.facteur
    for ligne, beta in ((2, 3), (3, 5)):
        etape = combiner(etape.matrice, ligne, beta, 1)
    assert etape.matrice == ETOILE
    assert facteur == 1


def test_l_exemple_1_developpe_puis_par_sarrus() -> None:
    """`(-1) . [1 - 20 - 27 - (15 - 6 - 6)] = 49`."""
    avant_l_etoile = M([[-1, 2, -1, -1], [0, -1, 2, 3], [0, 3, 1, -2], [0, 5, -3, -1]])
    termes = selon_la_colonne(avant_l_etoile, 0)
    mineur = M([[-1, 2, 3], [3, 1, -2], [5, -3, -1]])
    assert termes == (-mineur.det(), 0, 0, 0)
    assert par_sarrus(mineur) == 1 - 20 - 27 - (15 - 6 - 6)
    assert sum(termes) == 49


def test_l_exemple_1_par_la_matrice_triangulaire() -> None:
    """`l4 -> l4 - l3`, puis le produit des elements diagonaux : `(-1)(-1) 7.7`."""
    triangulaire = combiner(ETOILE, 3, -1, 2).matrice
    assert triangulaire[3, :] == M([[0, 0, 0, 7]])
    assert du_triangulaire(triangulaire) == 49


def test_l_exemple_1_en_mettant_7_en_evidence() -> None:
    """Page 58 : `det = 7 . 7 . det(...)`, et le dernier determinant vaut 1."""
    reduite = multiplier(
        multiplier(ETOILE, 2, sp.Rational(1, 7)).matrice, 3, sp.Rational(1, 7)
    )
    assert reduite.matrice[2:, :] == M([[0, 0, 1, 1], [0, 0, 1, 2]])
    assert reduite.matrice.det() == 1
    assert ETOILE.det() == 49 * reduite.matrice.det()


def test_l_exemple_1_par_echelonnement_complet() -> None:
    """Le chemin automatique rend la meme valeur que les quatre chemins du syllabus."""
    assert par_echelonnement(EXEMPLE_1) == 49
    assert Determinant(EXEMPLE_1).par_cofacteurs(0) == 49


def test_le_produit_diagonal_ne_vaut_que_pour_une_matrice_triangulaire() -> None:
    with pytest.raises(ValueError, match="matrice non triangulaire"):
        du_triangulaire(EXEMPLE_1)
    assert du_triangulaire(M([[2, 0], [5, 3]])) == 6


def test_la_premiere_methode_de_l_exemple_2() -> None:
    """Page 59 : `det = -1/4 . 2 . 2 . (-a)(a-1)(a-2) = a(a-1)(a-2)`."""
    etape = permuter(EXEMPLE_2, 0, 1)
    facteur = etape.facteur
    etape = multiplier(etape.matrice, 1, 2)
    facteur *= etape.facteur
    etape = combiner(etape.matrice, 1, -(A - 2), 0)
    etape = combiner(etape.matrice, 2, -A, 0)
    attendue = M(
        [
            [2, A, 2],
            [0, -(A**2) + 2 * A + 4, -2 * (A - 1)],
            [0, -(A**2) + 2 * A + 2, -(A - 1)],
        ]
    )
    assert sp.expand(etape.matrice - attendue) == sp.zeros(3, 3)
    assert facteur == -2
    etape = combiner(etape.matrice, 1, -1, 2)
    etape = multiplier(etape.matrice, 2, 2)
    facteur *= etape.facteur
    etape = combiner(etape.matrice, 2, -(-(A**2) + 2 * A + 2), 1)
    assert sp.expand(etape.matrice[2, 2] + PRODUIT) == 0
    assert sp.expand(etape.matrice[2, 1]) == 0
    assert facteur == -4
    assert sp.expand(du_triangulaire(sp.expand(etape.matrice)) / facteur - PRODUIT) == 0


def test_les_deuxieme_et_troisieme_methodes_de_l_exemple_2() -> None:
    """Le developpement selon la premiere colonne, puis la regle de Sarrus."""
    assert sp.expand(sum(selon_la_colonne(EXEMPLE_2, 0)) - PRODUIT) == 0
    assert sp.expand(par_sarrus(EXEMPLE_2) - PRODUIT) == 0
    assert sp.expand(par_sarrus(EXEMPLE_2) - (A**3 - 3 * A**2 + 2 * A)) == 0
    assert sp.simplify(par_echelonnement(EXEMPLE_2) - PRODUIT) == 0


def test_la_regle_de_sarrus_telle_que_la_page_55_l_ecrit() -> None:
    """Trois produits descendants, moins trois produits montants, terme a terme."""
    m = M(3, 3, sp.symbols("a11 a12 a13 a21 a22 a23 a31 a32 a33"))
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = m
    formule = (
        a11 * a22 * a33
        + a12 * a23 * a31
        + a13 * a21 * a32
        - (a31 * a22 * a13 + a32 * a23 * a11 + a33 * a21 * a12)
    )
    assert sp.expand(par_sarrus(m) - formule) == 0
    assert sp.expand(par_sarrus(m) - m.det()) == 0


def test_la_regle_de_sarrus_ne_vaut_que_pour_l_ordre_trois() -> None:
    """« Dans le cas particulier d'une matrice 3 x 3, et uniquement dans ce cas-la »."""
    with pytest.raises(ValueError, match="uniquement pour une matrice 3 x 3"):
        par_sarrus(EXEMPLE_1)


def test_un_determinant_ne_se_calcule_que_sur_une_matrice_carree() -> None:
    with pytest.raises(ValueError, match="matrice non carree"):
        par_echelonnement(sp.zeros(2, 3))
    with pytest.raises(ValueError, match="colonne hors de la matrice"):
        selon_la_colonne(EXEMPLE_2, 3)


def test_une_colonne_sans_pivot_annule_le_determinant() -> None:
    """Une matrice singuliere s'echelonne avec un zero sur la diagonale."""
    assert par_echelonnement(M([[1, 2, 3], [2, 4, 6], [1, 0, 1]])) == 0
    assert par_echelonnement(M([[0, 1], [1, 0]])) == -1


def test_le_determinant_decide_de_la_solution_unique() -> None:
    """V.5 : solution unique si et seulement si `det A != 0`, `a` hors de 0, 1, 2."""
    assert Determinant(EXEMPLE_2).valeurs_singulieres(A) == (0, 1, 2)
    systeme = SystemeLineaire(EXEMPLE_2.subs(A, 3), M([1, 1, 2]))
    assert systeme.rang() == 3
    assert systeme.solution_verifiee()


def test_les_reponses_a_c_d_de_l_exercice_1() -> None:
    """a) `-8` ; c) `72` ; d) `-14`."""
    a = M([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, 1, 1]])
    assert par_echelonnement(a) == -8
    c = M([[1, 2, 3, 4], [5, 6, 7, 8], [8, 4, 6, 5], [4, 4, 2, -2]])
    assert par_echelonnement(c) == 72
    d = M(
        [
            [1, 0, -3, 0, 7, 2],
            [11, -3, 2, 1, 10, -2],
            [0, 0, 1, 0, 6, 0],
            [5, 1, 4, 0, -9, 6],
            [0, 0, 0, 0, 2, 0],
            [-3, 0, -10, 0, 4, 1],
        ]
    )
    assert par_echelonnement(d) == -14
    assert d.det() == -14


def test_les_reponses_b_e_de_l_exercice_1() -> None:
    """b) `bcd + acd + abd + abc + abcd` ; e) `(a - b)(b - c)(c - a)`."""
    b = sp.ones(4, 4) + sp.diag(A, B, C, D)
    attendu = B * C * D + A * C * D + A * B * D + A * B * C + A * B * C * D
    assert sp.expand(b.det() - attendu) == 0
    assert sp.simplify(par_echelonnement(b) - attendu) == 0
    e = M([[1, A, B * C], [1, B, A * C], [1, C, A * B]])
    assert sp.expand(par_sarrus(e) - (A - B) * (B - C) * (C - A)) == 0


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_la_reponse_f_de_l_exercice_1(n: int) -> None:
    """La tridiagonale `1, 2, 1` d'ordre `n` : `n + 1`."""
    tridiagonale = M(n, n, lambda i, j: 2 if i == j else int(abs(i - j) == 1))
    assert par_echelonnement(tridiagonale) == n + 1


def test_la_reponse_de_l_exercice_2() -> None:
    """`A A^t = (a^2 + 3) I`, d'ou `det A = (a^2 + 3)^2`."""
    a = M([[A, 1, -1, 1], [-1, A, -1, -1], [1, 1, A, -1], [-1, 1, 1, A]])
    assert sp.expand(a * a.T) == (A**2 + 3) * sp.eye(4)
    assert sp.expand(a.det() - (A**2 + 3) ** 2) == 0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_le_determinant_de_vandermonde(n: int) -> None:
    """Exercice 3 : le produit des `a_i - a_j` pour `i > j`."""
    points = sp.symbols(f"a1:{n + 1}")
    vandermonde = M(n, n, lambda i, j: points[i] ** j)
    produit = sp.Mul(*(points[i] - points[j] for i in range(n) for j in range(i)))
    assert sp.expand(vandermonde.det() - produit) == 0


def test_l_identite_de_l_exercice_4() -> None:
    """`det(b + c, c + a, a + b) = 2 det(a, b, c)`, colonnes."""
    a, b, c = (M(sp.symbols(f"{nom}0:3")) for nom in "abc")
    gauche = M.hstack(b + c, c + a, a + b)
    assert sp.expand(gauche.det() - 2 * M.hstack(a, b, c).det()) == 0


@pytest.mark.parametrize(("n", "attendu"), [(1, 1), (2, -2), (3, 0), (4, 0), (6, 0)])
def test_la_reponse_de_l_exercice_5(n: int, attendu: int) -> None:
    """Les entiers de 1 a `n^2` ranges en lignes : 1, -2, puis 0."""
    matrice = M(n, n, lambda i, j: i * n + j + 1)
    assert par_echelonnement(matrice) == attendu


def test_le_determinant_de_la_matrice_des_cofacteurs() -> None:
    """Exercice 6 : `det A~ = (det A)^(n-1)`."""
    for matrice in (EXEMPLE_1, EXEMPLE_2.subs(A, 3), M([[1, 2], [3, 4]])):
        cofacteurs = Determinant(matrice).comatrice()
        assert cofacteurs.det() == matrice.det() ** (matrice.rows - 1)


def test_la_reponse_de_l_exercice_7() -> None:
    """`lambda = -2` ou `4`."""
    matrice = M([[LAMBDA - 1, 3, -3], [-3, LAMBDA + 5, -3], [-6, 6, LAMBDA - 4]])
    assert Determinant(matrice).valeurs_singulieres(LAMBDA) == (-2, 4)
    assert sp.expand(par_sarrus(matrice) - (LAMBDA + 2) ** 2 * (LAMBDA - 4)) == 0


def test_la_reponse_de_l_exercice_8() -> None:
    """`x = 0`, `2` ou `-2`."""
    matrice = M([[2, 2, 3, 4], [2, 2 - X**2, 3, 4], [2, 3, 1, 5], [2, 3, 1, 9 - X**2]])
    assert Determinant(matrice).valeurs_singulieres(X) == (-2, 0, 2)


def test_les_operations_elementaires_acceptent_une_matrice_immuable() -> None:
    """Regression : chaque operation rend une copie, meme d'une matrice immuable."""
    immuable = sp.ImmutableMatrix([[1, 2], [3, 4]])
    assert permuter(immuable, 0, 1).matrice == M([[3, 4], [1, 2]])
    assert multiplier(immuable, 0, 5).matrice == M([[5, 10], [3, 4]])
    assert combiner(immuable, 1, -3, 0).matrice == M([[1, 2], [0, -2]])
    assert par_echelonnement(immuable) == -2
    assert immuable == sp.ImmutableMatrix([[1, 2], [3, 4]])
