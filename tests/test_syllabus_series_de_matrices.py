"""Syllabus de mathematiques appliquees, chapitre 7 : suites et series de matrices.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 123 a 133. La norme d'une matrice, la convergence d'une suite
(exemple 7.1), la serie geometrique et son cas particulier, l'exponentielle
par la serie, par diagonalisation et par la forme normale de Jordan.

Trois erreurs du syllabus sont etablies : l'inegalite `g_i >= m_i` et
l'orthogonalite des espaces propres (page 129), l'inverse de `P` et donc
`e^A` dans l'exemple de Jordan (page 133).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.exponentielle import (
    exponentielle_d_un_bloc,
    par_diagonalisation,
    par_jordan,
    par_la_serie,
)
from polytech.maths.domain.algebre.jordan import (
    forme_de_jordan,
    multiplicites,
    vecteur_generalise,
)
from polytech.maths.domain.algebre.normes import distance, norme
from polytech.maths.domain.algebre.series_de_matrices import (
    limite_de_la_suite,
    somme_geometrique,
    somme_partielle_geometrique,
)
from polytech.maths.domain.differentielles.systemes import SystemeDifferentiel
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
E = sp.E
X = sp.Symbol("x", real=True)
ANTIDIAGONALE = sp.Matrix([[0, X], [X, 0]])
JORDAN = sp.Matrix([[1, 0, 1], [2, -2, -1], [0, 0, 1]])
PASSAGE = sp.Matrix([[0, 3, 1], [2, 2, -1], [0, 0, 3]]) / 2
BLOCS = sp.Matrix([[-2, 0, 0], [0, 1, 1], [0, 0, 1]])


def _nulle(matrice: sp.Matrix) -> bool:
    return bool(sp.simplify(matrice) == sp.zeros(*matrice.shape))


def test_le_produit_des_matrices_n_est_pas_commutatif() -> None:
    """Page 123 : les deux produits de l'exemple different."""
    a, b = sp.Matrix([[0, 1], [-1, 0]]), sp.Matrix([[0, 1], [2, 0]])
    assert a * b == sp.Matrix([[2, 0], [0, -1]])
    assert b * a == sp.Matrix([[-1, 0], [0, 2]])


def test_la_norme_d_une_matrice() -> None:
    """Page 124 : la racine de la somme des carres des coefficients."""
    assert norme(sp.Matrix([[1, -2], [2, 4]])) == 5
    assert norme(sp.zeros(2, 2)) == 0
    assert norme(sp.Matrix([3, 4])) == 5


def test_les_proprietes_de_la_norme() -> None:
    """Homogeneite, inegalite triangulaire, `||AB|| <= ||A|| ||B||`."""
    a, b = sp.Matrix([[1, 2], [0, -1]]), sp.Matrix([[3, 0], [1, 1]])
    assert norme(-3 * a) == 3 * norme(a)
    assert norme(a + b) <= norme(a) + norme(b)
    assert norme(a * b) <= norme(a) * norme(b)
    assert norme(a**3) <= norme(a) ** 3
    assert distance(a, b) == norme(a - b) == sp.sqrt(4 + 4 + 1 + 4)
    assert distance(a, a) == 0


def test_la_limite_de_la_suite_de_l_exemple_7_1() -> None:
    """Quatre suites numeriques : `2`, `e`, `3/2` et `ln 2`."""
    n, k = sp.symbols("n k", integer=True, positive=True)
    suite = sp.Matrix(
        [
            [sp.Sum(R(1, 2) ** k, (k, 0, n)), sp.Sum(1 / sp.factorial(k), (k, 0, n))],
            [sp.Sum(R(1, 3) ** k, (k, 0, n)), sp.Sum((-1) ** (k - 1) / k, (k, 1, n))],
        ]
    )
    limite = limite_de_la_suite(suite, n)
    assert limite == sp.Matrix([[2, E], [R(3, 2), sp.log(2)]])


def test_la_distance_a_la_limite_tend_vers_zero() -> None:
    """« `d(A_n, A) -> 0` » : sur une suite ou tout se calcule."""
    n = sp.Symbol("n", integer=True, positive=True)
    suite = sp.Matrix([[1 + 1 / n, 2**-n], [0, n / (n + 1)]])
    limite = limite_de_la_suite(suite, n)
    assert limite == sp.Matrix([[1, 0], [0, 1]])
    assert sp.limit(distance(suite, limite), n, sp.oo) == 0


def test_une_suite_divergente_n_a_pas_de_limite() -> None:
    n = sp.Symbol("n", integer=True, positive=True)
    with pytest.raises(ValueError, match="suite de matrices divergente"):
        limite_de_la_suite(sp.Matrix([[n, 0], [0, 1]]), n)


def test_les_sommes_partielles_de_la_serie_geometrique() -> None:
    """Page 127 : `1 + x^2 + ...` sur la diagonale, `x + x^3 + ...` a cote."""
    somme = somme_partielle_geometrique(ANTIDIAGONALE, 5)
    pairs, impairs = 1 + X**2 + X**4, X + X**3 + X**5
    assert _nulle(somme - sp.Matrix([[pairs, impairs], [impairs, pairs]]))


def test_la_somme_de_la_serie_geometrique() -> None:
    """`somme A^n = (I - A)^-1 = [[1, x], [x, 1]] / (1 - x^2)`."""
    attendue = sp.Matrix([[1, X], [X, 1]]) / (1 - X**2)
    assert _nulle(somme_geometrique(ANTIDIAGONALE) - attendue)
    numerique = ANTIDIAGONALE.subs(X, R(1, 2))
    assert somme_geometrique(numerique) == sp.Matrix([[4, 2], [2, 4]]) / 3
    ecart = somme_geometrique(numerique) - somme_partielle_geometrique(numerique, 60)
    assert norme(ecart) < R(1, 10**15)


def test_la_condition_sur_la_norme_est_suffisante_sans_etre_necessaire() -> None:
    """`||A|| < 1` demande `|x| < 1/racine(2)` ; la serie converge pour tout `|x| < 1`.

    Pour `x = 4/5`, la norme vaut `4 racine(2)/5 > 1`, et pourtant les sommes
    partielles tendent vers `(I - A)^-1` : c'est le rayon spectral qui decide.
    """
    assert norme(ANTIDIAGONALE) == sp.sqrt(2) * sp.Abs(X)
    numerique = ANTIDIAGONALE.subs(X, R(4, 5))
    assert norme(numerique) > 1
    ecart = somme_geometrique(numerique) - somme_partielle_geometrique(numerique, 400)
    assert norme(ecart) < R(1, 10**30)


def test_une_serie_geometrique_divergente_n_a_pas_de_somme() -> None:
    """`x = 2` : `(I - A)^-1` existe, mais la serie ne converge pas vers elle."""
    with pytest.raises(ValueError, match="serie geometrique divergente"):
        somme_geometrique(ANTIDIAGONALE.subs(X, 2))


def test_l_exponentielle_par_la_serie() -> None:
    """Page 129 : les sommes partielles de `e^A` portent `cosh x` et `sinh x`."""
    somme = par_la_serie(ANTIDIAGONALE, 7)
    pairs = sp.series(sp.cosh(X), X, 0, 8).removeO()
    impairs = sp.series(sp.sinh(X), X, 0, 8).removeO()
    assert _nulle(somme - sp.Matrix([[pairs, impairs], [impairs, pairs]]))


def test_l_exponentielle_par_diagonalisation() -> None:
    """Page 131 : `P = [[1, 1], [1, -1]]`, `D = diag(x, -x)`, `e^A = P e^D P^-1`."""
    attendue = sp.Matrix([[sp.cosh(X), sp.sinh(X)], [sp.sinh(X), sp.cosh(X)]])
    obtenue = par_diagonalisation(ANTIDIAGONALE)
    assert _nulle((obtenue - attendue).applyfunc(lambda e: e.rewrite(sp.exp)))
    passage, diagonale = sp.Matrix([[1, 1], [1, -1]]), sp.diag(X, -X)
    assert passage.inv() == -sp.Matrix([[-1, -1], [-1, 1]]) / 2
    assert passage * diagonale * passage.inv() == ANTIDIAGONALE


def test_les_proprietes_de_l_exponentielle() -> None:
    """7.4.2 : `e^0 = I`, `e^I = e I`, `(e^A)^-1 = e^(-A)`."""
    assert par_diagonalisation(sp.zeros(2, 2)) == sp.eye(2)
    assert par_diagonalisation(sp.eye(2)) == E * sp.eye(2)
    directe = par_diagonalisation(ANTIDIAGONALE)
    opposee = par_diagonalisation(-ANTIDIAGONALE)
    assert _nulle(
        (directe * opposee - sp.eye(2)).applyfunc(lambda e: e.rewrite(sp.exp))
    )


def test_l_exponentielle_d_une_somme_demande_que_les_matrices_commutent() -> None:
    """7.4.2 : `e^A e^B = e^(A+B)` « a condition que `AB = BA` »."""
    a, b = sp.Matrix([[1, 0], [0, 2]]), sp.Matrix([[3, 0], [0, -1]])
    assert a * b == b * a
    assert _nulle(par_jordan(a) * par_jordan(b) - par_jordan(a + b))
    c, d = sp.Matrix([[0, 1], [0, 0]]), sp.Matrix([[0, 0], [1, 0]])
    assert c * d != d * c
    assert not _nulle(par_jordan(c) * par_jordan(d) - par_jordan(c + d))


def test_les_multiplicites_de_l_exemple_de_jordan() -> None:
    """Page 132 : `-2` simple ; `1` de multiplicite algebrique 2, geometrique 1."""
    assert sp.factor(JORDAN.charpoly().as_expr()) == sp.factor(
        (sp.Symbol("lambda") - 1) ** 2 * (sp.Symbol("lambda") + 2)
    )
    assert multiplicites(JORDAN) == {-2: (1, 1), 1: (2, 1)}


def test_la_multiplicite_geometrique_ne_depasse_pas_l_algebrique() -> None:
    """Le syllabus ecrit `g_i >= m_i` page 129 ; son exemple a `g = 1 < m = 2`."""
    for algebrique, geometrique in multiplicites(JORDAN).values():
        assert geometrique <= algebrique
    assert multiplicites(JORDAN)[1] == (2, 1)


def test_des_espaces_propres_distincts_ne_sont_pas_toujours_orthogonaux() -> None:
    """Page 129 l'affirme ; ici `v_1 = (0, 1, 0)` et `v_2 = (3/2, 1, 0)` ne le sont pas.

    Ils sont lineairement independants -- c'est cela qui est toujours vrai ;
    l'orthogonalite demande une matrice symetrique.
    """
    v1, v2 = sp.Matrix([0, 1, 0]), sp.Matrix([R(3, 2), 1, 0])
    assert JORDAN * v1 == -2 * v1
    assert JORDAN * v2 == v2
    assert v1.dot(v2) == 1
    assert sp.Matrix.hstack(v1, v2).rank() == 2
    symetrique = sp.Matrix([[2, 1], [1, 2]])
    (_, _, (premier,)), (_, _, (second,)) = symetrique.eigenvects()
    assert premier.dot(second) == 0


def test_le_vecteur_propre_generalise_de_l_exemple() -> None:
    """`(A - I) v_3 = v_2` : `v_3 = (1/2, -1/2, 3/2)` convient."""
    v2, v3 = sp.Matrix([R(3, 2), 1, 0]), sp.Matrix([R(1, 2), -R(1, 2), R(3, 2)])
    assert (JORDAN - sp.eye(3)) * v3 == v2
    obtenu = vecteur_generalise(JORDAN, 1, v2)
    assert (JORDAN - sp.eye(3)) * obtenu == v2
    assert (JORDAN - sp.eye(3)) ** 2 * obtenu == sp.zeros(3, 1)


def test_la_forme_normale_de_jordan_de_l_exemple() -> None:
    """`A = P J P^-1` avec le `P` et le `J` de la page 133."""
    assert PASSAGE * BLOCS * PASSAGE.inv() == JORDAN
    passage, blocs = forme_de_jordan(JORDAN)
    assert passage * blocs * passage.inv() == JORDAN
    assert sorted(blocs.diagonal()) == [-2, 1, 1]
    assert sum(blocs[i, i + 1] for i in range(2)) == 1


def test_l_inverse_de_p_imprime_page_133_est_faux() -> None:
    """Le syllabus donne `P^-1 = [[-6, 5, 5], ...] / 9` ; c'est 9 et non 5.

    Avec la matrice imprimee, `P P^-1` porte `5/9` au lieu de 1, et `P J P^-1`
    ne redonne pas `A`.
    """
    imprimee = sp.Matrix([[-6, 5, 5], [6, 0, -2], [0, 0, 6]]) / 9
    juste = sp.Matrix([[-6, 9, 5], [6, 0, -2], [0, 0, 6]]) / 9
    assert PASSAGE.inv() == juste
    assert PASSAGE * imprimee == sp.diag(1, R(5, 9), 1)
    assert PASSAGE * BLOCS * imprimee != JORDAN


def test_l_exponentielle_d_un_bloc_de_jordan() -> None:
    """Page 132 : `e^lambda` fois la matrice triangulaire des `1/k!`."""
    lam = sp.Symbol("lambda")
    bloc = exponentielle_d_un_bloc(lam, 4)
    attendu = sp.exp(lam) * sp.Matrix(
        [
            [1, 1, R(1, 2), R(1, 6)],
            [0, 1, 1, R(1, 2)],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    assert bloc == attendu
    assert par_jordan(BLOCS) == sp.Matrix([[E**-2, 0, 0], [0, E, E], [0, 0, E]])


def test_l_exponentielle_de_l_exemple_de_jordan() -> None:
    """`e^A = P e^J P^-1` ; l'entree centrale vaut `e^-2`, et non `5 e^-2 / 9`.

    Second chemin : `A (0, 1, 0) = -2 (0, 1, 0)`, donc `e^A (0, 1, 0) =
    e^-2 (0, 1, 0)` -- la deuxieme colonne ne peut etre que `(0, e^-2, 0)`.
    """
    attendue = sp.Matrix(
        [
            [E, 0, E],
            [R(2, 3) * (E - E**-2), E**-2, (E + 5 * E**-2) / 9],
            [0, 0, E],
        ]
    )
    assert _nulle(par_jordan(JORDAN) - attendue)
    assert _nulle(par_jordan(JORDAN)[:, 1] - sp.Matrix([0, E**-2, 0]))
    assert par_jordan(JORDAN)[1, 1] != R(5, 9) * E**-2


def test_l_exponentielle_par_jordan_s_accorde_avec_la_serie() -> None:
    """Troisieme chemin : quarante termes de la serie, a `10^-20` pres."""
    ecart = par_jordan(JORDAN) - par_la_serie(JORDAN, 40)
    assert all(abs(sp.N(e, 30)) < R(1, 10**20) for e in ecart)


def test_une_matrice_non_diagonalisable_est_refusee_par_la_diagonalisation() -> None:
    with pytest.raises(ValueError, match="matrice non diagonalisable"):
        par_diagonalisation(JORDAN)


def test_l_exponentielle_de_at_est_celle_des_systemes_differentiels() -> None:
    """Lien avec le chapitre 6 : `e^(At)` par Jordan et par `SystemeDifferentiel`."""
    matrice = sp.Matrix([[1, -1], [-4, 1]])
    cote_systemes = SystemeDifferentiel(matrice).exponentielle()
    assert _nulle(par_jordan(matrice * T) - cote_systemes)


def test_une_valeur_propre_de_module_un_fait_diverger_la_serie() -> None:
    """`A = (-1)` : `(I - A)^-1 = 1/2` existe, mais `1 - 1 + 1 - ...` diverge."""
    with pytest.raises(ValueError, match="serie geometrique divergente"):
        somme_geometrique(sp.Matrix([[-1]]))


@pytest.mark.parametrize(("double", "simple"), [(1, 3), (3, 1)])
def test_un_bloc_double_suivi_ou_precede_d_un_bloc_simple(
    double: int, simple: int
) -> None:
    """Quel que soit l'ordre des blocs, chacun prend sa propre valeur propre."""
    matrice = sp.Matrix([[double, 1, 0], [0, double, 0], [0, 0, simple]])
    attendue = sp.Matrix(
        [[E**double, E**double, 0], [0, E**double, 0], [0, 0, E**simple]]
    )
    assert _nulle(par_jordan(matrice) - attendue)
