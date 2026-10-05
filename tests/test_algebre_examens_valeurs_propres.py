"""Syllabus d'algebre, chapitre X, section 8 : les examens 19 a 40.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 174 a 181, et
les solutions des pages 183 a 186. Vingt-deux questions d'examen (janvier
92 a janvier 05) : reduire au maximum, discuter selon un parametre,
determiner des parametres par une condition sur les valeurs propres.
Chaque reponse officielle est recalculee.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.diagonalisation import (
    est_diagonalisable,
    puissance,
)
from polytech.maths.domain.algebre.jordan import multiplicites
from polytech.maths.domain.algebre.matrices_particulieres import trace
from polytech.maths.domain.algebre.reduction import (
    Nature,
    nature_de_la_reduction,
    reduire_au_maximum,
)
from polytech.maths.domain.algebre.spectre import INCONNUE, Spectre
from polytech.maths.domain.algebre.triangularisation import (
    est_triangulaire_superieure,
    semblable_par,
)

M = sp.Matrix
R = sp.Rational
T = INCONNUE
A, B, L = sp.symbols("a b l")
DIAG, TRI, NI = (
    Nature.DIAGONALISABLE,
    Nature.TRIANGULARISABLE,
    Nature.NI_L_UN_NI_L_AUTRE,
)


def _reduite(matrice: sp.Matrix, nature: Nature, valeurs: dict[object, int]) -> int:
    """Verifie la nature et les valeurs propres ; rend le nombre d'etapes."""
    assert nature_de_la_reduction(matrice) is nature
    assert matrice.eigenvals() == valeurs
    reduction = reduire_au_maximum(matrice)
    assert semblable_par(matrice, reduction.passage) == reduction.reduite
    assert est_triangulaire_superieure(reduction.reduite)
    return reduction.etapes


def _identifie(matrice: sp.Matrix, cible: sp.Expr, inconnues: list[sp.Symbol]) -> list:
    """Les inconnues pour lesquelles le polynome caracteristique est la cible."""
    ecart = sp.Poly(Spectre(matrice).polynome_caracteristique() - cible, T)
    return sp.solve(ecart.coeffs(), inconnues, dict=True)


def test_l_examen_19_janvier_92() -> None:
    """Ordre 5 : triangularisable, `-2` double et `2` triple."""
    a = M(
        [
            [2, 1, 0, 1, -4],
            [0, 0, 0, 2, -2],
            [0, 2, 2, 0, 0],
            [0, 2, 0, 0, 2],
            [0, 0, 0, 0, -2],
        ]
    )
    assert _reduite(a, TRI, {-2: 2, 2: 3}) in {1, 2}


def test_les_parametres_de_l_examen_20_janvier_93() -> None:
    """`-1` double et une autre valeur propre positive : `alpha = 0`, `beta = -1`."""
    matrice = M([[A, 5, 7], [0, 3, 4], [-1, B, -4]])
    solutions = _identifie(matrice, -((T + 1) ** 2) * (T - L), [A, B, L])
    positives = [s for s in solutions if s[L] > 0]
    assert positives == [{A: 0, B: -1, L: 1}]
    assert _reduite(matrice.subs({A: 0, B: -1}), TRI, {-1: 2, 1: 1}) == 1


def test_l_examen_21_aout_93() -> None:
    """Triangularisable, 1 valeur propre triple."""
    a = M([[4, 5, -2], [-2, -2, 1], [-1, -1, 1]])
    assert _reduite(a, TRI, {1: 3}) in {1, 2}


def test_l_examen_22_aout_94() -> None:
    """Diagonalisable car la somme des `m_g` vaut 4 : valeurs 1, 1, -1, -2."""
    a = M([[0, 0, 2, 0], [1, 0, 1, 0], [0, 1, -2, 0], [0, 0, 0, 1]])
    assert _reduite(a, DIAG, {1: 2, -1: 1, -2: 1}) == 1
    assert sum(geometrique for _, geometrique in multiplicites(a).values()) == 4


def _examen_23(a: object) -> sp.Matrix:
    return M([[3 - a, 2 - a, a - 2], [6 - 3 * a, a + 3, -3], [8 - 4 * a, 4, a - 4]])


def test_l_examen_23_janvier_95() -> None:
    """`a = 1` : 1 triple, une etape ; `a = 2` : diagonalisable ; sinon `1, 1, a`."""
    polynome = Spectre(_examen_23(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) ** 2 * (T - A)) == 0
    assert _reduite(_examen_23(1), TRI, {1: 3}) == 1
    assert _reduite(_examen_23(2), DIAG, {1: 2, 2: 1}) == 1
    assert _reduite(_examen_23(5), TRI, {1: 2, 5: 1}) == 1
    assert [a for a in range(-3, 6) if est_diagonalisable(_examen_23(a))] == [2]


def test_la_puissance_de_l_examen_23() -> None:
    """Pour `a = 2`, `A^n` par diagonalisation."""
    n = sp.Symbol("n", integer=True, positive=True)
    officielle = M(
        [
            [1, 0, 0],
            [0, -3 + 2 ** (n + 2), 3 - 3 * 2**n],
            [0, -4 + 2 ** (n + 2), 4 - 3 * 2**n],
        ]
    )
    assert sp.simplify(puissance(_examen_23(2), n) - officielle) == sp.zeros(3, 3)
    assert officielle.subs(n, 3) == _examen_23(2) ** 3


def test_l_examen_24_septembre_95() -> None:
    """Diagonalisable pour tout `a` : `1, 2, a`, une valeur double si `a = 1` ou 2."""

    def matrice(a: object) -> sp.Matrix:
        return M(
            [
                [4 - a, 2 - a, a - 2],
                [6, 4, -3],
                [10 - 2 * a, 6 - 2 * a, 2 * a - 5],
            ]
        )

    polynome = Spectre(matrice(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T - 2) * (T - A)) == 0
    assert _reduite(matrice(1), DIAG, {1: 2, 2: 1}) == 1
    assert _reduite(matrice(2), DIAG, {2: 2, 1: 1}) == 1
    assert _reduite(matrice(7), DIAG, {1: 1, 2: 1, 7: 1}) == 1


def test_l_examen_25_janvier_96() -> None:
    """`-1` double et l'autre valeur propre negative : `alpha = 8`, `beta = -4`."""
    alpha, beta = sp.symbols("alpha beta")
    matrice = M([[beta, 5, 7], [0, 3, alpha], [-1, -1, -4]])
    solutions = _identifie(matrice, -((T + 1) ** 2) * (T - L), [alpha, beta, L])
    negatives = [s for s in solutions if s[L] < 0]
    assert negatives == [{alpha: 8, beta: -4, L: -3}]
    assert _reduite(matrice.subs({alpha: 8, beta: -4}), TRI, {-1: 2, -3: 1}) == 1


def test_l_examen_26_aout_96() -> None:
    """Diagonalisable si `b` hors de 1 et 2, ou si `a = 0` et `b = 1`."""

    def matrice(a: object, b: object) -> sp.Matrix:
        return M([[b, a, a], [-1, 1, -1], [1, 0, 2]])

    polynome = Spectre(matrice(A, B)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T - 2) * (T - B)) == 0
    assert nature_de_la_reduction(matrice(3, 5)) is DIAG
    assert nature_de_la_reduction(matrice(0, 1)) is DIAG
    assert _reduite(matrice(3, 1), TRI, {1: 2, 2: 1}) == 1
    assert _reduite(matrice(3, 2), TRI, {1: 1, 2: 2}) == 1
    assert _reduite(matrice(0, 2), TRI, {1: 1, 2: 2}) == 1


@pytest.mark.parametrize("alpha", [-2, 0, 3])
def test_l_examen_27_janvier_97(alpha: int) -> None:
    """Pour tout `alpha` : `alpha` valeur propre triple, triangularisable."""
    a = M([[alpha + 1, -1, 1], [5, alpha - 4, 3], [7, -5, alpha + 3]])
    assert _reduite(a, TRI, {alpha: 3}) in {1, 2}


def test_l_examen_28_aout_97() -> None:
    """`alpha = 1` : 2 triple ; sinon `2, 2, alpha + 1` ; une etape a chaque fois."""

    def matrice(a: object) -> sp.Matrix:
        return M(
            [
                [3 - a, 1 - a, a - 1],
                [6 - 8 * a, 4 - 4 * a, 5 * a - 3],
                [7 - 9 * a, 3 - 5 * a, 6 * a - 2],
            ]
        )

    polynome = Spectre(matrice(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 2) ** 2 * (T - A - 1)) == 0
    assert _reduite(matrice(1), TRI, {2: 3}) == 1
    assert _reduite(matrice(4), TRI, {2: 2, 5: 1}) == 1


def test_l_examen_29_janvier_98() -> None:
    """Une seule valeur propre triple, `a` et `b` non nuls : `a = 2`, `b = 3`."""
    matrice = M([[A, B, A], [-B, A - B, -1], [B, 2 * B, A + B]])
    solutions = _identifie(matrice, -((T - L) ** 3), [A, B, L])
    non_nulles = [s for s in solutions if s[A] != 0 and s[B] != 0]
    assert non_nulles == [{A: 2, B: 3, L: 2}]
    assert _reduite(matrice.subs({A: 2, B: 3}), TRI, {2: 3}) in {1, 2}


def test_l_examen_30_aout_98() -> None:
    """Non diagonalisable (`m_g(1) != m_a(1)`), triangularisable : 1, 1, -1, -2."""
    a = M([[0, 2, -2, 1], [4, 2, -4, 2], [3, 1, -3, 1], [-3, 2, -2, 0]])
    assert multiplicites(a)[1] == (2, 1)
    assert _reduite(a, TRI, {1: 2, -1: 1, -2: 1}) in {1, 2}


def test_l_examen_31_janvier_99() -> None:
    """Valeurs propres `1, b, b - 1` ; somme double du produit, une double : `b = 2`."""
    matrice = M(
        [
            [B - 1, A + 1, A],
            [B - 2, A + 3, A + 1],
            [2 - B, B - A - 4, B - A - 2],
        ]
    )
    polynome = Spectre(matrice).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T - B) * (T - B + 1)) == 0
    somme_double_du_produit = sp.solve(sp.Eq(trace(matrice), 2 * matrice.det()), B)
    assert somme_double_du_produit == [0, 2]
    assert matrice.subs({A: 4, B: 0}).eigenvals() == {-1: 1, 0: 1, 1: 1}
    for a in (-1, 0, 4):
        assert _reduite(matrice.subs({A: a, B: 2}), TRI, {1: 2, 2: 1}) == 1


def test_l_examen_32_aout_99() -> None:
    """Non diagonalisable (`m_g(2) != m_a(2)`), triangularisable : -1, -1, 2, 2."""
    a = M([[-2, 2, -2, 1], [-9, 8, -6, 3], [-8, 7, -5, 2], [-2, 4, -4, 1]])
    assert multiplicites(a)[2] == (2, 1)
    assert _reduite(a, TRI, {-1: 2, 2: 2}) in {1, 2}


def test_l_examen_33_janvier_00() -> None:
    """Valeurs propres racines de `x^3 - (b+2)x^2 + 8x - 4` : `a = 2`, `b = 3`."""
    matrice = M([[A, 3, B], [-1, 1, B - 4], [A - 1, 0, B - 1]])
    cible = -(T**3 - (B + 2) * T**2 + 8 * T - 4)
    assert _identifie(matrice, cible, [A, B]) == [{A: 2, B: 3}]
    assert _reduite(matrice.subs({A: 2, B: 3}), TRI, {1: 1, 2: 2}) == 1


def test_l_examen_34_aout_00() -> None:
    """Diagonalisable si `a != +-1`, toujours triangularisable."""

    def matrice(a: object) -> sp.Matrix:
        return M([[0, 1, 0], [0, 0, 1], [-a, 1, a]])

    polynome = Spectre(matrice(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T + 1) * (T - A)) == 0
    assert multiplicites(matrice(1)) == {-1: (1, 1), 1: (2, 1)}
    assert multiplicites(matrice(-1)) == {-1: (2, 1), 1: (1, 1)}
    assert _reduite(matrice(1), TRI, {-1: 1, 1: 2}) == 1
    assert _reduite(matrice(-1), TRI, {-1: 2, 1: 1}) == 1
    assert nature_de_la_reduction(matrice(3)) is DIAG
    assert nature_de_la_reduction(matrice(0)) is DIAG


def test_l_examen_35_janvier_01() -> None:
    """Valeurs propres positives de somme des carres 12 : `a = 3`, et 2 triple."""

    def matrice(a: object) -> sp.Matrix:
        return M([[3 * a, a + 1, -1], [4 - 5 * a, -a, 2 - a], [5 - 3 * a, 2 - a, 0]])

    candidats = sp.solve(sp.Eq(trace(matrice(A) ** 2), 12), A)
    positifs = [
        a for a in candidats if all(v.is_nonnegative for v in matrice(a).eigenvals())
    ]
    assert positifs == [3]
    assert _reduite(matrice(3), TRI, {2: 3}) in {1, 2}


def test_l_examen_36_aout_01() -> None:
    """`|a| > 2` : diagonalisable ; `a = +-2` : triangularisable ; sinon aucun."""

    def matrice(a: object) -> sp.Matrix:
        return M([[0, 1, 0], [-1, a, 0], [0, 0, 1]])

    assert nature_de_la_reduction(matrice(3)) is DIAG
    assert nature_de_la_reduction(matrice(-5)) is DIAG
    assert _reduite(matrice(-2), TRI, {-1: 2, 1: 1}) == 1
    assert _reduite(matrice(2), TRI, {1: 3}) == 1
    for a in (-1, 0, 1):
        assert nature_de_la_reduction(matrice(a)) is NI
        reelles = {v: m for v, m in matrice(a).eigenvals().items() if v.is_real}
        assert reelles == {1: 1}
        assert reduire_au_maximum(matrice(a)).etapes == 1


def test_l_examen_37_janvier_02() -> None:
    """1 toujours valeur propre ; les autres `a - 1` et `a + 2` ; `a = 2`."""

    def matrice(a: object) -> sp.Matrix:
        return M(
            [
                [a - 1, 2, 1],
                [2 * a - 3, 4, 1],
                [-2 * a + 3, a - 3, a - 1],
            ]
        )

    polynome = Spectre(matrice(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T - (A - 1)) * (T - (A + 2))) == 0
    positives = [
        a
        for a in range(-4, 8)
        if all(v > 0 for v in matrice(a).eigenvals())
        and not est_diagonalisable(matrice(a))
    ]
    assert positives == [2]
    assert _reduite(matrice(2), TRI, {1: 2, 4: 1}) == 1


def test_l_examen_38_janvier_04() -> None:
    """`a > 0` : aucun ; `a = 0` : triangularisable ; `a < 0` : diagonalisable."""

    def matrice(a: object) -> sp.Matrix:
        return M([[1, 0, 1], [1 + a, -a, 1 + a], [1 + a, -1 - a, a]])

    polynome = Spectre(matrice(A)).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T**2 + A)) == 0
    assert nature_de_la_reduction(matrice(2)) is NI
    assert multiplicites(matrice(0)) == {0: (2, 1), 1: (1, 1)}
    assert nature_de_la_reduction(matrice(0)) is TRI
    assert multiplicites(matrice(-1)) == {-1: (1, 1), 1: (2, 2)}
    assert nature_de_la_reduction(matrice(-1)) is DIAG
    assert matrice(-4).eigenvals() == {1: 1, 2: 1, -2: 1}
    reduction = reduire_au_maximum(matrice(2))
    assert reduction.etapes == 1
    assert reduction.reduite[:, 0] == M([1, 0, 0])


def test_l_examen_39_aout_04() -> None:
    """Pour `a >= 0` : diagonalisable si et seulement si `a != 1`."""

    def matrice(a: object) -> sp.Matrix:
        return M(
            [
                [a, a + 3, 2],
                [-a, a - 3, a - 2],
                [3 * a, -a + 6, 4 - 2 * a],
            ]
        )

    non_diagonalisables = [
        a for a in (0, R(1, 2), 1, 2, 3, 5) if not est_diagonalisable(matrice(a))
    ]
    assert non_diagonalisables == [1]
    assert nature_de_la_reduction(matrice(1)) is TRI
    assert reduire_au_maximum(matrice(1)).etapes == 1


def test_l_examen_40_janvier_05() -> None:
    """La trace vaut `3a` ; deux valeurs egales forcent les trois ; `a = 2`, `b = 3`.

    `a` est valeur propre si et seulement si `b = a + 1` ; les deux autres
    sont alors `a +- racine((a + 1)(a - 2))`, egales pour `a = 2` seulement,
    `a = -1` donnant `b = 0`, exclu. Entre -1 et 2 elles sont complexes : la
    reponse officielle est la seule a trois valeurs propres reelles.
    """

    def matrice(a: object, b: object) -> sp.Matrix:
        return M(
            [
                [a + b + 1, 4 * b + 1, 2 * b],
                [-1, a - b - 1, -b],
                [a, a + b, a],
            ]
        )

    generale = matrice(A, B)
    assert trace(generale) == 3 * A
    polynome = Spectre(generale).polynome_caracteristique()
    assert sp.factor(polynome.subs(T, A)) == -(B**2) * (A - B + 1)
    ecart = sp.Symbol("s")
    decale = sp.expand(polynome.subs({T: A + ecart, B: A + 1}))
    assert sp.expand(decale + ecart * (ecart**2 - (A + 1) * (A - 2))) == 0
    assert sp.solve((A + 1) * (A - 2), A) == [-1, 2]
    assert matrice(5, 6).eigenvals() == {
        5: 1,
        5 - 3 * sp.sqrt(2): 1,
        5 + 3 * sp.sqrt(2): 1,
    }
    officielle = matrice(2, 3)
    assert officielle.eigenvals() == {2: 3}
    assert not est_diagonalisable(officielle)
    assert nature_de_la_reduction(officielle) is TRI
    assert reduire_au_maximum(officielle).etapes == 2
