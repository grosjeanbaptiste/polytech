"""Syllabus d'algebre, chapitre VIII, sections 5 a 7 : matrices et changements de base.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 105 a 116, et
les solutions des pages 125 a 128. La matrice associee a une application
lineaire dans deux bases, la matrice de passage, la formule
`A_(v,w) = P_(e',w) A_(e,e') P_(v,e)`, les exemples 1 a 5, puis les
reponses officielles des exercices 14 a 32.

Trois reponses dont le texte extrait est illisible (23, 26, 30) ont ete
relues sur la page rendue en image.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.espaces_de_fonctions import (
    coordonnees_d_une_fonction,
    matrice_d_un_operateur,
)
from polytech.maths.domain.algebre.familles_de_vecteurs import coordonnees
from polytech.maths.domain.algebre.isomorphismes import (
    vecteur_d_un_polynome,
    vecteur_d_une_matrice,
)
from polytech.maths.domain.algebre.representation import (
    matrice_associee,
    matrice_canonique,
    matrice_de_passage,
)
from polytech.maths.domain.algebre.sous_espaces import sont_supplementaires
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741 -- l'unite imaginaire
A = sp.Symbol("a")
CANONIQUE_1 = M([[1, 0, 1, 0, 0], [0, 1, -1, 0, 1], [1, 0, 3, 0, 1], [0, 0, 0, 1, 0]])
CANONIQUE_2 = M([[1, A, A], [0, A - 1, 0], [-1, -1, 2], [1, 1, A]])
V_2 = ((1, 0, 0), (1, 1, 0), (1, 1, 1))
ASSOCIEE_2 = M(
    [
        [2, A + 3, 3 * (A + 1)],
        [-1, -2, -2 * (2 + A)],
        [-1, -2, 0],
        [-1, -2, -(2 + A)],
    ]
)


def _polynomes(*polynomes: sp.Expr, degre: int) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(vecteur_d_un_polynome(p, degre) for p in polynomes)


def _matrices(*matrices: list[list[object]]) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(vecteur_d_une_matrice(M(m)) for m in matrices)


def _canon(dimension: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension))


V_1 = _polynomes(1 - X, X + X**3, 1 - X + X**2 + X**4, X**3, X**4, degre=4)
W_1 = _matrices([[1, 0], [0, 0]], [[2, 1], [-1, 0]], [[1, 0], [2, 1]], [[1, 0], [1, 1]])
W_2 = _polynomes(1 + X, X, X + X**2, 1 - X**3, degre=3)


def test_la_matrice_de_l_exemple_1_dans_les_bases_v_et_w() -> None:
    """Page 108 : les cinq colonnes, coordonnees des `f(v_i)` dans `w`."""
    associee = matrice_associee(CANONIQUE_1, V_1, W_1)
    assert associee == M(
        [
            [3, -3, 4, -1, -2],
            [-1, 1, -1, 0, 1],
            [0, 0, 4, -1, 2],
            [0, 1, -4, 2, -2],
        ]
    )


def test_une_colonne_de_l_exemple_1_par_le_calcul_direct() -> None:
    """`f(X + X^3) = [[0, 1], [0, 1]]`, de coordonnees `(-3, 1, 0, 1)` dans `w`."""
    image = ApplicationLineaire(CANONIQUE_1).image_de(V_1[1])
    assert image == (0, 1, 0, 1)
    assert coordonnees(image, W_1) == (-3, 1, 0, 1)


def test_la_matrice_de_l_exemple_2_dans_les_bases_v_et_w() -> None:
    """Page 109, calcul direct : `f(1, 0, 0)` se lit `(2, -1, -1, -1)` dans `w`."""
    f = ApplicationLineaire(CANONIQUE_2)
    assert coordonnees(f.image_de(V_2[0]), W_2) == (2, -1, -1, -1)
    associee = matrice_associee(CANONIQUE_2, V_2, W_2)
    assert sp.expand(associee - ASSOCIEE_2) == sp.zeros(4, 3)


def test_l_exemple_3_retrouve_l_application() -> None:
    """`A_(v,w)` donnee : `f(x, y, z) = (x, x, x, x + y + z)`."""
    v = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
    w = ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1))
    associee = M([[1, 1, 1], [0, 0, 0], [1, 1, 1], [2, 1, 0]])
    x, y, z = sp.symbols("x y z")
    assert coordonnees((x, y, z), v) == (z, y - z, x - y)
    assert associee * M([z, y - z, x - y]) == M([x, 0, x, z + y])
    f = ApplicationLineaire(matrice_canonique(associee, v, w))
    assert f.image_de((x, y, z)) == (x, x, x, x + y + z)


def test_les_matrices_de_passage_de_l_exemple_4() -> None:
    """`P_(v,e)`, `P_(e',w)`, et leur produit avec `A_(e,e')` redonne `A_(v,w)`."""
    passage_v = matrice_de_passage(V_2, _canon(3))
    assert passage_v == M([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    passage_w = matrice_de_passage(_canon(4), W_2)
    assert passage_w == M([[1, 0, 0, 1], [-1, 1, -1, -1], [0, 0, 1, 0], [0, 0, 0, -1]])
    assert coordonnees(vecteur_d_un_polynome(X**2, 3), W_2) == (0, -1, 1, 0)
    produit = passage_w * CANONIQUE_2 * passage_v
    assert sp.expand(produit - ASSOCIEE_2) == sp.zeros(4, 3)


def test_la_matrice_de_passage_de_l_exemple_5() -> None:
    """Les colonnes de `P_(v,e)` sont les coordonnees des `v_i` dans `e`."""
    passage = matrice_de_passage(V_1, _canon(5))
    assert passage == M(
        [
            [1, 0, 1, 0, 0],
            [-1, 1, -1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ]
    )


def test_une_matrice_de_passage_demande_deux_bases() -> None:
    with pytest.raises(ValueError, match="n'est pas une base"):
        matrice_de_passage(((1, 0), (2, 0)), _canon(2))
    with pytest.raises(ValueError, match="n'est pas une base"):
        matrice_de_passage(_canon(2), ((1, 1), (2, 2)))


def test_les_reponses_des_exercices_14_et_15() -> None:
    """14 : `(5, 3)` et `(11, 5, -1)` ; 15 : une projection et `7 I`."""
    assert ApplicationLineaire(M([[2, 1], [1, 0]])).image_de((3, -1)) == (5, 3)
    g = ApplicationLineaire(M([[2, 1], [1, 0], [-1, 4]]))
    assert g.image_de((5, 1)) == (11, 5, -1)
    projection = ApplicationLineaire.depuis_la_regle(lambda a, b, _c, _d: (a, b), 4)
    assert projection.matrice == M([[1, 0, 0, 0], [0, 1, 0, 0]])
    homothetie = ApplicationLineaire.depuis_la_regle(
        lambda *x: tuple(7 * c for c in x), 4
    )
    assert homothetie.matrice == 7 * sp.eye(4)


def test_la_reponse_de_l_exercice_16() -> None:
    """a) sur `R^(2x2)` ; b) `p -> p(X + 1)` sur `R[X]<=2`."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (a + b, b + c, c + d, d + a), 4
    )
    assert f.matrice == M([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    colonnes = [vecteur_d_un_polynome(sp.expand((X + 1) ** k), 2) for k in range(3)]
    decalage = M(colonnes).T
    assert decalage == M([[1, 1, 1], [0, 1, 2], [0, 0, 1]])


def test_la_reponse_de_l_exercice_17() -> None:
    """`f(x, y, z) = (x + y, x - z)` dans `v` et `w` : `[[1, 2, 2], [0, -3, -2]]`."""
    f = ApplicationLineaire.depuis_la_regle(lambda x, y, z: (x + y, x - z), 3)
    v = ((1, 1, 0), (0, 1, 1), (1, 1, 1))
    w = ((2, 1), (1, 1))
    assert matrice_associee(f.matrice, v, w) == M([[1, 2, 2], [0, -3, -2]])


@pytest.mark.parametrize(
    ("base", "attendue"),
    [
        ((sp.exp(T), sp.exp(2 * T)), [[1, 0], [0, 2]]),
        ((sp.Integer(1), T), [[0, 1], [0, 0]]),
        ((sp.exp(T), T * sp.exp(T)), [[1, 1], [0, 1]]),
        ((sp.Integer(1), T, T**2), [[0, 1, 0], [0, 0, 2], [0, 0, 0]]),
        ((sp.sin(T), sp.cos(T)), [[0, -1], [1, 0]]),
    ],
    ids=["a", "b", "c", "d", "e"],
)
def test_la_reponse_de_l_exercice_18(
    base: tuple[sp.Expr, ...], attendue: list[list[int]]
) -> None:
    """La matrice de la derivation dans cinq bases d'espaces de fonctions."""
    derivation = matrice_d_un_operateur(lambda f: sp.diff(f, T), base, T)
    assert derivation == M(attendue)


def test_une_fonction_hors_de_l_espace_n_a_pas_de_coordonnees() -> None:
    """`t^2` n'est pas combinaison de 1 et de `t`."""
    with pytest.raises(ValueError, match="hors de l'espace engendre"):
        coordonnees_d_une_fonction(T**2, (sp.Integer(1), T), T)
    with pytest.raises(ValueError, match="ne separent pas la base"):
        coordonnees_d_une_fonction(T, (T, T**2), T)
    assert coordonnees_d_une_fonction(T, (T, T**2), T, point=1) == (1, 0)


def test_un_operateur_dans_une_base_que_l_origine_ne_separe_pas() -> None:
    """`f -> t f'` dans `{t, t^2}` : diagonale `1, 2`, a lire en `t = 1` et non en 0."""

    def euler(f: sp.Expr) -> sp.Expr:
        return T * sp.diff(f, T)

    assert matrice_d_un_operateur(euler, (T, T**2), T, point=1) == sp.diag(1, 2)
    with pytest.raises(ValueError, match="ne separent pas la base"):
        matrice_d_un_operateur(euler, (T, T**2), T)


def test_la_reponse_de_l_exercice_19() -> None:
    """`A -> MA`, `A -> AM`, `A -> MA - AM` dans la base canonique de `R^(2x2)`."""
    a, b, c, d = sp.symbols("a b c d")
    m = M([[a, b], [c, d]])
    unites = [M(2, 2, [int(i == j) for j in range(4)]) for i in range(4)]

    def matrice(regle: object) -> sp.Matrix:
        return M([vecteur_d_une_matrice(regle(u)) for u in unites]).T  # type: ignore[operator]

    assert matrice(lambda u: m * u) == M(
        [[a, 0, b, 0], [0, a, 0, b], [c, 0, d, 0], [0, c, 0, d]]
    )
    assert matrice(lambda u: u * m) == M(
        [[a, c, 0, 0], [b, d, 0, 0], [0, 0, a, c], [0, 0, b, d]]
    )
    assert matrice(lambda u: m * u - u * m) == M(
        [[0, -c, b, 0], [-b, a - d, 0, b], [c, 0, d - a, -c], [0, c, -b, 0]]
    )


def test_la_reponse_de_l_exercice_20() -> None:
    """`P_(v,v')`, `P_(v',v)`, et les coordonnees de `(1, 0, 0)` dans les deux bases."""
    v = ((1, 1, 0), (-1, 1, 1), (0, 1, 2))
    v_prime = ((2, 1, 1), (0, 0, 1), (-1, 1, 1))
    aller = matrice_de_passage(v, v_prime)
    assert aller == M([[2, 0, 1], [-3, 0, 3], [1, 3, 2]]) / 3
    retour = matrice_de_passage(v_prime, v)
    assert retour == M([[3, -1, 0], [-3, -1, 3], [3, 2, 0]]) / 3
    assert aller * retour == sp.eye(3)
    dans_v = coordonnees((1, 0, 0), v)
    assert dans_v == (R(1, 3), -R(2, 3), R(1, 3))
    assert tuple(aller * M(dans_v)) == (R(1, 3), 0, -R(1, 3))
    assert coordonnees((1, 0, 0), v_prime) == (R(1, 3), 0, -R(1, 3))


def test_la_reponse_de_l_exercice_21() -> None:
    """La primitive `R[X]<=3 -> R[X]<=4`, dans les bases canoniques puis `v` et `v'`."""
    colonnes = [vecteur_d_un_polynome(sp.integrate(X**k, X), 4) for k in range(4)]
    canonique = M(colonnes).T
    assert canonique == M(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, R(1, 2), 0, 0],
            [0, 0, R(1, 3), 0],
            [0, 0, 0, R(1, 4)],
        ]
    )
    v = _polynomes(sp.Integer(1), 1 + X, 1 + X + X**2, X**2 + X**3, degre=3)
    v_prime = _polynomes(
        sp.Integer(1), 1 + X, 1 + X + X**2, X**2 + X**3, X**3 + X**4, degre=4
    )
    assert matrice_de_passage(v, _canon(4)) == M(
        [[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    )
    assert matrice_de_passage(v_prime, _canon(5)) == M(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )
    assert matrice_associee(canonique, v, v_prime) == M(
        [
            [-1, -1, -1, 0],
            [1, R(1, 2), R(5, 6), R(1, 12)],
            [0, R(1, 2), R(1, 6), -R(1, 12)],
            [0, 0, R(1, 3), R(1, 12)],
            [0, 0, 0, R(1, 4)],
        ]
    )


def test_la_reponse_de_l_exercice_22() -> None:
    """La derivation dans `{e^t, ..., t^3 e^t}`, son inverse, puis `int t^3 e^t`."""
    base = tuple(T**k * sp.exp(T) for k in range(4))
    derivation = matrice_d_un_operateur(lambda f: sp.diff(f, T), base, T)
    assert derivation == M([[1, 1, 0, 0], [0, 1, 2, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
    inverse = derivation.inv()
    assert inverse == M([[1, -1, 2, -6], [0, 1, -2, 6], [0, 0, 1, -3], [0, 0, 0, 1]])
    primitive = sum(c * b for c, b in zip(inverse.col(3), base, strict=True))
    attendue = (-6 + 6 * T - 3 * T**2 + T**3) * sp.exp(T)
    assert sp.expand(primitive - attendue) == 0
    assert sp.simplify(sp.diff(primitive, T) - T**3 * sp.exp(T)) == 0


def test_la_reponse_de_l_exercice_23() -> None:
    """`P_(w,v)` dans `C^(2x2)`, relue sur l'image de la page 126."""
    v = _matrices(
        [[1, 0], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [0, 1]], [[1, 1], [0, 0]]
    )
    w = _matrices(
        [[I, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [I, 0]], [[0, 0], [0, 1]]
    )
    demi, i_demi = R(1, 2), I / 2
    assert matrice_de_passage(w, v) == M(
        [
            [i_demi, -demi, -i_demi, demi],
            [0, 0, I, 0],
            [-i_demi, demi, i_demi, demi],
            [i_demi, demi, -i_demi, -demi],
        ]
    )


def test_la_reponse_de_l_exercice_24() -> None:
    """Les colonnes de `P_(v,w)` ecrivent les `v_i` dans `w` ; l'inverse, les `w_i`."""
    passage = M([[2, -1, 3], [1, 1, 4], [0, 1, 2]])
    assert tuple(passage.col(0)) == (2, 1, 0)
    assert tuple(passage.col(1)) == (-1, 1, 1)
    assert tuple(passage.col(2)) == (3, 4, 2)
    assert passage.inv() == M([[-2, 5, -7], [-2, 4, -5], [1, -2, 3]])


def test_la_reponse_de_l_exercice_25() -> None:
    """La derivee `P2 -> P1` : canonique, deux passages, puis dans `v'` et `w'`."""
    canonique = M([[0, 1, 0], [0, 0, 2]])
    v_prime = _polynomes(X**2 - 2 * X + 5, 2 * X**2 - 3 * X, X + 3, degre=2)
    w_prime = _polynomes(-2 + 2 * X, -3 + 4 * X, degre=1)
    assert matrice_de_passage(v_prime, _canon(3)) == M(
        [[5, 0, 3], [-2, -3, 1], [1, 2, 0]]
    )
    assert matrice_de_passage(w_prime, _canon(2)) == M([[-2, -3], [2, 4]])
    assert matrice_associee(canonique, v_prime, w_prime) == M([[1, 0, -2], [0, 1, 1]])


def test_la_reponse_de_l_exercice_26() -> None:
    """`A_(v,v) = diag(1, 1, -3)`, et `A_(e,e)` relue sur l'image de la page 127."""
    v = ((0, 2, 1), (1, 0, 3), (0, -1, 1))
    canonique = matrice_canonique(sp.diag(1, 1, -3), v, v)
    assert canonique == M([[1, 0, 0], [-8, -R(1, 3), R(8, 3)], [8, R(4, 3), -R(5, 3)]])
    f = ApplicationLineaire(canonique)
    assert f.image_de(v[0]) == v[0]
    assert f.image_de(v[2]) == (0, 3, -3)


def test_la_reponse_de_l_exercice_27() -> None:
    """Noyau de dimension 2, rang 2, et `W = <e3, e4>` supplementaire du noyau."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (a - b + c + d, a + 2 * c - d, a + b + 3 * c - 3 * d), 4
    )
    assert f.matrice == M([[1, -1, 1, 1], [1, 0, 2, -1], [1, 1, 3, -3]])
    assert f.noyau() == ((-2, -1, 1, 0), (1, 2, 0, 1))
    assert f.rang() == 2
    assert sont_supplementaires(f.noyau(), ((0, 0, 1, 0), (0, 0, 0, 1)))


def test_la_reponse_de_l_exercice_28() -> None:
    """Quatre matrices canoniques, puis sommes, multiples et composees."""
    f1 = M([[1, 0], [0, 3], [-1, 1], [1, 1]])
    f2 = M([[1, 1], [0, 0], [0, 0], [0, 0]])
    f3 = M([[1, 1, 0, 0], [0, 0, 1, 1]])
    f4 = M([[0, 0, 0, 0], [0, 0, 1, 0]])
    regle = ApplicationLineaire.depuis_la_regle(
        lambda a, b: (a, 3 * b, b - a, a + b), 2
    )
    assert regle.matrice == f1
    assert f1 + f2 == M([[2, 1], [0, 3], [-1, 1], [1, 1]])
    assert 2 * f2 == M([[2, 2], [0, 0], [0, 0], [0, 0]])
    assert 3 * f4 - f3 == M([[-1, -1, 0, 0], [0, 0, 2, -1]])
    assert f4 * f1 == M([[0, 0], [-1, 1]])
    assert f2 * f3 == M([[1, 1, 1, 1]]).col_join(sp.zeros(3, 4))
    assert f3 * f2 * f4 == M([[0, 0, 1, 0], [0, 0, 0, 0]])


def test_la_reponse_de_l_exercice_29() -> None:
    """Les deux projections sur `<v1, v2>` et `<v3>`, dans `v` puis dans `e`."""
    v = ((1, 0, 1), (0, 1, 1), (0, 0, 1))
    assert matrice_de_passage(v, _canon(3)) == M([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    assert matrice_de_passage(_canon(3), v) == M([[1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    f = matrice_canonique(sp.diag(1, 1, 0), v, v)
    g = matrice_canonique(sp.diag(0, 0, 1), v, v)
    assert f == M([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert g == M([[0, 0, 0], [0, 0, 0], [-1, -1, 1]])
    assert f + g == sp.eye(3)
    assert f * f == f
    assert f * g == sp.zeros(3, 3)


def test_la_reponse_de_l_exercice_30() -> None:
    """a) `A_(v,v')` ; b) les `P` et `Q` officiels reduisent bien `A` a `[I | 0]`."""
    canonique = M([[2, 5, -3], [1, -4, 7]])
    v = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
    v_prime = ((1, 3), (2, 5))
    assert matrice_associee(canonique, v, v_prime) == M([[-12, -41, -8], [8, 24, 5]])
    p = M([[1, 0, -23], [0, 1, 17], [0, 0, 13]])
    q = M([[4, 5], [1, -2]]) / 13
    assert q * canonique * p == M([[1, 0, 0], [0, 1, 0]])
    assert p.det() != 0
    assert q.det() != 0


def test_la_reponse_de_l_exercice_31() -> None:
    """`f(x, y, z) = (3x - 2y + z, x + 3y, x - y + z)`."""
    v = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
    w = ((1, -1, 0), (2, -1, -2), (1, -1, -2))
    associee = M([[2, 1, 0], [-4, -2, -2], [7, 7, 5]])
    f = ApplicationLineaire(matrice_canonique(associee, w, v))
    x, y, z = sp.symbols("x y z")
    assert f.image_de((x, y, z)) == (3 * x - 2 * y + z, x + 3 * y, x - y + z)


def test_la_reponse_de_l_exercice_32() -> None:
    """Deux images et un noyau determinent `f` : la matrice officielle."""
    noyau = ((1, 1, 0, 1), (0, -1, 1, 0))
    antecedents = (*noyau, (0, 0, 1, 0), (0, 0, 0, 1))
    images = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1))
    f = ApplicationLineaire.depuis_les_images(antecedents, images)
    assert f.matrice == M([[-1, 0, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 0], [-1, 0, 0, 1]])
    for vecteur in noyau:
        assert vecteur[0] - vecteur[1] - vecteur[2] == 0
        assert vecteur[0] - vecteur[3] == 0
    assert len(f.noyau()) == 2
