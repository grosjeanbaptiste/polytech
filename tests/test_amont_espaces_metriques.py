"""Mathematiques I et II, algebre : les espaces metriques, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire1.toml`
et `AlgebreLineaire2.toml` rattachent un titre portent ce que la planche
enonce -- en recopiant ses exemples, page citee.

Source : `AlgebreLineaire2/SLIDES/Slides_06_Espaces_metriques.pdf`, que
`AlgebreLineaire1/SLIDES/` porte a l'identique (meme empreinte). Les pages
sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre import metrique, orthogonalisation
from polytech.maths.domain.algebre import produits_scalaires as produits
from polytech.maths.domain.algebre.moindres_carres import au_sens_des_moindres_carres

X_DU_PLAN, Y_DU_PLAN = (4.0, 2.0), (-2.0, -2.0)
"""Pages 70 a 73 (28 a 30) : l'illustration dans le plan."""

W1, W2 = sp.Matrix([1, 1, 0]), sp.Matrix([0, 1, 1])
V2 = sp.Matrix([-1, 1, 2]) / 2
TROIS = sp.Matrix([3, 3, 3])
"""Page 96 (36) : `W`, sa base, le second vecteur de Gram-Schmidt, et `X`."""

T = sp.Symbol("X", real=True)


def test_le_produit_la_norme_et_la_distance_de_l_illustration() -> None:
    """Page 71 (28) : `-12`, `2 racine(5)`, `2 racine(2)` et `2 racine(13)`."""
    assert metrique.produit_scalaire(X_DU_PLAN, Y_DU_PLAN) == pytest.approx(-12)
    assert metrique.norme(X_DU_PLAN) == pytest.approx(2 * 5**0.5)
    assert metrique.norme(Y_DU_PLAN) == pytest.approx(2 * 2**0.5)
    assert metrique.distance(X_DU_PLAN, Y_DU_PLAN) == pytest.approx(2 * 13**0.5)


def test_la_projection_et_gram_schmidt_de_l_illustration() -> None:
    """Pages 71 et 72 (28 et 29) : `proj_Y X = (3, 3)`, `v_2 = (2/5, -4/5)`."""
    assert metrique.projection_orthogonale(X_DU_PLAN, (Y_DU_PLAN,)) == pytest.approx(
        (3, 3)
    )
    v1, v2 = metrique.gram_schmidt((X_DU_PLAN, Y_DU_PLAN))
    assert v1 == X_DU_PLAN
    assert v2 == pytest.approx((2 / 5, -4 / 5))
    assert metrique.est_orthogonale((v1, v2))
    assert not metrique.est_orthonormale((v1, v2))


def test_les_produits_scalaires_des_pages_15_et_16() -> None:
    """Pages 15 et 16 (12 et 13) : usuel, hermitien, trace, integrale, points."""
    assert produits.usuel(sp.Matrix([1, 2]), sp.Matrix([3, 4])) == 11
    assert produits.usuel(sp.Matrix([sp.I, 1]), sp.Matrix([sp.I, 1])) == 2
    a, b = sp.Matrix([[1, 2], [3, 4]]), sp.Matrix([[0, 1], [1, 0]])
    assert produits.de_la_trace(a, b) == (a.T * b).trace() == 5
    assert produits.integral(T, -1, 1)(T, T) == sp.Rational(2, 3)
    assert produits.aux_points(T, (-2, -1, 0, 1, 2))(T, T) == 10
    assert produits.norme(sp.Matrix([3, 4]), produits.usuel) == 5


def test_gram_schmidt_et_la_projection_sur_w_de_la_page_96() -> None:
    """Page 96 (36) : `v_2 = (-1/2, 1/2, 1)`, puis `proj_W (3, 3, 3) = (2, 4, 2)`."""
    base = orthogonalisation.gram_schmidt((W1, W2), produits.usuel)
    assert base == (W1, V2)
    assert produits.est_orthogonale(base, produits.usuel)
    projete = orthogonalisation.projection(TROIS, base, produits.usuel)
    assert projete == sp.Matrix([2, 4, 2])
    unitaires = orthogonalisation.normaliser(base, produits.usuel)
    assert [produits.norme(u, produits.usuel) for u in unitaires] == [1, 1]


def test_la_meilleure_approximation_est_la_projection() -> None:
    """Page 104 (39) : `||X - proj_W X|| = min ||X - Y||` ; ici `racine(3)`."""
    base = (W1, V2)
    assert orthogonalisation.distance_au_sous_espace(
        TROIS, base, produits.usuel
    ) == sp.sqrt(3)
    en_flottants = tuple(tuple(float(c) for c in v) for v in base)
    proche = metrique.distance_au_sous_espace((3.0, 3.0, 3.0), en_flottants)
    assert proche == pytest.approx(3**0.5)


def test_l_orthogonal_de_w_est_une_droite() -> None:
    """Pages 77 et 84 (31 et 33) : `W^perp`, supplementaire orthogonal de `W`."""
    canonique = tuple(sp.Matrix(colonne) for colonne in sp.eye(3).columnspace())
    (normale,) = orthogonalisation.orthogonal((W1, W2), canonique, produits.usuel)
    assert produits.usuel(normale, W1) == produits.usuel(normale, W2) == 0
    assert normale.cross(sp.Matrix([1, -1, 1])).is_zero_matrix


@pytest.mark.parametrize(
    ("matrice", "second_membre", "solution"),
    [
        (sp.Matrix([[1, 2], [3, 4], [2, 1]]), sp.Matrix([5, 11, 4]), (1, 2)),
        (sp.Matrix([[1, 0], [1, 1], [0, 1]]), TROIS, (2, 2)),
    ],
    ids=["page 106, compatible", "page 110, impossible"],
)
def test_la_solution_au_sens_des_moindres_carres(
    matrice: sp.Matrix, second_membre: sp.Matrix, solution: tuple[int, int]
) -> None:
    """Pages 106 et 110 (40 et 41) : `A x = b'`, `b'` projete de `b`."""
    assert au_sens_des_moindres_carres(matrice, second_membre) == solution


def test_le_second_membre_projete_de_la_page_110_est_celui_de_la_page_96() -> None:
    """Page 110 (41) : `b' = (2, 4, 2)`, la projection deja calculee page 96."""
    matrice = sp.Matrix([[1, 0], [1, 1], [0, 1]])
    solution = sp.Matrix(au_sens_des_moindres_carres(matrice, TROIS))
    assert matrice * solution == sp.Matrix([2, 4, 2])


def test_sous_la_tolerance_un_produit_scalaire_est_tenu_pour_nul() -> None:
    """`polytech.maths.domain.algebre.metrique.TOLERANCE` : le seuil de nullite."""
    petit, grand = metrique.TOLERANCE / 10, metrique.TOLERANCE * 10
    assert metrique.est_orthogonale([(1.0, 0.0), (petit, 1.0)])
    assert not metrique.est_orthogonale([(1.0, 0.0), (grand, 1.0)])
