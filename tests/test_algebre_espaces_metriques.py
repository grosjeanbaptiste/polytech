"""Syllabus d'algebre, chapitre IX, sections 1 a 4 : produits scalaires, projections.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 135 a 144, et
les solutions des pages 151 et 152. Produit scalaire, norme, distance,
procede de Gram-Schmidt, projection orthogonale, sous-espace orthogonal ;
dans `R^n` et `C^n`, puis dans un espace de matrices et un espace de
polynomes. Les exemples 1 a 3 et les reponses des exercices 1 a 24.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import est_libre
from polytech.maths.domain.algebre.metrique import gram_schmidt as gram_schmidt_flottant
from polytech.maths.domain.algebre.orthogonalisation import (
    distance_au_sous_espace,
    gram_schmidt,
    normaliser,
    orthogonal,
    projection,
)
from polytech.maths.domain.algebre.produits_scalaires import (
    aux_points,
    de_la_trace,
    est_orthogonale,
    integral,
    norme,
    usuel,
)
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741 -- l'unite imaginaire
sqrt = sp.sqrt
SUR_MOINS_UN_UN = integral(X, -1, 1)
MONOMES = (sp.Integer(1), X, X**2)
UNITES = tuple(M(2, 2, [int(i == j) for j in range(4)]) for i in range(4))


def V(*composantes: object) -> sp.Matrix:  # noqa: N802 -- un vecteur colonne
    return M(list(composantes))


def _canon(dimension: int) -> tuple[sp.Matrix, ...]:
    return tuple(V(*(int(i == j) for j in range(dimension))) for i in range(dimension))


def _rang(famille: tuple[sp.Matrix, ...]) -> int:
    rang: int = M([list(v) for v in famille]).rank()
    return rang


def _memes(obtenue: tuple[sp.Matrix, ...], officielle: tuple[sp.Matrix, ...]) -> bool:
    """Deux familles de matrices ou de colonnes qui engendrent le meme sous-espace."""
    ensemble = _rang((*obtenue, *officielle))
    return _rang(obtenue) == ensemble == _rang(officielle)


def _egales(obtenue: tuple[sp.Matrix, ...], officielle: tuple[sp.Matrix, ...]) -> bool:
    return len(obtenue) == len(officielle) and all(
        sp.simplify(o - e).is_zero_matrix
        for o, e in zip(obtenue, officielle, strict=True)
    )


def test_le_produit_scalaire_usuel_et_ses_trois_proprietes() -> None:
    """IX.1 : positif, lineaire a gauche, symetrique -- hermitien dans `C^n`."""
    x, y, z = (M(sp.symbols(f"{nom}1:4", real=True)) for nom in "xyz")
    alpha, beta = sp.symbols("alpha beta", real=True)
    gauche = usuel(alpha * x + beta * y, z)
    assert sp.expand(gauche - alpha * usuel(x, z) - beta * usuel(y, z)) == 0
    assert usuel(x, y) == usuel(y, x)
    assert usuel(x, x) == sum(c**2 for c in x)
    u, v = M(sp.symbols("u1:3")), M(sp.symbols("v1:3"))
    assert sp.simplify(usuel(u, v) - sp.conjugate(usuel(v, u))) == 0


def test_l_orthogonalisation_de_l_exemple_1() -> None:
    """`v1 = (1, 2, 1)`, `v2 = (1/2, 0, -1/2)`, `v3 = (2/3, -2/3, 2/3)`."""
    base = (V(1, 2, 1), V(1, 1, 0), V(1, 0, 1))
    orthogonale = gram_schmidt(base, usuel)
    demi, deux_tiers = R(1, 2), R(2, 3)
    attendue = (V(1, 2, 1), V(demi, 0, -demi), V(deux_tiers, -deux_tiers, deux_tiers))
    assert orthogonale == attendue
    assert est_orthogonale(orthogonale, usuel)
    assert not est_orthogonale(base, usuel)


def test_la_normalisation_de_l_exemple_1() -> None:
    """`e1 = (1, 2, 1)/racine(6)`, `e2` et `e3` de meme : une base orthonormale."""
    base = (V(1, 2, 1), V(1, 1, 0), V(1, 0, 1))
    orthonormale = normaliser(gram_schmidt(base, usuel), usuel)
    attendue = (
        V(1, 2, 1) / sqrt(6),
        sqrt(2) / 2 * V(1, 0, -1),
        sqrt(3) / 3 * V(1, -1, 1),
    )
    assert _egales(orthonormale, attendue)
    assert all(norme(e, usuel) == 1 for e in orthonormale)


def test_la_projection_et_la_distance_de_l_exemple_1() -> None:
    """`proj_W x = (-1/3, -2/3, 5/3)`, a la distance `4 racine(6)/3`."""
    w = (V(R(1, 2), 0, -R(1, 2)), V(R(2, 3), -R(2, 3), R(2, 3)))
    x = V(1, 2, 3)
    assert projection(x, w, usuel) == V(-R(1, 3), -R(2, 3), R(5, 3))
    assert distance_au_sous_espace(x, w, usuel) == 4 * sqrt(6) / 3


def test_la_projection_demande_une_base_orthogonale() -> None:
    """La formule de la page 136 ne vaut que si les `v_i` sont orthogonaux."""
    with pytest.raises(ValueError, match="base non orthogonale"):
        projection(V(1, 2, 3), (V(1, 2, 1), V(1, 1, 0)), usuel)


def test_une_famille_liee_ne_s_orthogonalise_pas() -> None:
    with pytest.raises(ValueError, match="famille liee"):
        gram_schmidt((V(1, 2, 1), V(2, 4, 2)), usuel)


def test_des_polynomes_lies_ne_s_orthogonalisent_pas_non_plus() -> None:
    """`2X + 1` est combinaison de 1 et de `X`."""
    with pytest.raises(ValueError, match="famille liee"):
        gram_schmidt((sp.Integer(1), X, 2 * X + 1), SUR_MOINS_UN_UN)


def test_l_orthogonalisation_concorde_avec_le_calcul_flottant_existant() -> None:
    """Lien : `metrique.gram_schmidt` rend les memes vecteurs, non normes."""
    base = ((1.0, 2.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))
    exacte = gram_schmidt(tuple(V(*(int(c) for c in v)) for v in base), usuel)
    for flottant, exact in zip(gram_schmidt_flottant(base), exacte, strict=True):
        assert flottant == pytest.approx([float(c) for c in exact])


def test_l_exemple_2_dans_c3() -> None:
    """Gram-Schmidt hermitien, projection `(-1/2, (2 - i)/2, (1 + i)/2)`, distance 1."""
    base = (V(1, I, 0), V(1, 1, 1), V(0, 0, I))
    v1, v2, v3 = gram_schmidt(base, usuel)
    assert v1 == V(1, I, 0)
    assert sp.simplify(v2 - V((1 + I) / 2, (1 - I) / 2, 1)).is_zero_matrix
    assert sp.simplify(v3 - V((1 - I) / 4, -(1 + I) / 4, I / 2)).is_zero_matrix
    assert est_orthogonale((v1, v2, v3), usuel)
    x = V(0, 1 - I, I)
    projete = projection(x, (v1, v2), usuel)
    assert sp.simplify(projete - V(-R(1, 2), (2 - I) / 2, (1 + I) / 2)).is_zero_matrix
    assert distance_au_sous_espace(x, (v1, v2), usuel) == 1


def test_l_orthogonal_de_l_exemple_3() -> None:
    """Dans `R^6` : `dim W^perp = 6 - 2 = 4`, et la forme generale de la page 141."""
    generateurs = (V(1, 1, -2, 3, 4, 5), V(0, 0, 1, 1, 0, 7))
    obtenu = orthogonal(generateurs, _canon(6), usuel)
    assert len(obtenu) == 4
    a, b, c, d = sp.symbols("alpha beta gamma delta")
    generale = V(-a - 5 * b - 4 * c - 19 * d, a, -b - 7 * d, b, c, d)
    for generateur in generateurs:
        assert sp.expand(usuel(generale, generateur)) == 0
    officielle = tuple(generale.diff(parametre) for parametre in (a, b, c, d))
    assert _memes(obtenu, officielle)


def test_les_reponses_des_exercices_1_et_2() -> None:
    """Normes `racine(14)`, `racine(26)`, distance `racine(6)`, produit 17 ; `-8i`."""
    x1, x2 = V(1, 2, 3), V(-1, 3, 4)
    assert (norme(x1, usuel), norme(x2, usuel)) == (sqrt(14), sqrt(26))
    assert norme(x1 - x2, usuel) == sqrt(6)
    assert usuel(x1, x2) == 17
    x, y = V(3, 2, I), V(I, I, -3)
    assert usuel(x, y) == -8 * I
    assert norme(x, usuel) == sqrt(14)


def test_la_reponse_de_l_exercice_3() -> None:
    """Une base de `W`, ni orthogonale ni unitaire ; puis une base orthonormale."""
    famille = (V(1, 1, 0, 1), V(1, -2, 0, 0), V(1, 0, -1, 2))
    assert est_libre(tuple(tuple(v) for v in famille))
    assert not est_orthogonale(famille, usuel)
    assert all(norme(v, usuel) != 1 for v in famille)
    officielle = (
        V(1, 1, 0, 1) / sqrt(3),
        V(4, -5, 0, 1) / sqrt(42),
        V(-4, -2, -7, 6) / sqrt(105),
    )
    assert _egales(normaliser(gram_schmidt(famille, usuel), usuel), officielle)


def test_la_reponse_de_l_exercice_4() -> None:
    """Deux bases orthonormales de sous-espaces de `C^3`."""
    a = normaliser(gram_schmidt((V(1, I, 0), V(1, 1, 1)), usuel), usuel)
    officielle = (
        V(1, I, 0) / sqrt(2),
        V((1 + I) / (2 * sqrt(2)), (1 - I) / (2 * sqrt(2)), 1 / sqrt(2)),
    )
    assert _egales(a, officielle)
    b = normaliser(gram_schmidt((V(1, -1, -I), V(I, 1, 2)), usuel), usuel)
    officielle = (V(1, -1, -I) / sqrt(3), V(1, 2 + 3 * I, 3 - I) / (2 * sqrt(6)))
    assert _egales(b, officielle)


def test_les_reponses_des_exercices_5_et_6() -> None:
    """Trois bases orthonormales, dans `R^3` puis `R^4`."""
    a = normaliser(gram_schmidt((V(1, 1, -1), V(1, 0, 1)), usuel), usuel)
    assert _egales(a, (V(1, 1, -1) / sqrt(3), V(1, 0, 1) / sqrt(2)))
    b = normaliser(gram_schmidt((V(2, 1, 1), V(1, 3, -1)), usuel), usuel)
    assert _egales(b, (V(2, 1, 1) / sqrt(6), V(-1, 7, -5) / (5 * sqrt(3))))
    six = normaliser(gram_schmidt((V(1, 2, 1, 0), V(1, 2, 3, 1)), usuel), usuel)
    assert _egales(six, (V(1, 2, 1, 0) / sqrt(6), V(-1, -2, 5, 3) / sqrt(39)))


def test_la_reponse_de_l_exercice_7() -> None:
    """Base orthonormale relue sur l'image de la page 151 ; `d(X, W) = 3`."""
    famille = (V(1, 1, 0, 0), V(1, -1, 1, 1), V(-1, 0, 2, 1))
    orthogonale = gram_schmidt(famille, usuel)
    demi = R(1, 2)
    officielle = (
        V(1, 1, 0, 0) / sqrt(2),
        V(demi, -demi, demi, demi),
        V(-sqrt(2) / 3, sqrt(2) / 3, sqrt(2) / 2, sqrt(2) / 6),
    )
    assert _egales(normaliser(orthogonale, usuel), officielle)
    assert distance_au_sous_espace(V(1, 1, 1, -3), orthogonale, usuel) == 3


def test_la_reponse_de_l_exercice_8() -> None:
    """`(i, i, i)` est dans le sous-espace : la distance est nulle."""
    base = gram_schmidt((V(1, I, 0), V(1, 1, 1)), usuel)
    assert distance_au_sous_espace(V(I, I, I), base, usuel) == 0


def test_la_reponse_de_l_exercice_9() -> None:
    """Les vecteurs projetes sur `(1, -i, 1, -1)` : `(2 - i a, a, b, b - 2)`."""
    w = tuple(V(*v) for v in base_du_noyau(M([[1, -I, 0, 0], [0, 0, 1, 1]])))
    base = gram_schmidt(w, usuel)
    a, b = sp.symbols("alpha beta")
    officielle = V(2 - I * a, a, b, b - 2)
    ecart = projection(officielle, base, usuel) - V(1, -I, 1, -1)
    assert sp.simplify(ecart).is_zero_matrix
    assert len(orthogonal(w, _canon(4), usuel)) == 2


def test_les_reponses_des_exercices_10_a_12() -> None:
    """10 : dimension 4 ; 11 : `W^perp = <(1, ..., 1)>` ; 12 : `(3, -4, -i)`."""
    dix = orthogonal((V(1, 1, -2, 3, 4, 5), V(0, 0, 1, 1, 0, 7)), _canon(6), usuel)
    assert len(dix) == 4
    for n in (3, 5):
        w = tuple(V(*v) for v in base_du_noyau(sp.ones(1, n)))
        assert len(w) == n - 1
        assert _memes(orthogonal(w, _canon(n), usuel), (sp.ones(n, 1),))
    douze = orthogonal((V(3, 2, I), V(I, 0, -3)), _canon(3), usuel)
    assert len(douze) == 1
    assert _memes(douze, (V(3, -4, -I),))
    assert usuel(V(3, -4, -I), V(3, 2, I)) == 0
    assert usuel(V(3, -4, -I), V(I, 0, -3)) == 0


def test_les_reponses_des_exercices_13_et_14() -> None:
    """Base orthonormale dans `C^4`, projection `(1, -i, 1, -1)`, distance `2 r(2)`."""
    x1, x2 = V(0, 0, I, -I), V(-1, I, -I, I)
    orthogonale = gram_schmidt((x1, x2), usuel)
    officielle = (V(0, 0, I, -I) / sqrt(2), V(-1, I, 0, 0) / sqrt(2))
    assert _egales(normaliser(orthogonale, usuel), officielle)
    x = V(3, I, 1, -1)
    assert projection(x, orthogonale, usuel) == V(1, -I, 1, -1)
    assert distance_au_sous_espace(x, orthogonale, usuel) == 2 * sqrt(2)


def test_les_reponses_des_exercices_15_et_16() -> None:
    """Distance d'un point a une droite (environ 3,74) et a un plan (`13/7`)."""
    droite = tuple(V(*v) for v in base_du_noyau(M([[1, 1, 1], [4, 1, -1]])))
    distance = distance_au_sous_espace(V(1, 2, 3), droite, usuel)
    assert distance == sqrt(R(531, 38))
    direction = V(1, 1, 1).cross(V(4, 1, -1))
    assert (
        sp.simplify(distance - V(1, 2, 3).cross(direction).norm() / direction.norm())
        == 0
    )
    assert float(distance) == pytest.approx(3.74, abs=5e-3)
    plan = tuple(V(*v) for v in base_du_noyau(M([[2, -3, 6]])))
    orthogonale = gram_schmidt(plan, usuel)
    assert distance_au_sous_espace(V(1, 5, 0), orthogonale, usuel) == R(13, 7)


def test_la_trace_definit_un_produit_scalaire() -> None:
    """Exercice 17 : `tr(A B^t)` symetrique, lineaire, `tr(A A^t)` somme de carres."""
    a, b, c = (M(2, 2, sp.symbols(f"{nom}1:5", real=True)) for nom in "abc")
    alpha, beta = sp.symbols("alpha beta", real=True)
    assert sp.expand(de_la_trace(a, b) - de_la_trace(b, a)) == 0
    gauche = de_la_trace(alpha * a + beta * b, c)
    droite = alpha * de_la_trace(a, c) + beta * de_la_trace(b, c)
    assert sp.expand(gauche - droite) == 0
    assert sp.expand(de_la_trace(a, a) - sum(x**2 for x in a)) == 0
    assert sp.expand(de_la_trace(a, b) - (a.T * b).trace()) == 0


def test_la_reponse_de_l_exercice_18() -> None:
    """Base orthonormale des symetriques, et la plus proche de `[[1, 2], [3, 4]]`."""
    base = (M([[1, 0], [0, 0]]), M([[0, 1], [1, 1]]), M([[1, 1], [1, 2]]))
    orthogonale = gram_schmidt(base, de_la_trace)
    officielle = (
        M([[1, 0], [0, 0]]),
        M([[0, 1], [1, 1]]) / sqrt(3),
        M([[0, -1], [-1, 2]]) / sqrt(6),
    )
    assert _egales(normaliser(orthogonale, de_la_trace), officielle)
    proche = projection(M([[1, 2], [3, 4]]), orthogonale, de_la_trace)
    assert proche == M([[1, R(5, 2)], [R(5, 2), 4]])


def test_la_reponse_de_l_exercice_19() -> None:
    """La distance de `[[1, 2], [3, 4]]` a `W` vaut `1/racine(2)`."""
    w = (M([[1, 0], [0, 0]]), M([[0, 1], [1, 1]]), M([[0, 0], [0, 1]]))
    orthogonale = gram_schmidt(w, de_la_trace)
    distance = distance_au_sous_espace(M([[1, 2], [3, 4]]), orthogonale, de_la_trace)
    assert sp.simplify(distance - 1 / sqrt(2)) == 0


def test_la_reponse_de_l_exercice_20() -> None:
    """`W^perp`, la projection sur `W^perp`, puis celle sur `W` par difference."""
    w = (M([[1, 0], [2, 1]]), M([[2, 1], [2, 3]]), M([[0, 1], [-2, 1]]))
    perp = orthogonal(w, UNITES, de_la_trace)
    c, d = sp.symbols("c d")
    generale = M([[-2 * c - d, 2 * c - d], [c, d]])
    assert all(sp.expand(de_la_trace(generale, g)) == 0 for g in w)
    assert _memes(perp, (generale.diff(c), generale.diff(d)))
    cible = M([[1, 0], [3, 1]])
    sur_perp = projection(cible, gram_schmidt(perp, de_la_trace), de_la_trace)
    assert sur_perp == M([[-R(2, 9), R(2, 9)], [R(1, 9), 0]])
    assert cible - sur_perp == M([[R(11, 9), -R(2, 9)], [R(26, 9), 1]])


def test_la_reponse_de_l_exercice_21() -> None:
    """`<1 | X^2> = 2/3` et `||X|| = racine(2/3)` pour l'integrale sur `[-1, 1]`."""
    assert SUR_MOINS_UN_UN(sp.Integer(1), X**2) == R(2, 3)
    assert norme(X, SUR_MOINS_UN_UN) == sqrt(R(2, 3))


def test_les_reponses_des_exercices_22_et_23() -> None:
    """`{1, X, X^2 - 1/3}` pour l'integrale ; `{1, X, X^2 - 2}` pour les cinq points."""
    assert gram_schmidt(MONOMES, SUR_MOINS_UN_UN) == (1, X, X**2 - R(1, 3))
    cinq_points = aux_points(X, (-2, -1, 0, 1, 2))
    assert gram_schmidt(MONOMES, cinq_points) == (1, X, X**2 - 2)


def test_la_reponse_de_l_exercice_24() -> None:
    """`W^perp = {-3c/5 - 9c/5 X + c X^2}`, relue sur l'image de la page 152."""
    quatre_points = aux_points(X, (-1, 0, 1, 2))
    perp = orthogonal((sp.Integer(1), X**2), MONOMES, quatre_points)
    assert len(perp) == 1
    assert sp.expand(perp[0] - (-R(3, 5) - R(9, 5) * X + X**2)) == 0
    assert quatre_points(perp[0], sp.Integer(1)) == 0
    assert quatre_points(perp[0], X**2) == 0
