"""Syllabus d'algebre, chapitre IX, sections 5 a 7 : fonctions et moindres carres.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 145 a 150, et
les solutions des pages 152 et 153. L'approximation d'une fonction au
sens des moindres carres, la resolution d'un systeme impossible au sens
des moindres carres, la regression lineaire ; l'exemple d'aout 06, les
reponses des exercices 25 a 33.

S'y ajoutent les sous-questions des examens du chapitre VIII « relatives
au chapitre sur les espaces metriques » (33, 34, 36, 37, 41, 42), dont les
reponses sont aux pages 128 a 132.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.moindres_carres import (
    au_sens_des_moindres_carres,
    droite_de_regression,
)
from polytech.maths.domain.algebre.orthogonalisation import (
    distance_au_sous_espace,
    gram_schmidt,
    normaliser,
    orthogonal,
    projection,
)
from polytech.maths.domain.algebre.produits_scalaires import (
    de_la_trace,
    est_orthogonale,
    integral,
    norme,
    usuel,
)
from polytech.maths.domain.algebre.solutions import Nature, nature_des_solutions
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
sqrt, pi, sin, cos = sp.sqrt, sp.pi, sp.sin, sp.cos
MONOMES = (sp.Integer(1), X, X**2)
SUR_MOINS_UN_UN = integral(X, -1, 1)
SUR_LA_PERIODE = integral(X, 0, 2 * pi)
UNITES = tuple(M(2, 2, [int(i == j) for j in range(4)]) for i in range(4))


def _rang(famille: tuple[sp.Matrix, ...]) -> int:
    rang: int = M([list(v) for v in famille]).rank()
    return rang


def _memes(obtenue: tuple[sp.Matrix, ...], officielle: tuple[sp.Matrix, ...]) -> bool:
    ensemble = _rang((*obtenue, *officielle))
    return _rang(obtenue) == ensemble == _rang(officielle)


def _m(*lignes: list[object]) -> sp.Matrix:
    return M(list(lignes))


def test_l_orthogonalisation_de_l_exemple_d_aout_06() -> None:
    """Sur `[0, 1]` : `{1, x - 1/2, x^2 - x + 1/6}`."""
    produit = integral(X, 0, 1)
    orthogonale = gram_schmidt(MONOMES, produit)
    assert orthogonale == (1, X - R(1, 2), X**2 - X + R(1, 6))
    assert produit(X - R(1, 2), X - R(1, 2)) == R(1, 12)
    assert produit(orthogonale[2], orthogonale[2]) == R(1, 180)


def test_la_meilleure_approximation_de_racine_de_x() -> None:
    """`racine(x)` approchee par `-4/7 x^2 + 48/35 x + 6/35` sur `[0, 1]`."""
    produit = integral(X, 0, 1)
    orthogonale = gram_schmidt(MONOMES, produit)
    produits = tuple(produit(sqrt(X), v) for v in orthogonale)
    assert produits == (R(2, 3), R(1, 15), -R(1, 315))
    approchee = projection(sqrt(X), orthogonale, produit)
    attendue = -R(4, 7) * X**2 + R(48, 35) * X + R(6, 35)
    assert sp.expand(approchee - attendue) == 0


def test_la_projection_minimise_l_erreur_quadratique() -> None:
    """IX.5 : tout autre polynome de degre deux commet une erreur plus grande."""
    produit = integral(X, 0, 1)
    meilleure = -R(4, 7) * X**2 + R(48, 35) * X + R(6, 35)
    erreur = norme(sqrt(X) - meilleure, produit)
    for autre in (meilleure + R(1, 10), meilleure + X / 10, X, 1 - X**2):
        assert norme(sqrt(X) - autre, produit) > erreur


def test_les_reponses_des_exercices_25_et_26() -> None:
    """`{1, t, t^2 - 1/3}` ; `{1, sin, cos, sin 5x}` orthogonale, puis normee."""
    assert gram_schmidt(MONOMES, SUR_MOINS_UN_UN) == (1, X, X**2 - R(1, 3))
    sur_moins_pi_pi = integral(X, -pi, pi)
    famille = (sp.Integer(1), sin(X), cos(X), sin(5 * X))
    assert est_orthogonale(famille, sur_moins_pi_pi)
    normee = normaliser(famille, sur_moins_pi_pi)
    attendue = (
        1 / sqrt(2 * pi),
        sin(X) / sqrt(pi),
        cos(X) / sqrt(pi),
        sin(5 * X) / sqrt(pi),
    )
    assert all(sp.simplify(o - e) == 0 for o, e in zip(normee, attendue, strict=True))


def test_la_reponse_de_l_exercice_27() -> None:
    """`e^x` approchee par `(e - 1/e)/2 + 3x/e`, a la distance `racine(1 - 7/e^2)`."""
    base = gram_schmidt((sp.Integer(1), X), SUR_MOINS_UN_UN)
    approchee = projection(sp.exp(X), base, SUR_MOINS_UN_UN)
    attendue = (sp.E - 1 / sp.E) / 2 + 3 * X / sp.E
    assert sp.simplify(approchee - attendue) == 0
    distance = distance_au_sous_espace(sp.exp(X), base, SUR_MOINS_UN_UN)
    ecart = (distance**2 - (1 - 7 / sp.E**2)).rewrite(sp.exp)
    assert sp.simplify(ecart) == 0


def test_les_reponses_des_exercices_28_et_29() -> None:
    """`x` projete sur `{1, cos, sin}` : `pi - 2 sin x` ; `1 + x` sur cinq fonctions."""
    trois = (sp.Integer(1), cos(X), sin(X))
    assert est_orthogonale(trois, SUR_LA_PERIODE)
    assert sp.simplify(projection(X, trois, SUR_LA_PERIODE) - (pi - 2 * sin(X))) == 0
    cinq = (sp.Integer(1), cos(X), cos(2 * X), sin(X), sin(2 * X))
    assert est_orthogonale(cinq, SUR_LA_PERIODE)
    attendue = 1 + pi - 2 * sin(X) - sin(2 * X)
    assert sp.simplify(projection(1 + X, cinq, SUR_LA_PERIODE) - attendue) == 0


def test_la_reponse_de_l_exercice_30() -> None:
    """Un systeme impossible, resolu au sens des moindres carres : `(1/3, 1/3, 1/3)`."""
    a = M([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]])
    b = M([1, 0, 1, 1])
    assert nature_des_solutions(SystemeLineaire(a, b)) is Nature.IMPOSSIBLE
    assert au_sens_des_moindres_carres(a, b) == (R(1, 3), R(1, 3), R(1, 3))


def test_la_solution_approchee_resout_a_x_egale_la_projection_de_b() -> None:
    """IX.6 : « resoudre `A X = b'` ou `b' = proj_W b` », `W` celui des colonnes."""
    a = M([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]])
    b = M([1, 0, 1, 1])
    colonnes = gram_schmidt(tuple(a.col(j) for j in range(3)), usuel)
    projete = projection(b, colonnes, usuel)
    solution = M(au_sens_des_moindres_carres(a, b))
    assert a * solution == projete
    assert usuel(b - a * solution, a.col(0)) == 0
    ecart = norme(b - a * solution, usuel)
    for voisin in (solution + M([1, 0, 0]) / 10, solution - M([0, 1, 1]) / 7):
        assert norme(b - a * voisin, usuel) > ecart


def test_un_systeme_possible_garde_sa_solution() -> None:
    """Si `b` est deja dans `W`, la solution approchee est la solution."""
    a = M([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
    assert au_sens_des_moindres_carres(a, M([9, 1, 0])) == (1, 2, 3)


def test_des_colonnes_liees_n_ont_pas_de_solution_approchee_unique() -> None:
    with pytest.raises(ValueError, match="colonnes liees"):
        au_sens_des_moindres_carres(M([[1, 2], [2, 4], [3, 6]]), M([1, 0, 0]))


def test_la_reponse_de_l_exercice_31() -> None:
    """La droite `y = 9/10 x + 6/5` passe presque par les cinq points."""
    points = ((0, 1), (1, 3), (2, 2), (3, 4), (4, 5))
    assert droite_de_regression(points) == (R(9, 10), R(6, 5))


def test_la_droite_de_regression_minimise_les_ecarts_verticaux() -> None:
    """Page 150 : la somme des carres des `(a x_i + b) - y_i` est minimale."""
    points = ((0, 1), (1, 3), (2, 2), (3, 4), (4, 5))
    a, b = sp.symbols("a b")
    somme = sum((a * x + b - y) ** 2 for x, y in points)
    critique = sp.solve([sp.diff(somme, a), sp.diff(somme, b)], [a, b])
    assert (critique[a], critique[b]) == droite_de_regression(points)


def test_des_points_alignes_donnent_leur_droite() -> None:
    assert droite_de_regression(((0, 1), (1, 3), (2, 5))) == (2, 1)


def test_l_examen_32_aout_97() -> None:
    """`d(10x^3 - 3x^2 + 1, P2) = 4 racine(2/7)`."""
    base = gram_schmidt(MONOMES, SUR_MOINS_UN_UN)
    p = 10 * X**3 - 3 * X**2 + 1
    distance = distance_au_sous_espace(p, base, SUR_MOINS_UN_UN)
    assert sp.simplify(distance - 4 * sqrt(R(2, 7))) == 0


def test_l_examen_33_janvier_04() -> None:
    """`W` de dimension 3 sauf pour `a = -3` ; meilleure approximation de `A`."""
    a = sp.Symbol("a")
    generateurs = (_m([0, 1], [-1, 0]), _m([1, 1], [1, -1]), _m([-1, 1], [a, 1]))
    lignes = M([list(g) for g in generateurs])
    assert sp.factor(lignes[:, [0, 1, 2]].det()) == -(a + 3)
    assert lignes.subs(a, -3).rank() == 2
    cible = _m([1, 4], [1, -1])
    generique = gram_schmidt(tuple(g.subs(a, 5) for g in generateurs), de_la_trace)
    assert projection(cible, generique, de_la_trace) == cible
    singulier = gram_schmidt(generateurs[:2], de_la_trace)
    proche = projection(cible, singulier, de_la_trace)
    assert proche == _m([7, 13], [1, -7]) / 4
    assert cible - proche == _m([-3, 3], [3, 3]) / 4


def test_l_orthogonal_du_noyau_a_l_examen_33_du_chapitre_viii() -> None:
    """`Ker f = <1 + X^2>` : `(Ker f)^perp = <1 - 5/2 X^2, X>`, et les projections."""
    noyau = (1 + X**2,)
    perp = orthogonal(noyau, MONOMES, SUR_MOINS_UN_UN)
    assert len(perp) == 2
    for officiel in (1 - R(5, 2) * X**2, X):
        assert SUR_MOINS_UN_UN(officiel, noyau[0]) == 0
    a, b, c = sp.symbols("a b c")
    p = a * X**2 + b * X + c
    sur_le_noyau = projection(p, noyau, SUR_MOINS_UN_UN)
    assert sp.expand(sur_le_noyau - (16 * a + 40 * c) / 56 * (1 + X**2)) == 0
    reste = b * X + (-2 * a + 2 * c) / 14 * (2 - 5 * X**2)
    assert sp.expand(p - sur_le_noyau - reste) == 0


def test_les_orthogonaux_des_noyaux_aux_examens_34_a_42() -> None:
    """Pour `tr(A B^t)` : les bases officielles des pages 129 a 132."""
    cas = (
        (
            (_m([1, 0], [-1, 0]),),
            (_m([1, 0], [1, 0]), _m([0, 1], [0, 0]), _m([0, 0], [0, 1])),
        ),
        (
            (_m([1, 0], [0, 0]), _m([0, 1], [1, 0]), _m([0, 0], [0, 1])),
            (_m([0, 1], [-1, 0]),),
        ),
        (
            (_m([0, 1], [-1, 0]),),
            (_m([1, 0], [0, 0]), _m([0, 1], [1, 0]), _m([0, 0], [0, 1])),
        ),
        (
            (_m([1, 0], [-1, 0]), _m([0, 1], [0, -1])),
            (_m([1, 0], [1, 0]), _m([0, 1], [0, 1])),
        ),
        (
            (_m([1, 0], [1, 0]),),
            (_m([1, 0], [-1, 0]), _m([0, 1], [0, 0]), _m([0, 0], [0, 1])),
        ),
        (
            (_m([1, 0], [1, 1]), _m([0, 1], [-1, 0])),
            (_m([-1, 1], [1, 0]), _m([-1, 0], [0, 1])),
        ),
        (
            (_m([1, 0], [0, -1]), _m([0, 1], [0, 0]), _m([0, 0], [1, 0])),
            (_m([1, 0], [0, 1]),),
        ),
    )
    for noyau, officielle in cas:
        assert _memes(orthogonal(noyau, UNITES, de_la_trace), officielle)


def test_les_polynomes_orthogonaux_a_x_a_l_examen_41() -> None:
    """Dans `P3`, pour l'integrale sur `[-1, 1]` : `a0 - 3/5 a3 X + a2 X^2 + a3 X^3`."""
    base = (sp.Integer(1), X, X**2, X**3)
    perp = orthogonal((X,), base, SUR_MOINS_UN_UN)
    assert len(perp) == 3
    a0, a2, a3 = sp.symbols("a0 a2 a3")
    officiel = a0 - R(3, 5) * a3 * X + a2 * X**2 + a3 * X**3
    assert sp.expand(SUR_MOINS_UN_UN(officiel, X)) == 0
    assert all(SUR_MOINS_UN_UN(q, X) == 0 for q in perp)
