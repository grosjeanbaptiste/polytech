"""Syllabus d'algebre, chapitre VIII, section 8 : les examens 33 a 45.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 117 a 123, et
les solutions des pages 128 a 134. Treize questions d'examen (janvier 94 a
janvier 05) : matrice dans les bases canoniques, noyau, image, discussion
selon un parametre, antecedents d'un element. Chaque reponse officielle
est recalculee.

Les sous-questions « relatives au chapitre sur les espaces metriques »
(orthogonal du noyau, projection) sont traitees avec le chapitre IX.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.espaces_de_fonctions import (
    coordonnees_d_une_fonction,
)
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    est_libre,
    rang_de_la_famille,
)
from polytech.maths.domain.algebre.isomorphismes import (
    polynome_d_un_vecteur,
    vecteur_d_un_polynome,
    vecteur_d_une_matrice,
)
from polytech.maths.domain.algebre.matrices_particulieres import trace
from polytech.maths.domain.algebre.rang_exact import rangs_selon_le_parametre
from polytech.maths.domain.algebre.representation import (
    matrice_associee,
    matrice_canonique,
)
from polytech.maths.domain.algebre.resolution import (
    est_l_ensemble_des_solutions,
    par_echelonnement_simple,
)
from polytech.maths.domain.algebre.solutions import Nature, nature_des_solutions
from polytech.maths.domain.algebre.sous_espaces import (
    base_du_noyau,
    engendrent_le_meme_sous_espace,
    somme,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
A, B, T = sp.symbols("a b t")
Famille = tuple[tuple[sp.Expr, ...], ...]
Vecteurs = Sequence[Sequence[object]]
type Scalaire = sp.Expr | int


def _polynomes(*polynomes: sp.Expr, degre: int) -> Famille:
    return tuple(vecteur_d_un_polynome(sp.sympify(p), degre) for p in polynomes)


def _matrices(*matrices: list[list[object]]) -> Famille:
    return tuple(vecteur_d_une_matrice(M(m)) for m in matrices)


def _sur_les_matrices(regle: Callable[[sp.Matrix], sp.Matrix]) -> ApplicationLineaire:
    """La matrice d'une regle sur `R^(2x2)`, lue sur les quatre matrices unites."""
    unites = [M(2, 2, [int(i == j) for j in range(4)]) for i in range(4)]
    images = [vecteur_d_une_matrice(regle(u)) for u in unites]
    return ApplicationLineaire(M(images).T)


def _memes(obtenue: Vecteurs, officielle: Vecteurs) -> bool:
    """Deux bases du meme sous-espace : meme longueur, meme espace engendre."""
    if not officielle:
        return obtenue == ()
    return len(obtenue) == len(officielle) and engendrent_le_meme_sous_espace(
        obtenue, officielle
    )


def test_l_examen_33_janvier_94() -> None:
    """`p -> 2(X+1)(p + p') - (X+1)^2 (p' + p'')` : matrice, noyau, image."""

    def regle(p: sp.Expr) -> sp.Expr:
        derivee, seconde = sp.diff(p, X), sp.diff(p, X, 2)
        return sp.expand(
            2 * (X + 1) * (p + derivee) - (X + 1) ** 2 * (derivee + seconde)
        )

    colonnes = [vecteur_d_un_polynome(regle(X**k), 3) for k in range(3)]
    f = ApplicationLineaire(M(colonnes).T)
    assert f.matrice == M([[2, 1, -2], [2, 2, -2], [0, 1, 0], [0, 0, 0]])
    assert _memes(f.noyau(), _polynomes(1 + X**2, degre=2))
    assert _memes(f.image(), _polynomes(1 + X, 1 + 2 * X + X**2, degre=3))
    cible = M(vecteur_d_un_polynome(4 * X**2 + 21 * X + 8, 3))
    assert nature_des_solutions(SystemeLineaire(f.matrice, cible)) is Nature.IMPOSSIBLE


def test_l_examen_34_aout_94() -> None:
    """`[[a, b], [c, d]] -> [[a+b+c, 2b+d], [d, d]]` : noyau de dimension 1, rang 3."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (a + b + c, 2 * b + d, d, d), 4
    )
    assert _memes(f.noyau(), _matrices([[1, 0], [-1, 0]]))
    assert f.rang() == 3
    officielle = _matrices([[1, 0], [0, 0]], [[1, 2], [0, 0]], [[0, 1], [1, 1]])
    assert _memes(f.image(), officielle)


def _examen_35(valeur: Scalaire) -> ApplicationLineaire:
    mt = M([[valeur, 1], [1, valeur]])
    return _sur_les_matrices(lambda u: u * mt + mt * u)


def test_la_matrice_de_l_examen_35_janvier_96() -> None:
    """`f(A) = A M_t + M_t A`, et les valeurs de `t` ou son rang chute."""
    f = _examen_35(T)
    assert f.matrice == M(
        [[2 * T, 1, 1, 0], [1, 2 * T, 0, 1], [1, 0, 2 * T, 1], [0, 1, 1, 2 * T]]
    )
    assert rangs_selon_le_parametre(f.matrice, T) == (4, {-1: 3, 0: 2, 1: 3})


@pytest.mark.parametrize("t", [-1, 1])
def test_l_examen_35_pour_t_egal_plus_ou_moins_un(t: int) -> None:
    """Noyau `[[1, -t], [-t, 1]]` ; image de dimension 3."""
    f = _examen_35(t)
    assert _memes(f.noyau(), _matrices([[1, -t], [-t, 1]]))
    officielle = _matrices(
        [[2 * t, 1], [1, 0]], [[1, 2 * t], [0, 1]], [[1, 0], [2 * t, 1]]
    )
    assert _memes(f.image(), officielle)


def test_l_examen_35_pour_t_nul() -> None:
    """Noyau de dimension 2 ; image `{[[0, 1], [1, 0]], [[1, 0], [0, 1]]}`."""
    f = _examen_35(0)
    assert _memes(f.noyau(), _matrices([[1, 0], [0, -1]], [[0, 1], [-1, 0]]))
    assert _memes(f.image(), _matrices([[0, 1], [1, 0]], [[1, 0], [0, 1]]))
    assert _examen_35(3).noyau() == ()


def _examen_36(a: Scalaire, b: Scalaire) -> ApplicationLineaire:
    return _sur_les_matrices(lambda u: a * u + b * u.T)


def test_l_examen_36_janvier_98() -> None:
    """`f(A) = aA + bA^t` : matrice, puis les trois cas `a = -b`, `a = b`, sinon."""
    assert _examen_36(A, B).matrice == M(
        [[A + B, 0, 0, 0], [0, A, B, 0], [0, B, A, 0], [0, 0, 0, A + B]]
    )
    symetriques = _matrices([[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]])
    antisymetrique = _matrices([[0, 1], [-1, 0]])
    opposes = _examen_36(2, -2)
    assert _memes(opposes.noyau(), symetriques)
    assert _memes(opposes.image(), antisymetrique)
    egaux = _examen_36(2, 2)
    assert _memes(egaux.noyau(), antisymetrique)
    assert _memes(egaux.image(), symetriques)
    autres = _examen_36(2, 5)
    assert autres.noyau() == ()
    assert autres.rang() == 4


def test_l_inverse_de_l_examen_36() -> None:
    """`f^-1(Y) = (aY - bY^t)/(a^2 - b^2)`."""
    y = M(2, 2, sp.symbols("y1:5"))
    officielle = (A * y - B * y.T) / (A**2 - B**2)
    image = A * officielle + B * officielle.T
    assert sp.simplify(image - y) == sp.zeros(2, 2)
    inverse = _examen_36(A, B).matrice.inv()
    obtenue = M(2, 2, list(inverse * M(vecteur_d_une_matrice(y))))
    assert sp.simplify(obtenue - officielle) == sp.zeros(2, 2)


def _examen_37(alpha: object, beta: object) -> ApplicationLineaire:
    return ApplicationLineaire(
        M(
            [
                [0, beta, 0, alpha],
                [alpha, 0, beta, 0],
                [0, 1, 0, 1],
                [alpha, 0, beta, 0],
                [0, alpha, 0, beta],
                [beta, 0, alpha, 0],
            ]
        )
    )


def test_l_examen_37_janvier_99() -> None:
    """De `R^(2x2)` dans `R^(2x3)` : la matrice `6 x 4`, puis les trois cas."""
    alpha, beta = sp.symbols("alpha beta")
    regle = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (
            beta * b + alpha * d,
            alpha * a + beta * c,
            b + d,
            alpha * a + beta * c,
            alpha * b + beta * d,
            beta * a + alpha * c,
        ),
        4,
    )
    assert regle.matrice == _examen_37(alpha, beta).matrice
    egaux = _examen_37(3, 3)
    assert _memes(egaux.noyau(), _matrices([[1, 0], [-1, 0]], [[0, 1], [0, -1]]))
    assert _memes(egaux.image(), ((0, 3, 0, 3, 0, 3), (3, 0, 1, 0, 3, 0)))
    opposes = _examen_37(3, -3)
    assert _memes(opposes.noyau(), _matrices([[1, 0], [1, 0]]))
    officielle = ((0, 3, 0, 3, 0, -3), (-3, 0, 1, 0, 3, 0), (3, 0, 1, 0, -3, 0))
    assert _memes(opposes.image(), officielle)
    autres = _examen_37(3, 5)
    assert autres.noyau() == ()
    assert autres.rang() == 4


def _examen_38(a: object, b: object) -> ApplicationLineaire:
    return ApplicationLineaire(
        M(
            [
                [b, 0, a, 0, a, 0],
                [0, a, 0, 1, 0, b],
                [a, 0, b, 0, b, 0],
                [0, b, 0, 1, 0, a],
            ]
        )
    )


def test_la_matrice_de_l_examen_38_janvier_00() -> None:
    """De `P5` dans `R^(2x2)` : chaque colonne est l'image d'un monome."""
    regle = ApplicationLineaire.depuis_la_regle(
        lambda a0, a1, a2, a3, a4, a5: (
            A * (a2 + a4) + B * a0,
            a3 + A * a1 + B * a5,
            A * a0 + B * (a2 + a4),
            a3 + A * a5 + B * a1,
        ),
        6,
    )
    assert regle.matrice == _examen_38(A, B).matrice


def test_les_noyaux_de_l_examen_38() -> None:
    """`a = b` : dimension 4 ; `a = -b` : dimension 3 ; sinon dimension 2."""
    a = 3
    egaux = _polynomes(-1 + X**2, -1 + X**4, X - a * X**3, -a * X**3 + X**5, degre=5)
    assert _memes(_examen_38(a, a).noyau(), egaux)
    opposes = _polynomes(1 + X**2, 1 + X**4, X + X**5, degre=5)
    assert _memes(_examen_38(a, -a).noyau(), opposes)
    autres = _polynomes(X - (3 + 5) * X**3 + X**5, -(X**2) + X**4, degre=5)
    assert _memes(_examen_38(3, 5).noyau(), autres)


def test_les_images_de_l_examen_38() -> None:
    """Dimensions 2, 3 et 4, avec les bases officielles."""
    egaux = _matrices([[1, 0], [1, 0]], [[0, 1], [0, 1]])
    assert _memes(_examen_38(3, 3).image(), egaux)
    opposes = _matrices([[-1, 0], [1, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]])
    assert _memes(_examen_38(3, -3).image(), opposes)
    assert _examen_38(3, 5).rang() == 4


def test_l_examen_39_aout_00() -> None:
    """`p -> (p(1), p'(1), p(-1), p'(-1))` sur `P4` : noyau, image, antecedents."""
    colonnes = [(1, k, (-1) ** k, k * (-1) ** (k - 1) if k else 0) for k in range(5)]
    f = ApplicationLineaire(M(colonnes).T)
    assert f.matrice == M(
        [[1, 1, 1, 1, 1], [0, 1, 2, 3, 4], [1, -1, 1, -1, 1], [0, 1, -2, 3, -4]]
    )
    assert _memes(f.noyau(), _polynomes(1 - 2 * X**2 + X**4, degre=4))
    assert f.est_surjective()
    systeme = SystemeLineaire(f.matrice, M([0, 1, -2, 1]))
    libre = sp.Symbol("a4")
    officielle = (-1 + libre, 1, -2 * libre, 0, libre)
    assert est_l_ensemble_des_solutions(systeme, officielle, (libre,))
    assert par_echelonnement_simple(systeme).particuliere == (-1, 1, 0, 0, 0)


def test_l_examen_40_janvier_01() -> None:
    """Quatre fonctions dans la base `{2^x, x 2^x}` : matrice, noyau, antecedents."""
    x = sp.Symbol("x", real=True)
    base = (2**x, x * 2**x)
    generatrices = (
        2 ** (x + 2) - x * 2**x,
        x * 2**x + 2 ** (x + 1),
        (2 * x - 1) * 2 ** (x + 1),
        -(1 + x) * 2 ** (x + 2),
    )
    colonnes = [coordonnees_d_une_fonction(g, base, x) for g in generatrices]
    f = ApplicationLineaire(M(colonnes).T)
    assert f.matrice == M([[4, 2, -2, -4], [-1, 1, 4, -4]])
    assert f.rang() == 2
    assert _memes(f.noyau(), ((5, -7, 3, 0), (-2, 10, 0, 3)))
    u3, u4 = sp.symbols("u3 u4")
    officielle = (
        -(5 - 10 * u3 + 4 * u4) / 6,
        (13 - 14 * u3 + 20 * u4) / 6,
        u3,
        u4,
    )
    systeme = SystemeLineaire(f.matrice, M([1, 3]))
    assert est_l_ensemble_des_solutions(systeme, officielle, (u3, u4))


def test_l_examen_41_aout_01() -> None:
    """De `R^(2x2)` dans `P3` : matrice, noyau, image, et les polynomes atteints."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a0, a1, a2, a3: (-a0 + a3, -a0 + a1 + a2, -a0 + a1 + a2, -a0 + a3), 4
    )
    assert f.matrice == M([[-1, 0, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 0], [-1, 0, 0, 1]])
    assert _memes(f.noyau(), _matrices([[1, 0], [1, 1]], [[0, 1], [-1, 0]]))
    assert _memes(f.image(), _polynomes(1 + X**3, X + X**2, degre=3))
    alpha, beta = sp.symbols("alpha beta")
    cible = (3, -alpha, 5, beta)
    conditions = [cible[0] - cible[3], cible[1] - cible[2]]
    assert sp.solve(conditions, [alpha, beta]) == {alpha: -5, beta: 3}
    a0, a1 = sp.symbols("a0 a1")
    systeme = SystemeLineaire(f.matrice, M([3, 5, 5, 3]))
    officielle = (a0, a1, 5 + a0 - a1, a0 + 3)
    assert est_l_ensemble_des_solutions(systeme, officielle, (a0, a1))


def test_l_examen_42_janvier_02() -> None:
    """`g(M) = (tr M, tr M^t, tr 2M)`, `f` de matrice donnee, et `f o g`."""
    unites = [M(2, 2, [int(i == j) for j in range(4)]) for i in range(4)]
    g = ApplicationLineaire(M([(trace(u), trace(u.T), trace(2 * u)) for u in unites]).T)
    assert g.image() == ((1, 1, 2),)
    assert _memes(
        g.noyau(), _matrices([[1, 0], [0, -1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]])
    )
    f = ApplicationLineaire(M([[1, -2, -2], [2, 1, 5], [4, -3, 2]]))
    x, y, z = sp.symbols("x y z")
    image = polynome_d_un_vecteur(f.image_de((x, y, z)))
    attendue = (
        (x - 2 * y - 2 * z) + (2 * x + y + 5 * z) * X + (4 * x - 3 * y + 2 * z) * X**2
    )
    assert sp.expand(image - attendue) == 0
    m = M(2, 2, sp.symbols("m1:5"))
    composee = f.matrice * g.matrice * M(vecteur_d_une_matrice(m))
    officielle = trace(m) * (-5 + 13 * X + 5 * X**2)
    assert sp.expand(polynome_d_un_vecteur(tuple(composee)) - officielle) == 0


def _examen_43(valeur: Scalaire) -> ApplicationLineaire:
    associee = M(
        [
            [-1, -2, valeur - 4, valeur - 3],
            [1, 2, 4, 3],
            [0, valeur - 2, valeur - 2, 0],
        ]
    ).applyfunc(sp.sympify)
    v = _matrices(
        [[1, 0], [0, 0]], [[1, 1], [0, 0]], [[1, 1], [1, 0]], [[1, 1], [1, 1]]
    )
    w = _polynomes(1, 1 + X, 1 + X**2, degre=2)
    return ApplicationLineaire(sp.expand(matrice_canonique(associee, v, w)))


def test_la_matrice_canonique_de_l_examen_43_janvier_04() -> None:
    """`A_(v,w)` donnee : `A_(e,e')` s'en deduit par les deux passages."""
    assert _examen_43(A).matrice == M(
        [[0, A - 2, A, 2 - A], [1, 1, 2, -1], [0, A - 2, 0, 2 - A]]
    )


def test_la_discussion_de_l_examen_43() -> None:
    """Noyaux et images selon `a` ; surjective hors de 0 et 2 ; jamais injective."""
    nul = _examen_43(0)
    assert _memes(nul.noyau(), _matrices([[-2, 0], [1, 0]], [[0, 1], [0, 1]]))
    assert _memes(nul.image(), _polynomes(X, 1 + X**2, degre=2))
    deux = _examen_43(2)
    assert _memes(deux.noyau(), _matrices([[1, 0], [0, 1]], [[0, 1], [0, 1]]))
    assert _memes(deux.image(), _polynomes(1, X, degre=2))
    generique = _examen_43(5)
    assert _memes(generique.noyau(), _matrices([[0, 1], [0, 1]]))
    assert generique.est_surjective()
    assert rangs_selon_le_parametre(_examen_43(A).matrice, A) == (3, {0: 2, 2: 2})
    for valeur in (0, 2, 5):
        assert not _examen_43(valeur).est_injective()


def test_l_examen_44_aout_04() -> None:
    """De `P3` dans `P2` : matrice, noyau, base de `W + Ker f`, surjectivite."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a0, a1, a2, a3: (a1 + 2 * a2, a0 - a3, a1 + a2 + a3), 4
    )
    assert f.matrice == M([[0, 1, 2, 0], [1, 0, 0, -1], [0, 1, 1, 1]])
    assert _memes(f.noyau(), _polynomes(1 - 2 * X + X**2 + X**3, degre=3))
    w = base_du_noyau(M([[1, 0, -1, 0], [-1, -1, 2, 1]]))
    officielle = _polynomes(1 - 2 * X + X**2 + X**3, 1 + X**2 - X**3, X + X**3, degre=3)
    assert _memes(somme(w, f.noyau()), officielle)
    assert f.est_surjective()
    assert f.rang() == 4 - len(f.noyau())


def test_l_examen_45_janvier_05() -> None:
    """De `R^(2x2)` dans `P2` : `W != Im f`, surjective, noyau, antecedents de `W`."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (a, a + d, -b + c + 2 * d), 4
    )
    w = _polynomes(1 + X, X + 2 * X**2, degre=2)
    assert rang_de_la_famille(w) == 2
    assert f.est_surjective()
    assert not engendrent_le_meme_sous_espace(w, f.image())
    assert _memes(f.noyau(), _matrices([[0, 1], [1, 0]]))
    assert f.matrice.shape == (3, 4)
    a, b, d = sp.symbols("a b d")
    symetrique = f.image_de((a, b, b, d))
    assert sp.expand(M(symetrique) - (a * M(w[0]) + d * M(w[1]))) == sp.zeros(3, 1)
    assert not engendrent_le_meme_sous_espace(w, (*w, f.image_de((0, 1, 0, 0))))


def test_la_matrice_de_l_examen_45_dans_les_bases_v_et_w() -> None:
    """La question 7, relue sur la page 134."""
    f = ApplicationLineaire.depuis_la_regle(
        lambda a, b, c, d: (a, a + d, -b + c + 2 * d), 4
    )
    v = _matrices(
        [[1, 1], [1, 1]], [[0, 1], [1, 1]], [[0, 0], [1, 0]], [[1, 0], [0, 1]]
    )
    w = _polynomes(1 + X - X**2, 1 + X**2, 3 * X, degre=2)
    assert est_libre(w)
    demi = R(1, 2)
    assert matrice_associee(f.matrice, v, w) == M(
        [
            [-demi, -1, -demi, -demi],
            [3 * demi, 1, demi, 3 * demi],
            [R(5, 6), R(2, 3), R(1, 6), R(5, 6)],
        ]
    )
