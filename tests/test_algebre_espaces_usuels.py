"""Syllabus d'algebre, chapitre VII, sections 7 a 10 : matrices, polynomes, fonctions.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 86 a 93, et
les solutions des pages 95 et 96. La methode du syllabus est la
bijection : « l'espace vectoriel `R^4` est une autre facon de voir
l'espace vectoriel `R^(2x2)` », et de meme `R[X]<=2` et `R^3`. Les
exemples 10 a 13, puis les reponses officielles des exercices 23 a 44.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import (
    appartient,
    coordonnees,
    est_generatrice,
    est_libre,
    est_une_base,
    extraire_une_base,
    prolonger_en_base,
    rang_de_la_famille,
    relation_de_dependance,
)
from polytech.maths.domain.algebre.isomorphismes import (
    libres_par_evaluation,
    matrice_d_un_vecteur,
    polynome_d_un_vecteur,
    vecteur_d_un_polynome,
    vecteur_d_une_matrice,
)
from polytech.maths.domain.algebre.sous_espaces import (
    base_du_noyau,
    engendrent_le_meme_sous_espace,
    intersection,
    somme,
)
from polytech.maths.domain.algebre.stabilite import (
    contre_exemple,
    est_lineaire_homogene,
)
from polytech.maths.domain.differentielles.wronskien import Wronskien
from polytech.maths.domain.symboles import TEMPS
from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from collections.abc import Callable

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741 -- l'unite imaginaire
T = sp.Symbol("t", real=True)
F5 = (
    M([[1, 1], [2, 1]]),
    M([[1, -1], [0, 1]]),
    M([[1, 2], [2, 0]]),
    M([[0, 0], [-1, 1]]),
    M([[1, 1], [1, 1]]),
)
F7 = tuple(vecteur_d_une_matrice(matrice) for matrice in F5)
ZERO_2 = sp.zeros(2, 2)


def _polynomes(*polynomes: sp.Expr, degre: int) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(vecteur_d_un_polynome(p, degre) for p in polynomes)


def _matrices(*matrices: list[list[Any]]) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(vecteur_d_une_matrice(M(m)) for m in matrices)


def test_la_bijection_entre_matrices_et_quadruplets() -> None:
    """`[[a, b], [c, d]] -> (a, b, c, d)` respecte la somme et le produit scalaire."""
    assert F7 == (
        (1, 1, 2, 1),
        (1, -1, 0, 1),
        (1, 2, 2, 0),
        (0, 0, -1, 1),
        (1, 1, 1, 1),
    )
    assert matrice_d_un_vecteur(F7[0], 2, 2) == F5[0]
    somme_des_images = M(F7[0]) + 3 * M(F7[1])
    assert M(vecteur_d_une_matrice(F5[0] + 3 * F5[1])) == somme_des_images


def test_un_vecteur_ne_se_range_que_dans_un_format_de_sa_taille() -> None:
    with pytest.raises(ValueError, match="composantes pour un format"):
        matrice_d_un_vecteur((1, 2, 3), 2, 2)


def test_l_exemple_10_dans_l_espace_des_matrices() -> None:
    """`[[1, 1], [1, 1]] = F/4 + G/4 + H/2 + K/2`, et `F6` est une base de `R^(2x2)`."""
    assert not est_libre(F7)
    assert coordonnees(F7[4], F7[:4]) == (R(1, 4), R(1, 4), R(1, 2), R(1, 2))
    assert est_une_base(F7[:4])
    assert est_generatrice(F7)
    assert extraire_une_base(F7) == F7[:4]
    recombinee = (F5[0] + F5[1]) / 4 + (F5[2] + F5[3]) / 2
    assert recombinee == F5[4]


def test_l_exemple_11_dans_l_espace_des_polynomes() -> None:
    """`p3 = -p1 + 2 p2` : la famille correspond a `F1` de `R^3`."""
    p1, p2, p3 = 1 + 2 * X - X**2, 2 + 4 * X + X**2, 3 + 6 * X + 3 * X**2
    famille = _polynomes(p1, p2, p3, degre=2)
    assert famille == ((1, 2, -1), (2, 4, 1), (3, 6, 3))
    assert not est_libre(famille)
    assert relation_de_dependance(famille) == (1, -2, 1)
    assert sp.expand(p3 - (-p1 + 2 * p2)) == 0
    assert polynome_d_un_vecteur(famille[0]) == p1


def test_un_polynome_de_degre_trop_grand_n_a_pas_de_vecteur() -> None:
    with pytest.raises(ValueError, match="degre superieur"):
        vecteur_d_un_polynome(X**3, 2)


def test_l_exemple_12_donne_une_base_d_un_sous_espace_de_polynomes() -> None:
    """`a + b + c + d = 0` : `{-1 + X, -1 + X^2, -1 + X^3}`."""
    base = base_du_noyau(M([[1, 1, 1, 1]]))
    assert base == ((-1, 1, 0, 0), (-1, 0, 1, 0), (-1, 0, 0, 1))
    polynomes = tuple(polynome_d_un_vecteur(v) for v in base)
    assert polynomes == (-1 + X, -1 + X**2, -1 + X**3)


def test_l_exemple_13_donne_les_coordonnees_d_un_polynome() -> None:
    """`4 + 5X + 6X^2` dans la base de la page 90 : 1, 0, 3."""
    base = _polynomes(1 + 2 * X + 3 * X**2, 4 + 3 * X + 5 * X**2, 1 + X + X**2, degre=2)
    assert base == ((1, 2, 3), (4, 3, 5), (1, 1, 1))
    assert coordonnees(vecteur_d_un_polynome(4 + 5 * X + 6 * X**2, 2), base) == (
        1,
        0,
        3,
    )


def test_la_reponse_de_l_exercice_23() -> None:
    """Quatre matrices de `R^(2x2)` : la famille n'est pas libre."""
    famille = _matrices(
        [[1, -2], [0, 3]], [[2, 3], [0, -1]], [[2, -1], [2, 1]], [[3, 9], [-4, -2]]
    )
    assert not est_libre(famille)


def test_la_reponse_de_l_exercice_24() -> None:
    """E symetriques : dimension 3 ; H antisymetriques : dimension 1 ; F et G : non."""
    a, b, c, d = sp.symbols("a b c d")
    assert est_lineaire_homogene([b - c], (a, b, c, d))
    e = base_du_noyau(M([[0, 1, -1, 0]]))
    assert e == ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1))
    assert matrice_d_un_vecteur(e[1], 2, 2) == M([[0, 1], [1, 0]])
    assert est_lineaire_homogene([a, d, b + c], (a, b, c, d))
    h = base_du_noyau(M([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]]))
    assert len(h) == 1
    assert engendrent_le_meme_sous_espace(h, _matrices([[0, 1], [-1, 0]]))
    f = contre_exemple(lambda m: m[0, 0] == sp.sqrt(2), [M([[sp.sqrt(2), 0], [0, 0]])])
    assert f is not None
    assert f.nature == "vecteur nul"
    candidats = [M([[1, 0], [0, 0]]), M([[0, 0], [0, 1]])]
    g = contre_exemple(lambda m: m.det() == 0, candidats)
    assert g is not None
    assert g.nature == "somme"
    assert (g.temoins[0] + g.temoins[1]).det() != 0


def test_la_reponse_de_l_exercice_25() -> None:
    """Les deux premieres matrices forment une base ; dimension 2."""
    famille = _matrices([[1, 1], [1, 1]], [[1, 0], [0, 2]], [[2, -1], [-1, 5]])
    assert extraire_une_base(famille) == famille[:2]
    assert rang_de_la_famille(famille) == 2


def test_la_reponse_de_l_exercice_26() -> None:
    """Dans `C^(2x2)` : `{B, C}` base de `W`, completee par `E11` et `E21`."""
    a, b, c = _matrices([[1, I], [3, -I]], [[I, 0], [2, I]], [[-1 + I, -1], [5 * I, 0]])
    assert rang_de_la_famille((a, b, c)) == 2
    assert est_libre((b, c))
    assert engendrent_le_meme_sous_espace((b, c), (a, b, c))
    assert est_une_base((b, c, (1, 0, 0, 0), (0, 0, 1, 0)))


def test_la_reponse_de_l_exercice_27() -> None:
    """Trois matrices libres, completees par `[[2, -1], [1, 1]]` en une base."""
    famille = _matrices([[1, -2], [0, 3]], [[2, 3], [0, -1]], [[2, -1], [2, 1]])
    assert est_libre(famille)
    assert est_une_base((*famille, vecteur_d_une_matrice(M([[2, -1], [1, 1]]))))
    assert est_une_base(prolonger_en_base(famille))


def test_la_reponse_de_l_exercice_28() -> None:
    """`V = {[[a, b], [-b_barre, a_barre]]}` : espace sur `R`, pas sur `C`."""

    def dans_v(m: sp.Matrix) -> bool:
        return bool(
            sp.simplify(m[1, 1] - sp.conjugate(m[0, 0])) == 0
            and sp.simplify(m[1, 0] + sp.conjugate(m[0, 1])) == 0
        )

    base = (
        sp.eye(2),
        M([[I, 0], [0, -I]]),
        M([[0, 1], [-1, 0]]),
        M([[0, I], [I, 0]]),
    )
    assert all(dans_v(m) for m in base)
    assert contre_exemple(dans_v, base, scalaires=(-1, 2, R(1, 3))) is None
    sur_c = contre_exemple(dans_v, base, scalaires=(I,))
    assert sur_c is not None
    assert sur_c.nature == "multiple"
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    generale = a1 * base[0] + a2 * base[1] + b1 * base[2] + b2 * base[3]
    alpha, beta = a1 + I * a2, b1 + I * b2
    attendue = M([[alpha, beta], [-sp.conjugate(beta), sp.conjugate(alpha)]])
    assert sp.simplify(generale - attendue) == ZERO_2
    reels = tuple(
        (*vecteur_d_une_matrice(sp.re(m)), *vecteur_d_une_matrice(sp.im(m)))
        for m in base
    )
    assert est_libre(reels)


def test_la_reponse_de_l_exercice_29() -> None:
    """`W1 + W2` a pour base `[[0, 1], [-1, 0]]` et `[[1, 2], [3, 4]]`."""
    antisymetriques = _matrices([[0, 1], [-1, 0]])
    w2 = _matrices([[1, 2], [3, 4]], [[1, 1], [4, 4]], [[3, 5], [10, 12]])
    officielle = (antisymetriques[0], w2[0])
    assert len(somme(antisymetriques, w2)) == 2
    assert engendrent_le_meme_sous_espace(somme(antisymetriques, w2), officielle)
    assert len(intersection(antisymetriques, w2)) == 1


def test_la_reponse_de_l_exercice_30() -> None:
    """A, C, D, E, G : oui ; B, F : non -- sur les coefficients, au degre 3."""
    coefficients = a0, a1, a2, a3 = sp.symbols("a0:4")
    p = a0 + a1 * X + a2 * X**2 + a3 * X**3
    assert est_lineaire_homogene([p.subs(X, 0)], coefficients)
    paire = sp.Poly(p - p.subs(X, -X), X).coeffs()
    assert est_lineaire_homogene(paire, coefficients)
    impaire = sp.Poly(p + p.subs(X, -X), X).coeffs()
    assert est_lineaire_homogene(impaire, coefficients)
    assert est_lineaire_homogene([sum(p.subs(X, i) for i in (1, 2, 3))], coefficients)
    assert not est_lineaire_homogene([p.subs(X, 0) - 1], coefficients)
    b = contre_exemple(lambda q: q.subs(X, 0) == 1, [1 + X])
    assert b is not None
    assert b.nature == "vecteur nul"
    f = contre_exemple(lambda q: sp.degree(q, X) == 3, [X**3, -(X**3) + X])
    assert f is not None
    assert f.nature == "vecteur nul"


def test_la_reponse_de_l_exercice_31() -> None:
    """La famille de `R[X]<=4` n'est pas libre."""
    famille = _polynomes(
        1 + X - X**2 + X**4,
        X**3 - X**4,
        1 + X**3,
        2 + X - X**2 + 3 * X**3 - X**4,
        degre=4,
    )
    assert not est_libre(famille)


@pytest.mark.parametrize(
    ("polynomes", "libre", "generatrice"),
    [
        ((1 + X - 2 * X**2, 1 - 2 * X + X**2, -2 + X + X**2), False, False),
        (
            (
                3 + 2 * X + X**2,
                5 + 3 * X + 4 * X**2,
                1 + X + X**2,
                2 + X + 3 * X**2,
                1 + 3 * X + 2 * X**2,
            ),
            False,
            True,
        ),
        ((X, sp.Integer(0), 1 + X**2), False, False),
        ((1 + X, X**2 - 1), True, False),
    ],
    ids=["F1", "F2", "F3", "F4"],
)
def test_la_reponse_de_l_exercice_32(
    polynomes: tuple[sp.Expr, ...], *, libre: bool, generatrice: bool
) -> None:
    """Chaque famille est-elle libre, et generatrice de `R[X]<=2`."""
    famille = _polynomes(*polynomes, degre=2)
    assert est_libre(famille) is libre
    assert est_generatrice(famille) is generatrice


def test_les_bases_proposees_a_l_exercice_32() -> None:
    """e) les trois premiers polynomes de `F2` ; d) `F4` completee par `1 + X + X^2`."""
    f2 = _polynomes(
        3 + 2 * X + X**2, 5 + 3 * X + 4 * X**2, 1 + X + X**2, 2 + X + 3 * X**2, degre=2
    )
    assert extraire_une_base(f2) == f2[:3]
    assert est_une_base(_polynomes(1 + X, X**2 - 1, 1 + X + X**2, degre=2))


def test_les_reponses_des_exercices_33_a_35() -> None:
    """33 : `a = 11/4` ; 34 : coordonnees `-1, 1, 2` ; 35 : `{p1, p2}`, dimension 2."""
    a = sp.Symbol("a")
    generateurs = _polynomes(1 - X + X**2, -1 + 2 * X + 3 * X**2, degre=2)
    determinant = M([*generateurs, (-2, a, 1)]).det()
    assert sp.solve(determinant, a) == [R(11, 4)]
    assert appartient((-2, R(11, 4), 1), generateurs)
    base = _polynomes(X**2 - 2 * X + 5, 2 * X**2 - 3 * X, X + 3, degre=2)
    assert est_une_base(base)
    assert coordonnees(vecteur_d_un_polynome(1 + X + X**2, 2), base) == (-1, 1, 2)
    w = _polynomes(1 + X + X**2 + X**3, 1 + 2 * X**3, 2 - X - X**2 + 5 * X**3, degre=3)
    assert extraire_une_base(w) == w[:2]


def test_la_reponse_de_l_exercice_36() -> None:
    """`p(0) = p(1)`, `p(0) = 2 p(1)`, et leur intersection."""
    w1 = base_du_noyau(M([[0, 1, 1, 1]]))
    officielle_1 = _polynomes(sp.Integer(1), X - X**3, X**2 - X**3, degre=3)
    assert engendrent_le_meme_sous_espace(w1, officielle_1)
    w2 = base_du_noyau(M([[1, 2, 2, 2]]))
    officielle_2 = _polynomes(1 - X**3 / 2, X - X**3, X**2 - X**3, degre=3)
    assert engendrent_le_meme_sous_espace(w2, officielle_2)
    commune = intersection(w1, w2)
    officielle = _polynomes(X - X**3, X**2 - X**3, degre=3)
    assert len(commune) == 2
    assert engendrent_le_meme_sous_espace(commune, officielle)


def test_les_reponses_des_exercices_37_et_38() -> None:
    """37 : `{1 - X, X^2 + X^3}`, completee par `X` et `X^2` ; 38 : `p(1) = p(2)`."""
    famille = _polynomes(
        1 - X + X**2 + X**3,
        -1 + X + 2 * X**2 + 2 * X**3,
        2 - 2 * X + X**2 + X**3,
        degre=3,
    )
    officielle = _polynomes(1 - X, X**2 + X**3, degre=3)
    assert rang_de_la_famille(famille) == 2
    assert engendrent_le_meme_sous_espace(famille, officielle)
    assert est_une_base((*officielle, *_polynomes(X, X**2, degre=3)))
    w = base_du_noyau(M([[0, 1, 3, 7]]))
    exemple = _polynomes(sp.Integer(1), -3 * X + X**2, -7 * X + X**3, degre=3)
    assert engendrent_le_meme_sous_espace(w, exemple)
    for vecteur in exemple:
        p = polynome_d_un_vecteur(vecteur)
        assert p.subs(X, 1) == p.subs(X, 2)


def test_les_fonctions_libres_des_exercices_39_et_40() -> None:
    """`{sin t, cos t}`, `{sin t, sin 2t}`, `{e^t, ln t}` : des valeurs le prouvent."""
    quart, demi = sp.pi / 4, sp.pi / 2
    assert libres_par_evaluation((sp.sin(T), sp.cos(T)), (0, demi), T)
    assert libres_par_evaluation((sp.sin(T), sp.sin(2 * T)), (quart, demi), T)
    assert libres_par_evaluation((sp.exp(T), sp.log(T)), (1, 2), T)


def test_des_valeurs_mal_choisies_ne_prouvent_rien() -> None:
    """`sin t` et `sin 2t` s'annulent toutes deux en 0 et en `pi` : rien n'en sort."""
    assert not libres_par_evaluation((sp.sin(T), sp.sin(2 * T)), (0, sp.pi), T)
    assert not libres_par_evaluation((T, 2 * T), (1, 2), T)
    with pytest.raises(ValueError, match="autant de points que de fonctions"):
        libres_par_evaluation((sp.sin(T), sp.cos(T)), (0,), T)


def test_le_wronskien_confirme_l_exercice_39() -> None:
    """Lien avec le chapitre des equations differentielles : un wronskien non nul."""
    premier = Wronskien.des_solutions((sp.sin(TEMPS), sp.cos(TEMPS)))
    assert premier.determinant() == -1
    assert premier.est_fondamental()
    second = Wronskien.des_solutions((sp.sin(TEMPS), sp.sin(2 * TEMPS)))
    assert second.determinant().subs(TEMPS, sp.pi / 2) == -2
    assert second.est_fondamental()


def test_la_reponse_de_l_exercice_41() -> None:
    """A, E, H : sous-espaces ; B, C, D, F, G : non."""
    f0, f1, quart, plus, moins = sp.symbols("f0 f1 f_quart f_plus f_moins")
    assert est_lineaire_homogene([f1 - 2 * f0], (f0, f1))
    assert est_lineaire_homogene([quart], (quart,))
    assert est_lineaire_homogene([plus + moins], (plus, moins))
    assert not est_lineaire_homogene([f1 - f0 - 1], (f0, f1))
    assert not est_lineaire_homogene([quart - R(1, 4)], (quart,))
    fonctions = [T, 1 - T, T**2, sp.Integer(1)]

    def temoin(appartenance: Callable[[sp.Expr], bool]) -> str:
        violation = contre_exemple(appartenance, fonctions)
        assert violation is not None
        return violation.nature

    def en(f: sp.Expr, x: object) -> sp.Expr:
        evaluee: sp.Expr = sp.sympify(f).subs(T, x)
        return evaluee

    assert temoin(lambda f: en(f, 1) == en(f, 0) + 1) == "vecteur nul"
    assert temoin(lambda f: en(f, R(1, 4)) == R(1, 4)) == "vecteur nul"
    positive = lambda f: all(en(f, x) >= 0 for x in (0, R(1, 2), 1))  # noqa: E731
    assert temoin(positive) == "multiple"
    croissante = lambda f: bool(sp.diff(sp.sympify(f), T).subs(T, R(1, 2)) >= 0)  # noqa: E731
    assert temoin(croissante) == "multiple"
    s_annule = lambda f: bool(sp.solveset(sp.sympify(f), T, sp.Interval(0, 1)))  # noqa: E731
    assert temoin(s_annule) == "somme"


def test_la_reponse_de_l_exercice_42() -> None:
    """Dans les coordonnees de `{X, Y, Z}` : a) libre ; b) liee ; c) libre."""
    assert est_libre(((1, 0, 0), (1, 1, 0), (1, 1, 1)))
    assert not est_libre(((1, -1, 0), (0, 1, -1), (-1, 0, 1)))
    assert est_libre(((1, 1, 0), (0, 1, 1), (1, 0, 1)))


def test_la_reponse_de_l_exercice_43() -> None:
    """Trois bases d'un plan, et les coordonnees de `v1` et `v2` dans chacune."""
    a = sp.Symbol("a")
    v1, v2 = (1, 0), (0, 1)
    premiere = ((1, 1), (0, 1))
    assert est_une_base(premiere)
    assert (coordonnees(v1, premiere), coordonnees(v2, premiere)) == ((1, -1), (0, 1))
    deuxieme = ((1, 1), (1, -1))
    assert est_une_base(deuxieme)
    demi = R(1, 2)
    assert coordonnees(v1, deuxieme) == (demi, demi)
    assert coordonnees(v2, deuxieme) == (demi, -demi)
    troisieme = ((1, 0), (a, 1))
    assert est_une_base(troisieme)
    assert (coordonnees(v1, troisieme), coordonnees(v2, troisieme)) == ((1, 0), (-a, 1))


def test_la_reponse_de_l_exercice_44() -> None:
    """`X1`, `X2`, `X3` sont libres, et `v1` les complete en une base."""
    famille = ((1, 1, 0, 1), (1, -1, 1, -2), (-1, 1, -1, 1))
    assert est_libre(famille)
    assert est_une_base((*famille, (1, 0, 0, 0)))
    assert prolonger_en_base(famille) == (*famille, (1, 0, 0, 0))
