"""Syllabus d'algebre, chapitre VIII, sections 1 a 4 : applications lineaires.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 97 a 104, et
les solutions des pages 124 et 125. Definition, noyau, image, la relation
`dim Ker f + dim Im f = dim V`, les exemples 1 et 2, puis les reponses
officielles des exercices 1 a 13.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    est_une_base,
    rang_de_la_famille,
)
from polytech.maths.domain.algebre.isomorphismes import (
    matrice_d_un_vecteur,
    polynome_d_un_vecteur,
    vecteur_d_un_polynome,
    vecteur_d_une_matrice,
)
from polytech.maths.domain.algebre.linearite import (
    existe_une_application_lineaire,
    respecte_les_combinaisons,
)
from polytech.maths.domain.algebre.matrices_particulieres import (
    est_antisymetrique,
    est_symetrique,
    trace,
)
from polytech.maths.domain.algebre.rang_exact import rangs_selon_le_parametre
from polytech.maths.domain.algebre.sous_espaces import (
    engendrent_le_meme_sous_espace,
)
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.symboles import TEMPS
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
R = sp.Rational
A, T = sp.symbols("a t")
U, V = M(sp.symbols("u1:4")), M(sp.symbols("v1:4"))
EXEMPLE_1 = ApplicationLineaire(
    M([[1, 0, 1, 0, 0], [0, 1, -1, 0, 1], [1, 0, 3, 0, 1], [0, 0, 0, 1, 0]])
)
EXEMPLE_2 = M([[1, A, A], [0, A - 1, 0], [-1, -1, 2], [1, 1, A]])


def _generique(nom: str, degre: int) -> sp.Expr:
    """Un polynome a coefficients symboliques."""
    return sp.Add(*(c * X**k for k, c in enumerate(sp.symbols(f"{nom}0:{degre + 1}"))))


def test_la_symetrie_est_lineaire_et_la_translation_ne_l_est_pas() -> None:
    """VIII.1 : `s(x, y) = (-x, y)` ; contre-exemple `t(x, y) = (x, y) + (1, 1)`."""
    u, v = M(sp.symbols("u1:3")), M(sp.symbols("v1:3"))
    assert respecte_les_combinaisons(lambda p: M([-p[0], p[1]]), u, v)
    assert not respecte_les_combinaisons(lambda p: p + M([1, 1]), u, v)


def test_la_derivation_est_lineaire() -> None:
    """VIII.1 : `D : g -> g'`."""
    g, h = sp.Function("g")(TEMPS), sp.Function("h")(TEMPS)
    assert respecte_les_combinaisons(lambda f: sp.diff(f, TEMPS), g, h)


def test_les_colonnes_de_la_matrice_sont_les_images_de_la_base_canonique() -> None:
    """Remarque de la page 98 : la premiere colonne de `A` est `f(1, 0, ..., 0)`."""
    f = ApplicationLineaire.depuis_la_regle(lambda x, y, z: (x + y, x - z), 3)
    assert f.matrice == M([[1, 1, 0], [1, 0, -1]])
    assert f.image_de((1, 0, 0)) == (1, 1)
    assert f.image_de((2, 3, 5)) == (5, -3)


def test_une_regle_non_lineaire_n_a_pas_de_matrice() -> None:
    with pytest.raises(ValueError, match="regle non lineaire"):
        ApplicationLineaire.depuis_la_regle(lambda x, y: (x * y,), 2)


def test_le_noyau_et_l_image_de_la_projection_sur_oxy() -> None:
    """Page 98 : `Ker f = Oz` et `Im f = Oxy`."""
    projection = ApplicationLineaire.depuis_la_regle(lambda x, y, _z: (x, y, 0), 3)
    assert projection.noyau() == ((0, 0, 1),)
    assert projection.image() == ((1, 0, 0), (0, 1, 0))


def test_le_noyau_de_l_exemple_1() -> None:
    """`Ker f` de dimension 1, de base `1 - 3X - X^2 + 2X^4` (pour `alpha = 2`)."""
    regle = EXEMPLE_1.image_de((1, 0, 0, 0, 0))
    assert matrice_d_un_vecteur(regle, 2, 2) == M([[1, 0], [1, 0]])
    noyau = EXEMPLE_1.noyau()
    assert noyau == ((R(1, 2), -R(3, 2), -R(1, 2), 0, 1),)
    assert sp.expand(2 * polynome_d_un_vecteur(noyau[0])) == (
        1 - 3 * X - X**2 + 2 * X**4
    )


def test_l_image_de_l_exemple_1() -> None:
    """`dim Im f = 5 - 1 = 4` : `Im f` est `R^(2x2)` tout entier."""
    assert EXEMPLE_1.rang() == 5 - len(EXEMPLE_1.noyau()) == 4
    images = [
        EXEMPLE_1.image_de(tuple(int(i == j) for j in range(5))) for i in range(5)
    ]
    attendues = ([[1, 0], [1, 0]], [[0, 1], [0, 0]], [[1, -1], [3, 0]])
    for image, attendue in zip(images, attendues, strict=False):
        assert matrice_d_un_vecteur(image, 2, 2) == M(attendue)
    assert matrice_d_un_vecteur(images[3], 2, 2) == M([[0, 0], [0, 1]])
    assert matrice_d_un_vecteur(images[4], 2, 2) == M([[0, 1], [1, 0]])
    officielle = (images[0], images[1], images[3], images[4])
    assert est_une_base(officielle)
    assert est_une_base(EXEMPLE_1.image())
    assert EXEMPLE_1.est_surjective()
    assert not EXEMPLE_1.est_injective()


def test_le_noyau_de_l_exemple_2_selon_a() -> None:
    """`a = -2` : `(2 alpha, 0, alpha)` ; `a = 1` : `(alpha, -alpha, 0)` ; sinon nul."""
    assert rangs_selon_le_parametre(EXEMPLE_2, A) == (3, {-2: 2, 1: 2})
    assert ApplicationLineaire(EXEMPLE_2.subs(A, -2)).noyau() == ((2, 0, 1),)
    assert ApplicationLineaire(EXEMPLE_2.subs(A, 1)).noyau() == ((-1, 1, 0),)
    assert ApplicationLineaire(EXEMPLE_2).noyau() == ()
    assert ApplicationLineaire(EXEMPLE_2.subs(A, 5)).est_injective()


def test_l_image_de_l_exemple_2_selon_a() -> None:
    """Les bases extraites de la page 102, pour `a = -2` puis `a = 1`."""
    moins_deux = ApplicationLineaire(EXEMPLE_2.subs(A, -2)).image()
    polynomes = tuple(polynome_d_un_vecteur(v) for v in moins_deux)
    assert polynomes == (1 - X**2 + X**3, -2 - 3 * X - X**2 + X**3)
    un = ApplicationLineaire(EXEMPLE_2.subs(A, 1)).image()
    polynomes = tuple(polynome_d_un_vecteur(v) for v in un)
    assert polynomes == (1 - X**2 + X**3, 1 + 2 * X**2 + X**3)
    assert len(ApplicationLineaire(EXEMPLE_2).image()) == 3


def test_la_dimension_du_noyau_plus_celle_de_l_image() -> None:
    """`dim Ker f + dim Im f = dim V`, sur les deux exemples."""
    applications = [EXEMPLE_1] + [
        ApplicationLineaire(EXEMPLE_2.subs(A, valeur)) for valeur in (-2, 1, 3)
    ]
    for f in applications:
        assert len(f.noyau()) + len(f.image()) == f.matrice.cols


def test_la_reponse_de_l_exercice_1() -> None:
    """a) lineaire ; b) et c) non."""
    u, v = M(sp.symbols("u1:3")), M(sp.symbols("v1:3"))
    assert respecte_les_combinaisons(lambda p: M([2 * p[0] + p[1], p[1]]), u, v)
    assert not respecte_les_combinaisons(lambda p: p + M([0, -1, 0]), U, V)
    assert not respecte_les_combinaisons(lambda p: p[0] * p[1], u, v)


@pytest.mark.parametrize(
    ("antecedents", "images", "attendues"),
    [
        (((3, 1), (-1, 0)), ((1, 2), (1, 1)), ((-1, -1), (4, 5))),
        (((4, 1), (1, 1)), ((1, 1), (3, -2)), ((-R(2, 3), 1), (R(11, 3), -3))),
        (((1, 1), (-1, 1)), ((2, 1), (6, 3)), ((-2, -1), (4, 2))),
    ],
    ids=["a", "b", "c"],
)
def test_la_reponse_de_l_exercice_2(
    antecedents: tuple[tuple[int, ...], ...],
    images: tuple[tuple[int, ...], ...],
    attendues: tuple[tuple[sp.Expr, ...], ...],
) -> None:
    """`f(1, 0)` et `f(0, 1)` se deduisent des images d'une base."""
    f = ApplicationLineaire.depuis_les_images(antecedents, images)
    assert (f.image_de((1, 0)), f.image_de((0, 1))) == attendues
    for antecedent, image in zip(antecedents, images, strict=True):
        assert f.image_de(antecedent) == image


def test_la_reponse_de_l_exercice_3() -> None:
    """`f1`, `f3`, `f4` lineaires ; `f2 : p -> p + q` seulement si `q = 0`."""
    p, r = _generique("p", 3), _generique("r", 3)
    assert respecte_les_combinaisons(lambda s: s * (X**2 + 1), p, r)
    assert not respecte_les_combinaisons(lambda s: s + X + 1, p, r)
    assert respecte_les_combinaisons(lambda s: s + 0, p, r)

    def taylor(s: sp.Expr) -> sp.Expr:
        return s.subs(X, 1) + (X - 1) * sp.diff(s, X).subs(X, 1)

    assert respecte_les_combinaisons(taylor, p, r)
    assert respecte_les_combinaisons(lambda s: s.subs(X, X + 2), p, r)


def test_la_reponse_de_l_exercice_4() -> None:
    """L'evaluation en `a` est lineaire ; la conjugaison l'est sur `R`, pas sur `C`."""
    g, h = sp.Function("g")(TEMPS), sp.Function("h")(TEMPS)
    assert respecte_les_combinaisons(lambda f: f.subs(TEMPS, A), g, h)
    z, w = sp.symbols("z w")
    reels = sp.symbols("alpha beta", real=True)
    assert respecte_les_combinaisons(sp.conjugate, z, w, reels)
    assert not respecte_les_combinaisons(sp.conjugate, z, w)


def test_la_reponse_de_l_exercice_5() -> None:
    """a) `Ker f = {(x1, 0, x3, 0)}`, `Im f = R^2` ; b) noyau nul ; c) pair, impair."""
    f = ApplicationLineaire.depuis_la_regle(lambda _a, b, _c, d: (b, d), 4)
    assert f.noyau() == ((1, 0, 0, 0), (0, 0, 1, 0))
    assert f.est_surjective()
    conjugaison = ApplicationLineaire.depuis_la_regle(lambda x, y: (x, -y), 2)
    assert conjugaison.noyau() == ()
    assert conjugaison.est_surjective()

    def partie_impaire(h: sp.Expr) -> sp.Expr:
        return (h - h.subs(TEMPS, -TEMPS)) / 2

    assert partie_impaire(sp.cos(TEMPS) + TEMPS**2) == 0
    image = partie_impaire(sp.exp(TEMPS))
    assert sp.simplify(image + image.subs(TEMPS, -TEMPS)) == 0
    assert sp.simplify(partie_impaire(image) - image) == 0


def test_la_reponse_de_l_exercice_6() -> None:
    """Non : `(1,1,3) - 2 (1,1,2) + (1,1,1) = 0`, et les images ne suivent pas."""
    antecedents = ((1, 1, 3), (1, 1, 2), (1, 1, 1))
    images = ((1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 0, 0))
    assert not existe_une_application_lineaire(antecedents, images)
    assert M(antecedents[0]) - 2 * M(antecedents[1]) + M(antecedents[2]) == sp.zeros(
        3, 1
    )
    assert M(images[0]) - 2 * M(images[1]) + M(images[2]) == M([0, 0, -1, 1])
    compatibles = ((1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 1, -1))
    assert existe_une_application_lineaire(antecedents, compatibles)
    with pytest.raises(ValueError, match="ne forment pas une base"):
        ApplicationLineaire.depuis_les_images(antecedents, images)


def test_la_reponse_de_l_exercice_7() -> None:
    """`f(x, y, z) = (x, x, x, x + y + z)`."""
    f = ApplicationLineaire.depuis_les_images(
        ((1, 1, 1), (1, 1, 0), (1, 0, 0)),
        ((1, 1, 1, 3), (1, 1, 1, 2), (1, 1, 1, 1)),
    )
    x, y, z = sp.symbols("x y z")
    assert f.image_de((x, y, z)) == (x, x, x, x + y + z)


def test_la_reponse_de_l_exercice_8() -> None:
    """`Ker D` : les constantes ; `Ker(D - I)` : `C e^t` ; `Ker(D - aI)` : `C e^at`."""
    assert OperateurDifferentiel(D).base_des_solutions() == (1,)
    assert OperateurDifferentiel(D - 1).base_des_solutions() == (sp.exp(TEMPS),)
    assert OperateurDifferentiel(D - 3).base_des_solutions() == (sp.exp(3 * TEMPS),)


def test_la_reponse_de_l_exercice_9() -> None:
    """a), b), d) inversibles, d'inverses officiels ; c) non inversible."""
    a, b, c = sp.symbols("a b c")
    f = ApplicationLineaire.depuis_la_regle(lambda x, y: (x + y, x - y), 2)
    assert f.inverse().image_de((a, b)) == (a / 2 + b / 2, a / 2 - b / 2)
    g = ApplicationLineaire.depuis_la_regle(lambda x, y: (2 * x + y, 3 * x - 5 * y), 2)
    attendue = ((5 * a + b) / 13, (3 * a - 2 * b) / 13)
    assert all(
        sp.simplify(o - e) == 0
        for o, e in zip(g.inverse().image_de((a, b)), attendue, strict=True)
    )
    h = ApplicationLineaire.depuis_la_regle(
        lambda x, y, z: (x - y, x + z, x + y + 2 * z), 3
    )
    with pytest.raises(ValueError, match="application non inversible"):
        h.inverse()
    k = ApplicationLineaire.depuis_la_regle(
        lambda x, y, z: (2 * x - y + z, x + y, 3 * x + y + z), 3
    )
    assert k.inverse().image_de((a, b, c)) == (
        a + 2 * b - c,
        -a - b + c,
        -2 * a - 5 * b + 3 * c,
    )


def test_ni_l_injection_seule_ni_la_surjection_seule_ne_s_inversent() -> None:
    """`R^2 -> R^3` injective sans etre surjective, `R^3 -> R^2` l'inverse."""
    injective = ApplicationLineaire(M([[1, 0], [0, 1], [0, 0]]))
    assert injective.est_injective()
    assert not injective.est_surjective()
    with pytest.raises(ValueError, match="application non inversible"):
        injective.inverse()
    surjective = ApplicationLineaire(M([[1, 0, 0], [0, 1, 0]]))
    assert surjective.est_surjective()
    assert not surjective.est_injective()
    with pytest.raises(ValueError, match="application non inversible"):
        surjective.inverse()


@pytest.mark.parametrize("k", [1, 2, 4, 5, 6])
def test_la_reponse_de_l_exercice_10(k: int) -> None:
    """Sur `R[X]<=4` : `Ker D^k = R[X]<=k-1`, `Im D^k = R[X]<=4-k`, ou tout et rien."""
    n = 4
    colonnes = [vecteur_d_un_polynome(sp.diff(X**j, X), n) for j in range(n + 1)]
    derivee = ApplicationLineaire(M(colonnes).T)
    puissance = ApplicationLineaire(derivee.matrice**k)
    noyau = tuple(
        tuple(int(i == j) for j in range(n + 1)) for i in range(min(k, n + 1))
    )
    assert puissance.noyau() == noyau
    assert len(puissance.image()) == max(n + 1 - k, 0)
    for vecteur in puissance.image():
        assert sp.degree(polynome_d_un_vecteur(vecteur), X) <= n - k


@pytest.mark.parametrize("n", [2, 3])
def test_la_reponse_de_l_exercice_11(n: int) -> None:
    """`A -> (A + A^t)/2` : noyau antisymetrique, image symetrique ; et l'inverse."""
    unites = [
        matrice_d_un_vecteur(tuple(int(i == j) for j in range(n * n)), n, n)
        for i in range(n * n)
    ]
    f1 = ApplicationLineaire(
        M([vecteur_d_une_matrice((u + u.T) / 2) for u in unites]).T
    )
    f2 = ApplicationLineaire(
        M([vecteur_d_une_matrice((u - u.T) / 2) for u in unites]).T
    )
    assert len(f1.noyau()) == len(f2.image()) == n * (n - 1) // 2
    assert len(f1.image()) == len(f2.noyau()) == n * (n + 1) // 2
    for vecteur in (*f1.noyau(), *f2.image()):
        assert est_antisymetrique(matrice_d_un_vecteur(vecteur, n, n))
    for vecteur in (*f1.image(), *f2.noyau()):
        assert est_symetrique(matrice_d_un_vecteur(vecteur, n, n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_la_reponse_de_l_exercice_12(n: int) -> None:
    """La trace est lineaire ; les matrices de trace nulle : dimension `n^2 - 1`."""
    u = M(n, n, sp.symbols(f"u1:{n * n + 1}"))
    v = M(n, n, sp.symbols(f"v1:{n * n + 1}"))
    assert respecte_les_combinaisons(trace, u, v)
    forme = ApplicationLineaire(M([vecteur_d_une_matrice(sp.eye(n))]))
    assert len(forme.noyau()) == n * n - 1


def test_la_reponse_de_l_exercice_13() -> None:
    """Rang 1 si `t = -1`, 2 si `t = 1`, 3 sinon ; bases de l'image et completions."""
    matrice = M([[1, T, 1], [T, 1, T], [1, T, -T]])
    assert rangs_selon_le_parametre(matrice, T) == (3, {-1: 1, 1: 2})
    moins_un = ApplicationLineaire(matrice.subs(T, -1)).image()
    assert moins_un == ((1, -1, 1),)
    assert est_une_base((*moins_un, (0, 1, 0), (0, 0, 1)))
    un = ApplicationLineaire(matrice.subs(T, 1)).image()
    assert rang_de_la_famille(un) == 2
    assert engendrent_le_meme_sous_espace(un, ((1, 1, 1), (1, 1, -1)))
    assert est_une_base(((1, 1, 1), (1, 1, -1), (0, 1, 0)))
