"""Chapitre VII du recueil d'algebre : sous-espaces ou non, et demonstrations.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1, 2, 11,
20, 21, 24, 28, 30, 39 a 43. Un « non » se prouve par un contre-exemple, et
chaque contre-exemple que le corrige avance est rejoue ici ; un « oui » se
prouve par la stabilite, verifiee sur des elements generaux.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch07_enonces as enonces
from polytech.maths.domain.algebre.familles_de_vecteurs import est_libre
from polytech.maths.domain.algebre.sous_espaces import (
    engendrent_le_meme_sous_espace,
    sont_supplementaires,
)

M = sp.Matrix
V = enonces.VECTEURS
T = sp.Symbol("t", real=True)


def test_exercice_1_les_trois_non() -> None:
    """B, C et D : le vecteur nul manque, ou une somme sort de l'ensemble."""
    nul = (0, 0, 0)
    assert not nul[0] < nul[1]
    assert sum(x**2 for x in nul) != 1
    u, v = M([1, 0, 0]), M([0, 1, 0])
    assert u[0] * u[1] == 0
    assert v[0] * v[1] == 0
    assert (u + v)[0] * (u + v)[1] != 0


def test_exercice_1_les_quatre_oui_sont_des_noyaux() -> None:
    """A, E, F, G sont definis par des equations lineaires homogenes."""
    a, b, k = sp.symbols("a b k")
    for equations in (
        M([[1, -1, 0]]),
        M([[1, 1, 1]]),
        M([[1, -1, 0], [0, 2, -1]]),
        M([[1, 1, -3]]),
    ):
        combinaisons = zip((a, b), equations.nullspace(), strict=False)
        general = sum((c * w for c, w in combinaisons), sp.zeros(3, 1))
        produit = sp.expand(equations * (k * general))
        assert produit == sp.zeros(equations.rows, 1)


def test_exercice_2_le_second_membre_fait_tout() -> None:
    systeme = M([[1, 1, 1], [2, -1, 0]])
    assert systeme * sp.zeros(3, 1) == sp.zeros(2, 1)
    assert systeme * sp.zeros(3, 1) != M([0, 3])


def test_exercice_11_les_deux_familles_engendrent_le_meme_sous_espace() -> None:
    assert engendrent_le_meme_sous_espace(V["11X"], V["11Y"])


def test_exercices_20_et_21_des_supplementaires() -> None:
    assert sont_supplementaires(V["20W1"], V["20W2"])
    assert sont_supplementaires(V["20W1"], V["20W3"])
    assert sont_supplementaires(V["21W1"], V["21W2"])


def test_exercice_24_les_deux_non() -> None:
    """F ne contient pas la matrice nulle ; G n'est pas stable par addition."""
    assert sp.sqrt(2) != 0
    gauche, droite = M([[1, 0], [0, 0]]), M([[0, 0], [0, 1]])
    assert gauche.det() == 0
    assert droite.det() == 0
    assert (gauche + droite).det() != 0


def test_exercice_24_les_dimensions_des_symetriques_et_des_antisymetriques() -> None:
    a, b, c, d = sp.symbols("a b c d")
    generale = M([[a, b], [c, d]])
    symetrique = sp.solve(list(generale - generale.T), [a, b, c, d], dict=True)[0]
    antisymetrique = sp.solve(list(generale + generale.T), [a, b, c, d], dict=True)[0]
    assert len(generale.subs(symetrique).free_symbols) == 3
    assert len(generale.subs(antisymetrique).free_symbols) == 1


def test_exercice_28_stable_par_un_reel_pas_par_i() -> None:
    a, b = sp.symbols("a b")
    k = sp.Symbol("k", real=True)

    def forme(p: sp.Expr, q: sp.Expr) -> sp.Matrix:
        return M([[p, q], [-sp.conjugate(q), sp.conjugate(p)]])

    assert sp.simplify(k * forme(a, b) - forme(k * a, k * b)) == sp.zeros(2, 2)
    produit = sp.I * forme(1, 0)
    assert produit != forme(produit[0, 0], produit[0, 1])


def test_exercice_28_la_base_du_syllabus_est_libre_sur_r() -> None:
    """En separant parties reelle et imaginaire : quatre vecteurs de `R^8`."""
    base = (
        M([[1, 0], [0, 1]]),
        M([[sp.I, 0], [0, -sp.I]]),
        M([[0, 1], [-1, 0]]),
        M([[0, sp.I], [sp.I, 0]]),
    )
    reels = tuple(tuple(sp.re(x) for x in m) + tuple(sp.im(x) for x in m) for m in base)
    assert est_libre(reels)


def test_exercice_30_les_deux_non() -> None:
    x = sp.Symbol("X")
    assert sp.Integer(0).subs(x, 0) != 1
    n = 3
    somme = x**n + (-(x**n) + 1)
    assert sp.degree(sp.expand(somme), x) != n


def test_exercice_30_pair_et_impair_sont_stables() -> None:
    x, a, b, c, k = sp.symbols("X a b c k")
    pair = a + b * x**2
    impair = a * x + b * x**3
    assert sp.expand((k * pair + c).subs(x, -x) - (k * pair + c)) == 0
    assert sp.expand((k * impair).subs(x, -x) + k * impair) == 0


@pytest.mark.parametrize(
    ("famille", "instants"),
    [
        ((sp.sin(T), sp.cos(T)), (0, sp.pi / 2)),
        ((sp.sin(T), sp.sin(2 * T)), (sp.pi / 2, sp.pi / 4)),
        ((sp.exp(T), sp.log(T)), (1, sp.E)),
    ],
)
def test_exercices_39_et_40_deux_instants_suffisent(
    famille: tuple[sp.Expr, sp.Expr], instants: tuple[sp.Expr, sp.Expr]
) -> None:
    """La matrice des valeurs en deux instants est inversible."""
    valeurs = M(2, 2, lambda i, j: famille[j].subs(T, instants[i]))
    assert sp.simplify(valeurs.det()) != 0


def test_exercice_41_les_cinq_non() -> None:
    """B, C, D, F, G : un contre-exemple chacun."""
    nulle, identite = sp.Integer(0), T
    assert nulle.subs(T, 1) != nulle.subs(T, 0) + 1
    assert (-1 * sp.Integer(1)) < 0
    assert sp.diff(-identite, T) < 0
    f, g = T, 1 - T
    assert f.subs(T, 0) == 0
    assert g.subs(T, 1) == 0
    assert sp.solve(sp.Eq(f + g, 0), T) == []
    assert nulle.subs(T, sp.Rational(1, 4)) != sp.Rational(1, 4)


@pytest.mark.parametrize(
    ("coefficients", "libre"),
    [
        (((1, 0, 0), (1, 1, 0), (1, 1, 1)), True),
        (((1, -1, 0), (0, 1, -1), (-1, 0, 1)), False),
        (((1, 1, 0), (0, 1, 1), (1, 0, 1)), True),
    ],
)
def test_exercice_42_dans_la_base_x_y_z(coefficients: tuple, *, libre: bool) -> None:
    assert est_libre(coefficients) is libre


def test_exercice_43_les_coordonnees_dans_les_trois_bases() -> None:
    a = sp.Symbol("a")
    v1, v2 = M([1, 0]), M([0, 1])
    assert v1 == (v1 + v2) - v2
    assert v1 == (v1 + v2) / 2 + (v1 - v2) / 2
    assert v2 == (v1 + v2) / 2 - (v1 - v2) / 2
    assert v2 == -a * v1 + (a * v1 + v2)
    for base in ((v1 + v2, v2), (v1 + v2, v1 - v2), (v1, a * v1 + v2)):
        assert M.hstack(*base).det() != 0
