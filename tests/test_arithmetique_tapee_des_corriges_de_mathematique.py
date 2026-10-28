"""Les evaluations que les corriges de mathematique ecrivent a la main, recalculees.

Un corrige n'ecrit aucun nombre calculable en clair (AGENTS.md). Les
evaluations de polynomes de `algebre-04-polynomes` et les valeurs de
`examen-2025-janvier` et d'`exs-02-equations-differentielles` sont pourtant
tapees dans le texte : une macro n'y rendrait pas la somme terme a terme que
le lecteur doit suivre. Cette garde les refait : chaque terme tape est celui
du polynome, la somme est celle que le texte annonce, et le tout vient de
sympy -- ce que le texte ecrit ne peut donc plus deriver sans que le test
rougisse.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import examen_2025 as vingt_cinq
from polytech.maths.domain.symboles import IMAGE as P

X = sp.Symbol("x")
R = sp.Rational

# (polynome, point, termes tapes dans le corrige, somme annoncee)
EVALUATIONS = [
    (
        3 * X**3 - 19 * X**2 + 34 * X - 14,
        R(7, 3),
        [R(343, 9), R(-931, 9), R(714, 9), R(-126, 9)],
        0,
    ),
    (6 * X**4 - 5 * X**3 - 38 * X**2 - 5 * X + 6, -2, [96, 40, -152, 10, 6], 0),
    (6 * X**4 - 5 * X**3 - 38 * X**2 - 5 * X + 6, 3, [486, -135, -342, -15, 6], 0),
    (X**4 + X**3 - 11 * X**2 + X - 12, 3, [81, 27, -99, 3, -12], 0),
    (X**4 + X**3 - 11 * X**2 + X - 12, -4, [256, -64, -176, -4, -12], 0),
    (6 * X**4 - 13 * X**3 + 13 * X - 6, 1, [6, -13, 13, -6], 0),
    (6 * X**4 - 13 * X**3 + 13 * X - 6, -1, [6, 13, -13, -6], 0),
]


@pytest.mark.parametrize(("poly", "point", "termes", "somme"), EVALUATIONS)
def test_chaque_terme_tape_est_celui_du_polynome(
    poly: sp.Expr, point: sp.Expr | int, termes: list[sp.Expr | int], somme: int
) -> None:
    p = sp.Poly(poly, X)
    calcules = [
        c * sp.sympify(point) ** k for (k,), c in sorted(p.terms(), reverse=True)
    ]
    # le corrige omet les termes nuls ; on compare les non nuls, du plus haut degre
    assert [sp.nsimplify(t) for t in calcules if t != 0] == [
        sp.nsimplify(t) for t in termes
    ]
    assert sum(termes) == somme


def test_le_discriminant_tape_au_chapitre_quatre_vaut_25() -> None:
    assert sp.discriminant(6 * X**2 - 13 * X + 6, X) == 169 - 144 == 25


def test_les_valeurs_tapees_de_la_decomposition_de_2025() -> None:
    """`Y(p)` : numerateur 3087 et -216 aux poles 7 et -2, residus 1 et -3."""
    image = sp.together(vingt_cinq.equation()[0])
    numerateur, denominateur = sp.fraction(sp.factor(image))
    assert sp.expand(denominateur) == sp.expand(P**3 * (P - 7) * (P + 2))
    assert numerateur.subs(P, 7) == 13 * 343 - 26 * 49 - 70 - 28 == 3087
    assert numerateur.subs(P, -2) == -104 - 104 + 20 - 28 == -216
    assert (P**3 * (P + 2)).subs(P, 7) == 343 * 9 == 3087
    assert (P**3 * (P - 7)).subs(P, -2) == (-8) * (-9) == 72
    assert sp.residue(image, P, 7) == 1
    assert sp.residue(image, P, -2) == -3
    serie = sp.series(
        image * P**3, P, 0, 3
    ).removeO()  # coefficients de p^-3, p^-2, p^-1
    assert [serie.coeff(P, k) for k in (0, 1, 2)] == [2, 0, 2]  # 1/p^2 est absent


def test_les_factorielles_tapees_de_l_exercice_deux() -> None:
    assert (sp.factorial(7), sp.factorial(5)) == (5040, 120)
