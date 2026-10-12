"""Recueil d'algebre, chapitre VIII, section VIII.8 : treize sujets d'examen.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 33 a 45
(pages 117 a 123). Chaque application est sa matrice dans les bases
canoniques : une matrice 2 x 2 (ou 2 x 3) se lit ligne apres ligne, un
polynome du terme independant au plus haut degre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire

if TYPE_CHECKING:
    from collections.abc import Callable

M = sp.Matrix
X: Final = sp.Symbol("X")
T: Final = sp.Symbol("t")
A, B = sp.symbols("a b")
ALPHA, BETA = sp.symbols("alpha beta")
_REGLE = ApplicationLineaire.depuis_la_regle
M_T: Final = M([[T, 1], [1, T]])
V_43: Final = M([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
W_43: Final = M([[1, 1, 1], [0, 1, 0], [0, 0, 1]])
ASSOCIEE_43: Final = M([[-1, -2, A - 4, A - 3], [1, 2, 4, 3], [0, A - 2, A - 2, 0]])
"""La matrice du 43 dans les bases `v` et `w`, dont `V_43` et `W_43` portent
les vecteurs en colonnes."""


def _du_33() -> sp.Matrix:
    """`p -> 2 (X + 1)(p + p') - (X + 1)^2 (p' + p'')`, de `P_2` dans `P_3`."""
    colonnes = []
    for k in range(3):
        p = X**k
        derivee, seconde = sp.diff(p, X), sp.diff(p, X, 2)
        image = sp.expand(
            2 * (X + 1) * (p + derivee) - (X + 1) ** 2 * (derivee + seconde)
        )
        colonnes.append([sp.Poly(image, X).coeff_monomial(X**j) for j in range(4)])
    return M(colonnes).T


def _sur_les_matrices(regle: Callable[[sp.Matrix], sp.Matrix]) -> sp.Matrix:
    return _REGLE(lambda *termes: list(regle(M(2, 2, termes))), 4).matrice


APPLICATIONS: Final = {
    33: _du_33(),
    34: _REGLE(lambda a, b, c, d: (a + b + c, 2 * b + d, d, d), 4).matrice,
    35: _sur_les_matrices(lambda m: m * M_T + M_T * m),
    36: _sur_les_matrices(lambda m: A * m + B * m.T),
    37: _REGLE(
        lambda p, q, r, s: (
            BETA * q + ALPHA * s,
            ALPHA * p + BETA * r,
            q + s,
            ALPHA * p + BETA * r,
            ALPHA * q + BETA * s,
            BETA * p + ALPHA * r,
        ),
        4,
    ).matrice,
    38: _REGLE(
        lambda a0, a1, a2, a3, a4, a5: (
            A * (a2 + a4) + B * a0,
            a3 + A * a1 + B * a5,
            A * a0 + B * (a2 + a4),
            a3 + A * a5 + B * a1,
        ),
        6,
    ).matrice,
    39: M([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4], [1, -1, 1, -1, 1], [0, 1, -2, 3, -4]]),
    40: M([[4, 2, -2, -4], [-1, 1, 4, -4]]),
    41: M([[-1, 0, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 0], [-1, 0, 0, 1]]),
    42: M([[1, -2, -2], [2, 1, 5], [4, -3, 2]]),
    43: sp.simplify(W_43 * ASSOCIEE_43 * V_43.inv()),
    44: M([[0, 1, 2, 0], [1, 0, 0, -1], [0, 1, 1, 1]]),
    45: M([[1, 0, 0, 0], [1, 0, 0, 1], [0, -1, 1, 2]]),
}
"""Numero de l'exercice : la matrice de `f` dans les bases canoniques."""

CAS: Final[dict[int, dict[str, dict[sp.Symbol, sp.Expr]]]] = {
    35: {"TZero": {T: 0}, "TUn": {T: 1}, "TMoinsUn": {T: -1}},
    36: {"Opposes": {B: -A}, "Egaux": {B: A}},
    37: {"Egaux": {BETA: ALPHA}, "Opposes": {BETA: -ALPHA}},
    38: {"Egaux": {B: A}, "Opposes": {B: -A}},
    43: {"AZero": {A: 0}, "ADeux": {A: 2}},
}
"""Les cas singuliers de chaque discussion, par leur cle."""
