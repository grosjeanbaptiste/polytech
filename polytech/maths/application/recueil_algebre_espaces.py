r"""Les ecritures d'un vecteur du recueil d'algebre : uplet, matrice, polynome.

Les chapitres VII a X manipulent la meme suite de nombres sous trois
formes, par la bijection que le syllabus pose a ses exemples 10 et 11 :
une matrice 2 x 2 se lit ligne apres ligne, un polynome du terme
independant au plus haut degre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.recueil_algebre import ecriture

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

INDETERMINEE = sp.Symbol("X")
ORDRE = 2
"""Les matrices du recueil sont toutes 2 x 2."""


def vecteur(composantes: Sequence[object]) -> str:
    r"""`\left(1,\, -2\right)`."""
    interieur = r",\, ".join(ecriture(sp.sympify(c)) for c in composantes)
    return rf"\left({interieur}\right)"


def polynome(coefficients: Sequence[object]) -> str:
    """Le polynome dont voici les coefficients, du terme independant au plus haut."""
    termes = [sp.sympify(c) * INDETERMINEE**k for k, c in enumerate(coefficients)]
    return ecriture(sp.Add(*termes))


def matrice(termes: Sequence[object]) -> str:
    """La matrice 2 x 2 dont voici les termes, ligne apres ligne."""
    if len(termes) != ORDRE * ORDRE:
        msg = f"{len(termes)} termes : une matrice 2 x 2 en a quatre"
        raise ValueError(msg)
    return ecriture(sp.Matrix(ORDRE, ORDRE, [sp.sympify(t) for t in termes]))


def famille(
    vecteurs: Iterable[Sequence[object]],
    forme: Callable[[Sequence[object]], str] = vecteur,
) -> str:
    r"""`\left\{ ... \right\}`, chaque element compose par `forme`."""
    interieur = r",\; ".join(forme(v) for v in vecteurs) or r"\;"
    return rf"\left\{{{interieur}\right\}}"


def _combinaison(coefficients: Sequence[object], noms: Sequence[sp.Symbol]) -> sp.Expr:
    paires = zip(coefficients, noms, strict=True)
    return sp.Add(*[sp.sympify(c) * n for c, n in paires])


def relation(coefficients: Sequence[object], lettre: str) -> str:
    """`3 X_{1} - 2 X_{2} + X_{3} = 0` : une combinaison lineaire nulle."""
    noms = sp.symbols(f"{lettre}1:{len(coefficients) + 1}")
    return f"{ecriture(_combinaison(coefficients, noms))} = 0"


def equation(coefficients: Sequence[object], inconnues: str) -> str:
    """`t - y = 0` : une equation lineaire homogene en ces inconnues."""
    noms = sp.symbols(" ".join(inconnues))
    return f"{ecriture(_combinaison(coefficients, noms))} = 0"
