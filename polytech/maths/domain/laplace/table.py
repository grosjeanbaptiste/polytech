"""La table des transformees de Laplace, et le demi-plan ou chacune vaut.

`Formulaire_Aide_memoire.pdf`, section 7 (page 10) : huit originaux et
leurs images. La table est une **donnee**, recopiee de la page ; un test
recalcule chaque image par l'integrale.

Le formulaire ne dit pas ou l'integrale converge. Le cours le dit
(Mathematiques appliquees, 3.1 et proposition 3.1) : `L(H) = 1/p` « si
Re(p) > 0 », `L(e^(alpha t)) = 1/(p - alpha)` « si Re(p) > alpha ». Ce
seuil est l'**abscisse de convergence** ; elle est rangee ici a cote de
chaque image, pour un `alpha` reel.

La **linearite** (3.3.1) fait le reste : l'image d'une combinaison est la
combinaison des images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.laplace.signaux import echelon
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ALPHA: Final = sp.Symbol("alpha", real=True)
"""Le taux de `e^(alpha t)`, `sinh(alpha t)` et `cosh(alpha t)`."""
OMEGA: Final = sp.Symbol("omega", positive=True)
"""La pulsation de `sin(omega t)` et `cos(omega t)`."""
ORDRE: Final = sp.Symbol("n", integer=True, nonnegative=True)
"""L'exposant de `t^n`."""

TABLE_DE_LAPLACE: Final[dict[sp.Expr, tuple[sp.Expr, sp.Expr]]] = {
    sp.DiracDelta(T): (sp.Integer(1), -sp.oo),
    echelon(): (1 / P, sp.Integer(0)),
    sp.exp(ALPHA * T): (1 / (P - ALPHA), ALPHA),
    T**ORDRE: (sp.factorial(ORDRE) / P ** (ORDRE + 1), sp.Integer(0)),
    sp.sin(OMEGA * T): (OMEGA / (P**2 + OMEGA**2), sp.Integer(0)),
    sp.cos(OMEGA * T): (P / (P**2 + OMEGA**2), sp.Integer(0)),
    sp.sinh(ALPHA * T): (ALPHA / (P**2 - ALPHA**2), sp.Abs(ALPHA)),
    sp.cosh(ALPHA * T): (P / (P**2 - ALPHA**2), sp.Abs(ALPHA)),
}
"""L'original, puis son image et son abscisse de convergence."""


def _ligne(original: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    if original not in TABLE_DE_LAPLACE:
        msg = f"{original} n'est pas dans la table des transformees de Laplace"
        raise ValueError(msg)
    return TABLE_DE_LAPLACE[original]


def image_de_la_table(original: sp.Expr) -> sp.Expr:
    """L'image que la table donne ; leve si l'original n'y figure pas."""
    return _ligne(original)[0]


def abscisse_de_convergence(original: sp.Expr) -> sp.Expr:
    """Le seuil `a` tel que l'integrale converge pour `Re(p) > a`."""
    return _ligne(original)[1]


def par_linearite(
    combinaison: Sequence[tuple[object, sp.Expr, Mapping[sp.Symbol, object]]],
) -> sp.Expr:
    """`L(somme c_i f_i) = somme c_i L(f_i)`, chaque `f_i` pris dans la table.

    Chaque terme donne son coefficient, sa ligne de la table, et les
    valeurs de ses parametres (`{ALPHA: 2}`).
    """
    images = (
        sp.sympify(coefficient) * image_de_la_table(original).subs(valeurs)
        for coefficient, original, valeurs in combinaison
    )
    return sp.simplify(sum(images, sp.Integer(0)))
