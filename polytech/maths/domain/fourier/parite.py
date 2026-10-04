"""Les fonctions paires et impaires, et les prolongements qui en fabriquent.

`f` est **paire** si `f(-x) = f(x)`, **impaire** si `f(-x) = -f(x)`. Une
fonction periodique paire n'a que des cosinus (`b_n = 0`), une impaire que
des sinus (`a_n = 0`).

D'ou le moyen de developper une fonction donnee sur `[0, A]` seulement :
on la prolonge sur `[-A, 0]` de facon impaire pour une serie de sinus, ou
paire pour une serie de cosinus, et l'on developpe avec `T = 2 A`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.symboles import VARIABLE


def est_paire(fonction: sp.Expr) -> bool:
    """Vrai si `f(-x) = f(x)`."""
    return bool(sp.simplify(fonction.subs(VARIABLE, -VARIABLE) - fonction) == 0)


def est_impaire(fonction: sp.Expr) -> bool:
    """Vrai si `f(-x) = -f(x)`."""
    return bool(sp.simplify(fonction.subs(VARIABLE, -VARIABLE) + fonction) == 0)


def _prolongement(fonction: sp.Expr, borne: object, signe: int) -> SerieDeFourier:
    miroir = signe * fonction.subs(VARIABLE, -VARIABLE)
    motif = sp.Piecewise((miroir, VARIABLE < 0), (fonction, True))
    demi = sp.sympify(borne)
    return SerieDeFourier(motif, -demi, 2 * demi)


def prolongement_pair(fonction: sp.Expr, borne: object) -> SerieDeFourier:
    """`f(-x) = f(x)` sur `[-A, 0]` : la serie n'a que des cosinus."""
    return _prolongement(fonction, borne, 1)


def prolongement_impair(fonction: sp.Expr, borne: object) -> SerieDeFourier:
    """`f(-x) = -f(x)` sur `[-A, 0]` : la serie n'a que des sinus."""
    return _prolongement(fonction, borne, -1)
