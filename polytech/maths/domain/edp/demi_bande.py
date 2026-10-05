"""La methode de Fourier sur un domaine non borne : la demi-bande.

On cherche `u(x, y)` harmonique et bornee sur `x >= 0`, `0 <= y <= 1`,
de derivees normales nulles en `x = 0` et en `y = 0`, valant `f(x)` en
`y = 1`. La separation donne `a'' / a = -b'' / b = -lambda^2` -- negatif,
pour rester borne -- et les bords ne laissent que

    cos(lambda x) ch(lambda y),

pour **tout** `lambda` : les valeurs propres forment un ensemble continu,
et la serie devient une integrale. La condition en `y = 1` est une
transformee de Fourier en cosinus :

    u = int_0^oo A(lambda) cos(lambda x) ch(lambda y) dlambda,
    A(lambda) = (2 / pi) (1 / ch lambda) int_0^oo f(t) cos(lambda t) dt.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.edp.operateurs import ORDONNEE
from polytech.maths.domain.fourier.cosinus import transformee_en_cosinus
from polytech.maths.domain.symboles import VARIABLE


def mode_de_la_demi_bande(valeur: object) -> sp.Expr:
    """`cos(lambda x) ch(lambda y)`."""
    lam = sp.sympify(valeur)
    return sp.cos(lam * VARIABLE) * sp.cosh(lam * ORDONNEE)


def amplitude_de_la_demi_bande(bord: sp.Expr, valeur: sp.Symbol) -> sp.Expr:
    """`A(lambda) = (2/pi) F_c(lambda) / ch(lambda)`."""
    en_cosinus = transformee_en_cosinus(bord, valeur)
    return sp.simplify(2 * en_cosinus / (sp.pi * sp.cosh(valeur)))
