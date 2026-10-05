"""La transformee de Fourier d'une fonction non periodique.

Quand la periode tend vers l'infini, les harmoniques `n w` se resserrent
en une variable continue et la serie de Fourier devient une integrale :

    F(w) = int f(x) e^(-j w x) dx,        f(x) = (1 / 2 pi) int F(w) e^(+j w x) dw.

`F = R + j X = A e^(j b)` : `A` est le spectre d'amplitude, `b` le spectre
de phase, `A^2` le spectre d'energie. `F(0)` est l'integrale de `f`.

L'integrale existe si `f` est sommable (conditions de Dirichlet) ou de
carre sommable. En un point ou l'image s'ecrit `0/0` -- la porte en
`w = 0` --, elle se prolonge par continuite : c'est la forme generique qui
est rendue.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

PULSATION: Final = sp.Symbol("omega", real=True)
"""`w`, la variable de l'image."""


def integrale_sur_la_droite(fonction: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`int` sur toute la droite, morceau par morceau si la fonction en a.

    Une impulsion de Dirac s'integre telle quelle : la recrire par morceaux
    en ferait une fonction nulle presque partout, d'integrale nulle.
    """
    brute = sp.sympify(fonction)
    if brute.has(sp.DiracDelta):
        return sp.integrate(brute, (variable, -sp.oo, sp.oo))
    pliee = sp.piecewise_fold(brute.rewrite(sp.Piecewise))
    directe = sp.integrate(pliee, (variable, -sp.oo, sp.oo))
    if not directe.has(sp.Integral):
        return directe
    morceaux = pliee.as_expr_set_pairs(sp.S.Reals)
    return sum(
        (
            sp.integrate(
                expression, (variable, ensemble.inf, ensemble.sup), conds="none"
            )
            for expression, ensemble in morceaux
        ),
        sp.Integer(0),
    )


def _generique(image: sp.Expr) -> sp.Expr:
    """La forme valable hors des points isoles ou l'image se prolonge par continuite."""
    return image.args[0][0] if isinstance(image, sp.Piecewise) else image


def transformee_de_fourier(fonction: sp.Expr) -> sp.Expr:
    """`F(w) = int f(x) e^(-j w x) dx`."""
    noyau = sp.exp(-sp.I * PULSATION * VARIABLE)
    return _generique(sp.simplify(integrale_sur_la_droite(fonction * noyau, VARIABLE)))


def transformee_inverse(image: sp.Expr) -> sp.Expr:
    """`f(x) = (1 / 2 pi) int F(w) e^(+j w x) dw`."""
    noyau = sp.exp(sp.I * PULSATION * VARIABLE)
    integrale = integrale_sur_la_droite(image * noyau, PULSATION)
    return sp.simplify(integrale / (2 * sp.pi))


def spectres(image: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(A, b, A^2)` : amplitude, phase et energie de `F = A e^(j b)`."""
    reelle, imaginaire = sp.re(image), sp.im(image)
    energie = sp.simplify(reelle**2 + imaginaire**2)
    return sp.sqrt(energie), sp.simplify(sp.atan2(imaginaire, reelle)), energie
