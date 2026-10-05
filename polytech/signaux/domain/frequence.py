"""La transformee de Fourier en notation `f`, et les signaux qui l'illustrent.

Syllabus de signaux et systemes, section 2.1. Un signal d'energie finie
se decompose sur les exponentielles imaginaires, base orthonormee :

`f(t) = integrale de F(f) exp(j 2 pi f t) df`,
`F(f) = <f(t), exp(j omega t)> = integrale de f(t) exp(-j 2 pi f t) dt`.

La notation en `f` garde la symetrie entre la transformee et son inverse ;
celle en `omega = 2 pi f` allege l'ecriture et coute un facteur
`1 / (2 pi)` a l'inverse. Passer de l'une a l'autre ne change que l'axe,
sauf pour l'impulsion de Dirac, dont le poids est une surface :
`delta(f) = 2 pi delta(omega)`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.fourier.transformee import integrale_sur_la_droite

INSTANT: Final = sp.Symbol("t", real=True)
"""Le temps sur toute la droite : Fourier ne suppose pas le signal causal."""

FREQUENCE: Final = sp.Symbol("f", real=True)
"""La frequence, en hertz."""

DEMI: Final = sp.Rational(1, 2)


def rect(variable: sp.Expr) -> sp.Expr:
    """La porte de largeur et de hauteur un, centree."""
    return sp.Piecewise((1, sp.Abs(variable) <= DEMI), (0, True))


def tri(variable: sp.Expr) -> sp.Expr:
    """Le triangle de hauteur un et de base deux : `rect * rect`."""
    return sp.Piecewise((1 - sp.Abs(variable), sp.Abs(variable) <= 1), (0, True))


def sinc(variable: sp.Expr) -> sp.Expr:
    """`sin(pi x) / (pi x)`, prolonge par un en zero."""
    return sp.sinc(sp.pi * variable)


def transformee_en_frequence(signal: sp.Expr) -> sp.Expr:
    """`F(f) = integrale de f(t) exp(-j 2 pi f t) dt`."""
    noyau = sp.exp(-2 * sp.pi * sp.I * FREQUENCE * INSTANT)
    image = sp.simplify(integrale_sur_la_droite(signal * noyau, INSTANT))
    return image.args[0][0] if isinstance(image, sp.Piecewise) else image


def signal_depuis_le_spectre(spectre: sp.Expr) -> sp.Expr:
    """`f(t) = integrale de F(f) exp(j 2 pi f t) df`."""
    noyau = sp.exp(2 * sp.pi * sp.I * FREQUENCE * INSTANT)
    signal = sp.simplify(integrale_sur_la_droite(spectre * noyau, FREQUENCE))
    return signal.args[0][0] if isinstance(signal, sp.Piecewise) else signal


def en_frequence(image: sp.Expr, pulsation: sp.Symbol) -> sp.Expr:
    """L'image en `omega` recrite en `f` ; les Dirac y changent de poids."""
    recrite = image.subs(pulsation, 2 * sp.pi * FREQUENCE)
    return sp.simplify(recrite.expand(diracdelta=True, wrt=FREQUENCE))
