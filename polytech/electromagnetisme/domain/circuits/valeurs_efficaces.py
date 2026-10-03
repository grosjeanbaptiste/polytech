"""La valeur efficace d'un signal periodique, par sa definition integrale.

`X = racine((1/T) int_0^T x^2 dt)` : la tension continue qui dissiperait
la meme puissance dans une resistance. Pour un sinus elle vaut
`x_m / racine 2`, et c'est la seule forme pour laquelle ce rapport vaut :
le carre a la meme valeur efficace que sa crete, le triangle sa crete sur
racine de trois. Diviser toute crete par racine de deux est l'erreur que
le labo I fait commettre a qui ne repasse pas par la definition.

Les trois formes sont symetriques, de moyenne nulle, et de crete `A`,
comme l'enonce les regle au generateur. Chacune est decrite sur une
periode `[0, T)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS


def sinus(amplitude: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`A sin(2 pi t / T)`."""
    return amplitude * sp.sin(2 * sp.pi * TEMPS / periode)


def carre(amplitude: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`A` sur la premiere demi-periode, `-A` sur la seconde."""
    return sp.Piecewise((amplitude, periode / 2 > TEMPS), (-amplitude, True))


def triangle(amplitude: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """Monte de `0` a `A`, redescend a `-A`, remonte a `0`, par droites."""
    pente = 4 * amplitude / periode
    return sp.Piecewise(
        (pente * TEMPS, periode / 4 > TEMPS),
        (2 * amplitude - pente * TEMPS, 3 * periode / 4 > TEMPS),
        (pente * TEMPS - 4 * amplitude, True),
    )


def valeur_efficace(signal: sp.Expr, periode: sp.Expr) -> sp.Expr:
    """`racine((1/T) int_0^T x^2 dt)` ; leve si la periode est nulle."""
    if sp.simplify(periode) == 0:
        msg = "periode nulle : la moyenne sur une periode n'a pas de sens"
        raise ValueError(msg)
    carre_moyen = sp.integrate(signal**2, (TEMPS, 0, periode)) / periode
    return sp.simplify(sp.sqrt(sp.simplify(carre_moyen)))
