"""La desintegration radioactive, et la datation qu'elle permet.

Un nucleide instable se desintegre a un rythme proportionnel a la
quantite presente : `x'(t) = -beta x(t)`. Le cours de mathematique s'en
sert pour illustrer l'unicite d'une solution (datation au carbone 14,
W. F. Libby, 1946) ; la loi, elle, releve de la chimie nucleaire, et
c'est d'ici que le cours de mathematique la tire.

    x(t) = x(t0) exp(-beta (t - t0)).

La **demi-vie** `T = ln 2 / beta` est la duree au bout de laquelle la
quantite a diminue de moitie. Mesurer la fraction restante d'un
echantillon donne donc son age : `t = T log2(x0 / x)`.
"""

from __future__ import annotations

import sympy as sp


def quantite_restante(initiale: object, constante: object, duree: object) -> sp.Expr:
    """`x0 exp(-beta t)` : ce qui reste apres la duree `t`."""
    return sp.sympify(initiale) * sp.exp(-sp.sympify(constante) * sp.sympify(duree))


def demi_vie(constante: object) -> sp.Expr:
    """`T = ln 2 / beta`."""
    return sp.log(2) / sp.sympify(constante)


def constante_de_desintegration(periode: object) -> sp.Expr:
    """`beta = ln 2 / T`, la demi-vie etant donnee."""
    return sp.log(2) / sp.sympify(periode)


def age_d_un_echantillon(fraction_restante: object, periode: object) -> sp.Expr:
    """`t = T log2(1 / fraction)` ; leve si la fraction n'est pas dans `]0, 1]`."""
    fraction = sp.sympify(fraction_restante)
    if fraction.is_number and not 0 < fraction <= 1:
        msg = f"fraction restante hors de ]0, 1] : {fraction}"
        raise ValueError(msg)
    return sp.simplify(sp.sympify(periode) * sp.log(1 / fraction, 2))
