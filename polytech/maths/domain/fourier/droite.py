"""La transformee de Fourier sur la droite : porte, convolution, modulation.

Mathematiques appliquees, chapitre 10, dans la convention (10.2) :
`F(w) = int f(x) e^(-j w x) dx`.

* 10.2.1 -- pour une fonction rendue periodique, `c_n = F_T(w_n) / T` avec
  `w_n = 2 pi n / T` ; et (10.7) : `F(0)` est l'integrale de `f` ;
* 10.2.2 -- la **porte** `Pi`, qui vaut 1 sur un segment centre a
  l'origine et 0 ailleurs ;
* 10.2.3 -- la convention en **frequence**, `w = 2 pi nu` ;
* 10.6.3 -- moduler par `cos(w0 x)` ou `sin(w0 x)` translate le spectre
  des deux cotes ;
* 10.6.5 -- la **convolution** `(f * g)(x) = int f(u) g(x - u) du` sur
  toute la droite, d'image `F G` ; et l'image d'un produit,
  `(1 / 2 pi) F * G`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    integrale_sur_la_droite,
)
from polytech.maths.domain.symboles import VARIABLE

FREQUENCE: Final = sp.Symbol("nu", real=True)
"""`nu`, la frequence : `omega = 2 pi nu`."""

_MUETTE: Final = sp.Symbol("u", real=True)


def porte(largeur: object = 1) -> sp.Expr:
    """`Pi` : 1 pour `|x| <= largeur / 2`, 0 ailleurs."""
    demi = sp.sympify(largeur) / 2
    if not demi.is_positive:
        msg = f"largeur strictement positive attendue, {largeur} recue"
        raise ValueError(msg)
    return sp.Piecewise((1, sp.Abs(VARIABLE) <= demi), (0, True))


def valeur_a_l_origine(image: sp.Expr) -> sp.Expr:
    """`F(0)`, prise par continuite : c'est l'integrale de `f` (10.7)."""
    return sp.limit(image, PULSATION, 0)


def coefficient_depuis_la_transformee(
    image: sp.Expr, periode: object, rang: int
) -> sp.Expr:
    """`c_n = F(w_n) / T`, avec `w_n = 2 pi n / T`."""
    duree = sp.sympify(periode)
    return sp.limit(image, PULSATION, 2 * sp.pi * rang / duree) / duree


def convolution_sur_la_droite(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    """`(f * g)(x) = int f(u) g(x - u) du`, sur toute la droite."""
    produit = premiere.subs(VARIABLE, _MUETTE) * seconde.subs(
        VARIABLE, VARIABLE - _MUETTE
    )
    return sp.simplify(integrale_sur_la_droite(produit, _MUETTE))


def image_du_produit(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    """`F[f g] = (1 / 2 pi) (F * G)`, la convolution prise en `omega`."""
    produit = premiere.subs(PULSATION, _MUETTE) * seconde.subs(
        PULSATION, PULSATION - _MUETTE
    )
    return sp.simplify(integrale_sur_la_droite(produit, _MUETTE) / (2 * sp.pi))


def _translatees(image: sp.Expr, pulsation: object) -> tuple[sp.Expr, sp.Expr]:
    ecart = sp.sympify(pulsation)
    return (
        image.subs(PULSATION, PULSATION - ecart),
        image.subs(PULSATION, PULSATION + ecart),
    )


def par_un_cosinus(image: sp.Expr, pulsation: object) -> sp.Expr:
    """`F[f cos(w0 x)] = (F(w - w0) + F(w + w0)) / 2`."""
    droite, gauche = _translatees(image, pulsation)
    return (droite + gauche) / 2


def par_un_sinus(image: sp.Expr, pulsation: object) -> sp.Expr:
    """`F[f sin(w0 x)] = (F(w - w0) - F(w + w0)) / (2j)`."""
    droite, gauche = _translatees(image, pulsation)
    return (droite - gauche) / (2 * sp.I)


def en_frequence(image: sp.Expr) -> sp.Expr:
    """L'image ecrite en `nu` : `F(2 pi nu)`."""
    return image.subs(PULSATION, 2 * sp.pi * FREQUENCE)
