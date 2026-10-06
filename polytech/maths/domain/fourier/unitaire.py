"""La transformee de Fourier dans sa convention unitaire.

Trois cours du depot ecrivent la transformee de Fourier, et chacun place le
facteur `2 pi` ailleurs. Mathematiques III le partage entre la transformee
et son inverse :

    F(f)(s) = (1 / racine(2 pi)) int f(x) e^(-i s x) dx,
    F^-1(g)(x) = (1 / racine(2 pi)) int g(s) e^(+i s x) ds.

Ce partage rend la transformee **unitaire** sur `L2` : `||f|| = ||F(f)||`,
sans facteur. Il en coute un `racine(2 pi)` a la convolution :
`F(f * g) = racine(2 pi) F(f) F(g)`.

Le cours du soir (`transformee.py`) laisse tout le `1 / 2 pi` a l'inverse :
`F_unitaire = F / racine(2 pi)`. Ce module s'appuie sur lui, et
`tests/test_liens_des_conventions_de_fourier.py` verifie le passage.

Pour une fonction reelle paire, `F(f) = racine(2 / pi) int_0^oo f cos(s x) dx` ;
impaire, `F(f) = -i racine(2 / pi) int_0^oo f sin(s x) dx`.

Source : Mathematiques pour l'ingenieur III, section 11, partie 1,
planches 8 a 18.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.fourier.parite import est_impaire, est_paire
from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    integrale_sur_la_droite,
    transformee_de_fourier,
)
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Callable

RACINE_DE_DEUX_PI: Final = sp.sqrt(2 * sp.pi)
"""Le facteur que la convention unitaire partage en deux."""


def transformee_unitaire(fonction: sp.Expr) -> sp.Expr:
    """`(1 / racine(2 pi)) int f(x) e^(-i s x) dx`."""
    return sp.simplify(transformee_de_fourier(fonction) / RACINE_DE_DEUX_PI)


def _demi_integrale(
    fonction: sp.Expr,
    noyau: Callable[[sp.Expr], sp.Expr],
    variable: sp.Symbol,
    duale: sp.Symbol,
) -> sp.Expr:
    """`racine(2 / pi) int_0^oo f(v) noyau(v d) dv`, ecrite pour `|d|`."""
    positive = sp.Symbol("sigma", positive=True)
    integrande = fonction * noyau(variable * positive)
    integrale = sp.integrate(integrande, (variable, 0, sp.oo), conds="none")
    image = sp.simplify(sp.sqrt(2 / sp.pi) * integrale)
    return image.subs(positive, sp.Abs(duale))


def inverse_unitaire(image: sp.Expr) -> sp.Expr:
    """`(1 / racine(2 pi)) int g(s) e^(+i s x) ds`.

    L'image est partagee en sa partie paire et sa partie impaire, et chacune
    s'inverse par sa demi-integrale, en cosinus ou en sinus -- formules (2)
    et (4) : c'est la meme integrale, pliee sur la demi-droite.
    """
    directe = sp.sympify(image)
    miroir = directe.subs(PULSATION, -PULSATION)
    paire, impaire = (
        sp.simplify((directe + miroir) / 2),
        sp.simplify((directe - miroir) / 2),
    )
    en_cos = _demi_integrale(paire, sp.cos, PULSATION, VARIABLE)
    en_sin = _demi_integrale(impaire, sp.sin, PULSATION, VARIABLE)
    return sp.simplify(en_cos + sp.I * sp.sign(VARIABLE) * en_sin)


def en_cosinus(fonction: sp.Expr) -> sp.Expr:
    """La transformee d'une fonction paire, par son seul cosinus ; paire en `s`."""
    if not est_paire(fonction):
        msg = f"fonction non paire : {fonction} n'a pas de transformee en cosinus"
        raise ValueError(msg)
    return _demi_integrale(fonction, sp.cos, VARIABLE, PULSATION)


def en_sinus(fonction: sp.Expr) -> sp.Expr:
    """La transformee d'une fonction impaire, par son seul sinus ; impaire en `s`."""
    if not est_impaire(fonction):
        msg = f"fonction non impaire : {fonction} n'a pas de transformee en sinus"
        raise ValueError(msg)
    demi = _demi_integrale(fonction, sp.sin, VARIABLE, PULSATION)
    return -sp.I * sp.sign(PULSATION) * demi


def de_la_convolution(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    """`F(f * g) = racine(2 pi) F(f) F(g)`."""
    return RACINE_DE_DEUX_PI * premiere * seconde


def energie_spectrale(image: sp.Expr) -> sp.Expr:
    """`||F(f)||^2 = int |F(f)|^2 ds`, egale a `||f||^2` sans facteur."""
    carre = sp.simplify(sp.expand_complex(image * sp.conjugate(image)))
    return sp.simplify(integrale_sur_la_droite(carre, PULSATION))
