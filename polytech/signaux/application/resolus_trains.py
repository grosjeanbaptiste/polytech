"""Exercices resolus, 0.7 (b) : le produit scalaire de deux trains de portes.

Le syllabus (1.39) pose le produit scalaire de deux signaux periodiques
nul des que leurs periodes ne sont pas multiples l'une de l'autre. La
moyenne de leur produit, elle, se calcule : sur la periode commune, c'est
la longueur ou les portes se recouvrent.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

PERIODE_DE_X: Final = sp.Integer(2)
PERIODE_DE_Y: Final = sp.Rational(3, 4)
DEMI_LARGEUR: Final = sp.Rational(1, 2)
"""Les portes ont une largeur d'une seconde."""


def sont_multiples(premiere: sp.Expr, seconde: sp.Expr) -> bool:
    """Vrai si l'une des periodes est un multiple entier de l'autre."""
    rapport = sp.Rational(premiere) / sp.Rational(seconde)
    return bool(rapport.is_integer or (1 / rapport).is_integer)


def produit_scalaire_des_trains(
    premiere: sp.Rational = PERIODE_DE_X, seconde: sp.Rational = PERIODE_DE_Y
) -> sp.Expr:
    """La regle (1.39) du syllabus, pour deux trains de ces periodes."""
    if sont_multiples(premiere, seconde):
        msg = "periodes multiples : (1.39) demande alors de calculer l'integrale"
        raise NotImplementedError(msg)
    return sp.Integer(0)


def periode_commune(
    premiere: sp.Rational = PERIODE_DE_X, seconde: sp.Rational = PERIODE_DE_Y
) -> sp.Expr:
    """La plus petite duree qui contient un nombre entier des deux periodes."""
    return sp.Rational(sp.ilcm(premiere.p, seconde.p), sp.igcd(premiere.q, seconde.q))


def _centres(periode: sp.Expr, duree: sp.Expr) -> list[sp.Expr]:
    """Les centres des portes qui peuvent toucher `[0, duree]`."""
    return [periode * rang for rang in range(-1, int(duree / periode) + 2)]


def _recouvrement(premier: sp.Expr, second: sp.Expr, duree: sp.Expr) -> sp.Expr:
    """La longueur commune a deux portes, restreinte a `[0, duree]`."""
    debut = sp.Max(premier - DEMI_LARGEUR, second - DEMI_LARGEUR, 0)
    fin = sp.Min(premier + DEMI_LARGEUR, second + DEMI_LARGEUR, duree)
    return sp.Max(fin - debut, 0)


def moyenne_du_produit_des_trains() -> sp.Expr:
    """`(1/T) integrale de x y` sur la periode commune."""
    duree = periode_commune()
    total = sum(
        _recouvrement(centre_x, centre_y, duree)
        for centre_x in _centres(PERIODE_DE_X, duree)
        for centre_y in _centres(PERIODE_DE_Y, duree)
    )
    return sp.nsimplify(total / duree)


def produit_des_moyennes_des_trains() -> sp.Expr:
    """La moyenne d'un train de portes unite est sa largeur sur sa periode."""
    largeur = 2 * DEMI_LARGEUR
    return (largeur / PERIODE_DE_X) * (largeur / PERIODE_DE_Y)
