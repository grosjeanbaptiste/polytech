"""Les proprietes de la transformee de Fourier.

Pour `f` d'image `F(w)` :

* echelle -- `f(a x)` a pour image `F(w / a) / |a|` : etroite ici, large la ;
* glissement -- `f(x - x0)` a pour image `e^(-j w x0) F(w)` ;
* modulation -- `f(x) e^(j w0 x)` a pour image `F(w - w0)` ;
* derivation -- `f^(n)` a pour image `(j w)^n F(w)` ;
* integration -- `int_(-oo)^x f` a pour image `F / (j w)`, si `F(0) = 0` ;
* energie -- `int |f|^2 dx = (1 / 2 pi) int |F|^2 dw` (Parseval-Plancherel).

La derivation fait d'une equation differentielle a coefficients constants
`K(D) f = h` une equation algebrique : `K(j w) F(w) = H(w)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    integrale_sur_la_droite,
)
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from polytech.maths.domain.differentielles.operateurs import OperateurDifferentiel


def du_changement_d_echelle(image: sp.Expr, facteur: object) -> sp.Expr:
    """`F(w / a) / |a|`."""
    a = sp.sympify(facteur)
    return image.subs(PULSATION, PULSATION / a) / sp.Abs(a)


def du_glissement(image: sp.Expr, retard: object) -> sp.Expr:
    """`e^(-j w x0) F(w)`."""
    return sp.exp(-sp.I * PULSATION * sp.sympify(retard)) * image


def de_la_modulation(image: sp.Expr, pulsation: object) -> sp.Expr:
    """`F(w - w0)`."""
    return image.subs(PULSATION, PULSATION - sp.sympify(pulsation))


def de_la_derivee(image: sp.Expr, ordre: int = 1) -> sp.Expr:
    """`(j w)^n F(w)`."""
    return (sp.I * PULSATION) ** ordre * image


def de_la_primitive(image: sp.Expr) -> sp.Expr:
    """`F / (j w)` ; leve si `F(0)` n'est pas nul."""
    en_zero = sp.limit(image, PULSATION, 0)
    if en_zero != 0:
        msg = f"valeur en zero non nulle ({en_zero}) : la primitive n'est pas sommable"
        raise ValueError(msg)
    return image / (sp.I * PULSATION)


def energie_temporelle(fonction: sp.Expr) -> sp.Expr:
    """`int |f|^2 dx`."""
    carre = sp.sympify(fonction) * sp.conjugate(fonction)
    return sp.simplify(integrale_sur_la_droite(carre, VARIABLE))


def energie_frequentielle(image: sp.Expr) -> sp.Expr:
    """`(1 / 2 pi) int |F|^2 dw`."""
    carre = sp.simplify(sp.expand_complex(image * sp.conjugate(image)))
    return sp.simplify(integrale_sur_la_droite(carre, PULSATION) / (2 * sp.pi))


def image_de_la_solution(operateur: OperateurDifferentiel, force: sp.Expr) -> sp.Expr:
    """`F = H / K(j w)` ; leve si `K(j w)` s'annule pour une pulsation reelle.

    `w` etant declare reel, la resolution ne rend que des racines reelles.
    """
    caracteristique = operateur.en(sp.I * PULSATION)
    resonances = sp.solve(caracteristique, PULSATION)
    if resonances:
        msg = f"resonance : le polynome caracteristique s'annule en w = {resonances}"
        raise ValueError(msg)
    return force / caracteristique
