"""La fonction de Green d'une equation `K(D) y = x(t)` : sa reponse impulsionnelle.

Au repos (`y(0) = ... = y^(n-1)(0) = 0`), l'image de l'equation est
`K(p) Y = X` : `Y = X / K(p)`, et `1 / K(p)` est la **fonction de
transfert**. Son original `g(t)` est la fonction de Green, et

    y(t) = (g * x)(t).

`g` est la reponse a l'impulsion de Dirac, puisque `g = g * delta`. On
l'obtient aussi sans mesure de Dirac, de deux manieres :

* comme limite, quand `n` croit, des reponses aux creneaux de hauteur `n`
  et de duree `1/n` -- de plus en plus intenses, de plus en plus brefs,
  toujours d'aire 1 ;
* comme solution de `K(D) y = 0` avec `y^(n-1)(0) = 1/a_n` et toutes les
  autres conditions initiales nulles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D
from polytech.maths.domain.laplace.convolution import produit_de_convolution
from polytech.maths.domain.laplace.heaviside import par_fractions_simples
from polytech.maths.domain.symboles import IMAGE, TEMPS

if TYPE_CHECKING:
    from polytech.maths.domain.differentielles.operateurs import OperateurDifferentiel


def fonction_de_green(operateur: OperateurDifferentiel) -> sp.Expr:
    """`g(t)`, l'original de `1 / K(p)`."""
    return par_fractions_simples(1 / operateur.polynome.subs(D, IMAGE))


def reponse(operateur: OperateurDifferentiel, sollicitation: sp.Expr) -> sp.Expr:
    """`y = g * x`, la reponse du systeme au repos."""
    return produit_de_convolution(fonction_de_green(operateur), sollicitation)


def reponse_au_creneau(
    operateur: OperateurDifferentiel, hauteur: object
) -> tuple[sp.Expr, sp.Expr]:
    """La reponse au creneau de hauteur `n` et de duree `1/n` : pendant, puis apres."""
    n = sp.sympify(hauteur)
    muette = sp.Symbol("s", real=True, nonnegative=True)
    decalee = n * fonction_de_green(operateur).subs(TEMPS, TEMPS - muette)
    pendant = sp.integrate(decalee, (muette, 0, TEMPS))
    apres = sp.integrate(decalee, (muette, 0, 1 / n))
    return sp.simplify(pendant), sp.simplify(apres)


def conditions_initiales_equivalentes(
    operateur: OperateurDifferentiel,
) -> tuple[sp.Expr, ...]:
    """`(0, ..., 0, 1/a_n)` : celles qui font de `g` une reponse libre."""
    dominant = sp.Poly(operateur.polynome, D).LC()
    return (*(sp.Integer(0),) * (operateur.ordre - 1), 1 / dominant)
