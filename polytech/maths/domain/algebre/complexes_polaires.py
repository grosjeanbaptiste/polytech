"""Calculer avec les complexes en notation trigonometrique, et lire un cercle.

Le syllabus d'algebre (I.2.2, I.2.3) : `z = |z| (cos theta + i sin theta)`,
et sous cette forme

* le produit multiplie les modules et ajoute les arguments ;
* `z^n = |z|^n (cos n theta + i sin n theta)`, la formule de Moivre ;
* `1/z = (1/|z|) (cos(-theta) + i sin(-theta))` ;
* le quotient divise les modules et retranche les arguments.

Un couple `(module, argument)` porte la notation. L'argument n'est pas
ramene dans un intervalle : le syllabus ecrit `15 pi / 4` tel quel, et il
ne vaut qu'a `2 k pi` pres.

Le lieu d'une condition sur `z` (I.3.1) se reconnait en completant les
carres : `(x - a)^2 + (y - b)^2 = r^2` est le cercle de centre `(a, b)`
et de rayon `r`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import ORDONNEE
from polytech.maths.domain.symboles import VARIABLE

Polaire = tuple[sp.Expr, sp.Expr]
"""Le module et un argument."""
Cercle = tuple[tuple[sp.Expr, sp.Expr], sp.Expr]
"""Le centre `(a, b)` et le rayon."""


def forme_algebrique(module: object, argument: object) -> sp.Expr:
    """`a + bi`, avec `a = |z| cos theta` et `b = |z| sin theta`."""
    rayon, angle = sp.sympify(module), sp.sympify(argument)
    if rayon.is_negative:
        msg = f"module negatif : {rayon} n'est pas une distance a l'origine"
        raise ValueError(msg)
    return sp.simplify(rayon * sp.cos(angle)) + sp.I * sp.simplify(
        rayon * sp.sin(angle)
    )


def produit_polaire(premier: Polaire, second: Polaire) -> Polaire:
    """Les modules se multiplient, les arguments s'ajoutent."""
    return sp.simplify(premier[0] * second[0]), sp.simplify(premier[1] + second[1])


def puissance_polaire(nombre: Polaire, exposant: int) -> Polaire:
    """La formule de Moivre : `(|z|^n, n theta)`."""
    return sp.simplify(nombre[0] ** exposant), sp.simplify(exposant * nombre[1])


def inverse_polaire(nombre: Polaire) -> Polaire:
    """`(1/|z|, -theta)` ; leve pour zero, qui n'a pas d'inverse."""
    if sp.simplify(nombre[0]) == 0:
        msg = "module nul : zero n'a ni inverse ni argument"
        raise ValueError(msg)
    return sp.simplify(1 / nombre[0]), sp.simplify(-nombre[1])


def quotient_polaire(dividende: Polaire, diviseur: Polaire) -> Polaire:
    """Les modules se divisent, les arguments se retranchent."""
    return produit_polaire(dividende, inverse_polaire(diviseur))


def cercle_du_lieu(condition: sp.Expr) -> Cercle:
    """Le cercle `condition = 0`, en `x` et `y` ; leve si ce n'en est pas un."""
    polynome = sp.Poly(sp.expand(condition), VARIABLE, ORDONNEE)
    carre = polynome.coeff_monomial(VARIABLE**2)
    if (
        polynome.total_degree() != 2  # noqa: PLR2004 -- un cercle est du second degre
        or carre != polynome.coeff_monomial(ORDONNEE**2)
        or polynome.coeff_monomial(VARIABLE * ORDONNEE) != 0
    ):
        msg = f"ce n'est pas un cercle : {condition} = 0"
        raise ValueError(msg)
    abscisse = -polynome.coeff_monomial(VARIABLE) / (2 * carre)
    ordonnee = -polynome.coeff_monomial(ORDONNEE) / (2 * carre)
    rayon_carre = abscisse**2 + ordonnee**2 - polynome.coeff_monomial(1) / carre
    if rayon_carre.is_negative:
        msg = f"aucun point : le carre du rayon vaudrait {rayon_carre}"
        raise ValueError(msg)
    return (abscisse, ordonnee), sp.sqrt(rayon_carre)
