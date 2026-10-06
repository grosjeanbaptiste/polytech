"""Mathematique, exercices 1, section 1.1 : les primitives 1 a 7.

Chaque fonction rend la resolution d'un exercice par le geste que la
feuille indique : changement de variable, linearisation, integration par
parties. Les gestes sont ceux de `polytech.maths.domain.integration`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.resolution_exs_01 import A, Resolution, par_equation
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.substitution import NOUVELLE as U
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.integration.trigonometrie import (
    T_DEMI,
    lineariser,
    par_la_tangente_de_l_arc_moitie,
)
from polytech.maths.domain.symboles import VARIABLE as X

_Y = sp.Symbol("y", real=True)
ANGLE = sp.Symbol("u", real=True)
"""L'angle des changements `x = a sin u` et `x = sin u`."""


def substitution(integrande: sp.Expr, reecrit: sp.Expr, u: sp.Expr) -> Resolution:
    """`reecrit` est l'integrande, ecrit pour que `u` et `du` s'y lisent."""
    geste = Substitution(reecrit, u)
    etapes = (geste.integrale_transformee(),)
    return Resolution(integrande, "changement de variable", etapes, geste.resultat())


def parties(u: sp.Expr, dv: sp.Expr, gestes: int = 1) -> Resolution:
    """`gestes` integrations par parties : l'integrale restante est reprise."""
    etapes, courant, facteur = [], u, dv
    for _ in range(gestes):
        geste = IntegrationParParties(courant, facteur)
        etapes.append(geste.etape())
        courant, facteur = geste.derivee_de_u(), geste.v()
    primitive = IntegrationParParties(u, dv).resultat()
    return Resolution(u * dv, "integration par parties", tuple(etapes), primitive)


def linearisation(integrande: sp.Expr, modele: sp.Expr, angle: sp.Expr) -> Resolution:
    """Lineariser `modele(y)`, y remplacer `y` par `angle`, integrer terme a terme.

    Avec `angle = i x`, les sinus et cosinus deviennent hyperboliques :
    c'est ainsi que `sinh^4` se linearise comme `sin^4`.
    """
    lineaire = sp.expand(lineariser(modele).subs(_Y, angle))
    return Resolution(
        integrande, "linearisation", (lineaire,), sp.integrate(lineaire, X)
    )


def exercice_1() -> Resolution:
    """`e^(sin^2 x) sin 2x` : `u = sin^2 x`, `du = sin 2x dx`."""
    integrande = sp.exp(sp.sin(X) ** 2) * sp.sin(2 * X)
    reecrit = sp.exp(sp.sin(X) ** 2) * 2 * sp.sin(X) * sp.cos(X)
    return substitution(integrande, reecrit, sp.sin(X) ** 2)


def en_u(integrande: sp.Expr, transformee: sp.Expr, u: sp.Expr) -> Resolution:
    """Un changement `u = g(x)` dont l'integrande en `u` est donne tout ecrit.

    Pour une puissance impaire, `cos^5 t = (1 - sin^2 t)^2 cos t` : la
    reecriture est le geste, et une simplification automatique la defait.
    """
    primitive = sp.integrate(transformee, U).subs(U, u)
    return Resolution(integrande, "changement de variable", (transformee,), primitive)


def exercice_2() -> tuple[Resolution, Resolution, Resolution, Resolution]:
    """`cos^5`, `cosh^3` par `u = sin`, `u = sinh` ; `cos^4`, `sinh^4` linearises."""
    return (
        en_u(sp.cos(X) ** 5, (1 - U**2) ** 2, sp.sin(X)),
        en_u(sp.cosh(X) ** 3, 1 + U**2, sp.sinh(X)),
        linearisation(sp.cos(X) ** 4, sp.cos(_Y) ** 4, X),
        linearisation(sp.sinh(X) ** 4, sp.sin(_Y) ** 4, sp.I * X),
    )


def formes_linearisees() -> tuple[sp.Expr, sp.Expr]:
    """Les secondes reponses de l'exercice 2 : `cos^5` et `cosh^3` linearises."""
    cosinus = linearisation(sp.cos(X) ** 5, sp.cos(_Y) ** 5, X)
    hyperbolique = linearisation(sp.cosh(X) ** 3, sp.cos(_Y) ** 3, sp.I * X)
    return cosinus.primitive, hyperbolique.primitive


def exercice_3() -> Resolution:
    """`x^3 e^x` : trois integrations par parties, `u` etant le polynome."""
    return parties(X**3, sp.exp(X), gestes=3)


def exercice_4() -> tuple[Resolution, Resolution, Resolution]:
    """`ln x`, `x ln x`, `arcsin x` : `u` est la fonction qu'on ne sait pas integrer."""
    return (
        parties(sp.log(X), sp.Integer(1)),
        parties(sp.log(X), X),
        parties(sp.asin(X), sp.Integer(1)),
    )


def exercice_5() -> Resolution:
    """`cosh t sin t` : deux parties ramenent l'integrale, changee de signe."""
    return par_equation((sp.sin(X), sp.cosh(X)), (sp.cos(X), sp.sinh(X)))


def exercice_6() -> Resolution:
    """`1 / sin x` : `t = tan(x/2)` rend l'integrande rationnel, ici `1 / t`."""
    integrande = 1 / sp.sin(X)
    en_t = par_la_tangente_de_l_arc_moitie(integrande)
    primitive = sp.log(sp.Abs(T_DEMI)).subs(T_DEMI, sp.tan(X / 2))
    return Resolution(integrande, "tangente de l'arc moitie", (en_t,), primitive)


def exercice_7() -> Resolution:
    """`racine(a^2 - x^2)` : `x = a sin u`, ou `cos u >= 0` sur `[-pi/2, pi/2]`."""
    en_u = A**2 * sp.cos(ANGLE) ** 2
    primitive_en_u = A**2 / 2 * (ANGLE + sp.sin(ANGLE) * sp.cos(ANGLE))
    retour = {
        sp.sin(ANGLE): X / A,
        sp.cos(ANGLE): sp.sqrt(A**2 - X**2) / A,
        ANGLE: sp.asin(X / A),
    }
    primitive = primitive_en_u.subs(retour, simultaneous=True)
    etapes = (en_u, primitive_en_u)
    return Resolution(sp.sqrt(A**2 - X**2), "changement de variable", etapes, primitive)
