"""L'integrale d'une fonction complexe le long d'une courbe.

`int_gamma f(z) dz = int_a^b f(gamma(t)) gamma'(t) dt` pour une courbe
`gamma : [a, b] -> D` continument derivable par morceaux (definition 15,
relation (27)). La valeur ne depend pas du parametrage, pour peu que
l'orientation soit respectee (remarque 1) ; et si `f = F'`, elle ne depend
que des extremites, `F(gamma(b)) - F(gamma(a))` (relation (28)).

Deux chemins : la forme close, par sympy, qui leve si l'integrale ne se
ferme pas -- aucun resultat partiel ne passe pour une valeur ; et la
quadrature, pour une courbe ou une fonction que sympy ne sait pas integrer.

Source : Mathematiques pour l'ingenieur III, syllabus, section 5.1, page 8.
"""

from __future__ import annotations

import mpmath
import sympy as sp

Z = sp.Symbol("z")
"""La variable de la fonction integree."""


def _integrande(fonction: sp.Expr, courbe: sp.Expr, parametre: sp.Symbol) -> sp.Expr:
    """`f(gamma(t)) gamma'(t)`."""
    return sp.sympify(fonction).subs(Z, courbe) * sp.diff(courbe, parametre)


def integrale_curviligne(
    fonction: sp.Expr, courbe: sp.Expr, parametre: sp.Symbol, debut: object, fin: object
) -> sp.Expr:
    """La forme close de (27) ; leve si sympy ne la trouve pas."""
    valeur = sp.integrate(
        sp.expand_complex(_integrande(fonction, courbe, parametre)),
        (parametre, debut, fin),
    )
    if valeur.has(sp.Integral):
        msg = f"pas de forme close pour l'integrale de {fonction} le long de {courbe}"
        raise ValueError(msg)
    return sp.simplify(valeur)


def integrale_curviligne_numerique(
    fonction: sp.Expr, courbe: sp.Expr, parametre: sp.Symbol, debut: object, fin: object
) -> complex:
    """(27) par quadrature, l'intervalle decoupe : la courbe peut froler un pole."""
    integrande = sp.lambdify(
        parametre, _integrande(fonction, courbe, parametre), "mpmath"
    )
    a, b = float(sp.sympify(debut)), float(sp.sympify(fin))
    return complex(mpmath.quad(integrande, mpmath.linspace(a, b, 65)))
