"""Les racines d'un polynome, et ce que ses coefficients en disent sans les calculer.

Le syllabus d'algebre (IV.2, IV.3) :

* un **bicarre** `a x^4 + b x^2 + c` se resout en posant `y = x^2` ;
  chaque `y` donne deux `x`, sauf `y = 0` ;
* un polynome a **coefficients entiers** n'a de racine rationnelle `p/q`
  que si `p` divise `a_0` et `q` divise `a_n` : la liste des candidats est
  finie, et l'evaluation tranche ;
* dans les reels, un polynome se **factorise** en facteurs du premier
  degre -- un par racine reelle -- et du second degre a discriminant
  negatif -- un par paire de racines conjuguees ;
* toute expression **symetrique** des racines se calcule sur les seuls
  coefficients, par `e_k = (-1)^k a_(n-k) / a_n`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from sympy.polys.polyfuncs import symmetrize

from polytech.maths.domain.algebre.complexes_exacts import racines_du_second_degre
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence


def racines_du_bicarre_dans_c(a: object, b: object, c: object) -> tuple[sp.Expr, ...]:
    """Les racines distinctes de `a x^4 + b x^2 + c`, par `y = x^2`."""
    racines: list[sp.Expr] = []
    for y in racines_du_second_degre(a, b, c):
        for x in (sp.sqrt(y), -sp.sqrt(y)):
            if x not in racines:
                racines.append(x)
    return tuple(racines)


def racines_rationnelles(coefficients: Sequence[int]) -> tuple[sp.Rational, ...]:
    """Les `p/q` qui annulent le polynome, `p` divisant `a_0` et `q` divisant `a_n`."""
    utiles = list(coefficients)
    if not any(utiles):
        msg = "polynome nul : tout nombre en est racine"
        raise ValueError(msg)
    trouvees: set[sp.Rational] = set()
    while utiles[-1] == 0:
        utiles.pop()
        trouvees.add(sp.Integer(0))
    polynome = sp.Poly(utiles, VARIABLE)
    for p in sp.divisors(utiles[-1]):
        for q in sp.divisors(utiles[0]):
            for candidat in (sp.Rational(p, q), -sp.Rational(p, q)):
                if polynome.eval(candidat) == 0:
                    trouvees.add(candidat)
    return tuple(sorted(trouvees))


def factoriser_dans_r(
    polynome: sp.Expr, variable: sp.Symbol = VARIABLE
) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le coefficient dominant, et les facteurs unitaires de degre un ou deux."""
    forme = sp.Poly(polynome, variable)
    racines = sp.roots(forme)
    if sum(racines.values()) != forme.degree():
        msg = "racines introuvables en forme close : pas de factorisation rendue"
        raise ValueError(msg)
    facteurs: list[sp.Expr] = []
    for racine, ordre in racines.items():
        imaginaire = sp.im(racine)
        if imaginaire == 0:
            facteurs += [variable - racine] * ordre
        elif imaginaire > 0:
            carre = variable**2 - 2 * sp.re(racine) * variable + sp.Abs(racine) ** 2
            facteurs += [sp.simplify(carre)] * ordre
    return forme.LC(), tuple(facteurs)


def valeur_symetrique_des_racines(
    coefficients: Sequence[sp.Expr],
    expression: sp.Expr,
    racines: Sequence[sp.Symbol],
) -> sp.Expr:
    """La valeur d'une expression symetrique des racines, sans calculer celles-ci."""
    degre = len(coefficients) - 1
    if len(racines) != degre:
        msg = "il faut autant de racines que le degre du polynome"
        raise ValueError(msg)
    valeurs = [
        (-1) ** k * sp.sympify(coefficients[k]) / coefficients[0]
        for k in range(1, degre + 1)
    ]
    morceaux = []
    for partie in sp.fraction(sp.together(expression)):
        elementaire, reste, definitions = symmetrize(partie, *racines, formal=True)
        if reste != 0:
            msg = "expression non symetrique : sa valeur depend de l'ordre des racines"
            raise ValueError(msg)
        symboles = [symbole for symbole, _ in definitions]
        morceaux.append(elementaire.subs(dict(zip(symboles, valeurs, strict=True))))
    return sp.simplify(morceaux[0] / morceaux[1])
