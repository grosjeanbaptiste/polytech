"""La primitive d'une fraction rationnelle, racine par racine (notes de 2011, p. 49).

Pour une racine simple `a` du denominateur, `G(x) = f(x) (x - a)` :
- une racine reelle contribue `G(a) ln|x - a|` ;
- une paire conjuguee `a +- ib` contribue, avec `gamma = G(a + ib) = l + i m`,
  `l ln((x - a)^2 + b^2) - 2 m arctg((x - a) / b)` -- la somme de
  `gamma ln(x - alpha)` et de sa conjuguee, a une constante pres.

Une racine multiple demande les derivees de `G` : elle est refusee ici.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def _g_en(fraction: sp.Expr, racine: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`G(racine)`, en verifiant que la racine est simple."""
    denominateur = sp.Poly(sp.denom(sp.together(fraction)), variable)
    multiplicite = 0
    while denominateur.eval(racine).equals(0):
        multiplicite += 1
        denominateur = denominateur.diff(variable)
    if multiplicite == 0:
        msg = f"{racine} n'est pas racine du denominateur de {fraction}"
        raise ValueError(msg)
    if multiplicite > 1:
        msg = f"racine {racine} de multiplicite {multiplicite} : seules les simples"
        raise ValueError(msg)
    return sp.simplify(sp.cancel(fraction * (variable - racine)).subs(variable, racine))


def contribution_d_une_racine_reelle(
    fraction: sp.Expr, racine: sp.Expr, variable: sp.Symbol = VARIABLE
) -> sp.Expr:
    """`G(a) ln|x - a|`."""
    return _g_en(fraction, racine, variable) * sp.log(sp.Abs(variable - racine))


def contribution_de_racines_conjuguees(
    fraction: sp.Expr, racine: sp.Expr, variable: sp.Symbol = VARIABLE
) -> sp.Expr:
    """`l ln((x - a)^2 + b^2) - 2 m arctg((x - a) / b)`, `G(a + ib) = l + i m`."""
    a, b = sp.re(racine), sp.im(racine)
    if b == 0:
        msg = f"{racine} n'a pas de partie imaginaire : c'est une racine reelle"
        raise ValueError(msg)
    gamma = sp.expand_complex(_g_en(fraction, racine, variable))
    ecart = variable - a
    return sp.re(gamma) * sp.log(ecart**2 + b**2) - 2 * sp.im(gamma) * sp.atan(
        ecart / b
    )
