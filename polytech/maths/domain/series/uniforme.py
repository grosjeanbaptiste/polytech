"""La convergence uniforme d'une serie de fonctions.

La serie converge **simplement** si, en chaque `x`, les sommes partielles
`S_n(x)` tendent vers `S(x)` ; **uniformement** si elles le font toutes a
des vitesses comparables, c'est-a-dire si

    d(S_n, S) = sup_x |S_n(x) - S(x)|

tend vers 0. Une condition suffisante commode : si `|f_n(x)| <= u_n` pour
tout `x` et si la serie numerique des `u_n` converge, la serie des `f_n`
converge uniformement.

Une serie uniformement convergente de fonctions continues a une somme
continue et s'integre terme a terme sur un intervalle borne.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.util import function_range


def distance_uniforme(
    partielle: sp.Expr, limite: object, variable: sp.Symbol, intervalle: sp.Set
) -> sp.Expr:
    """`sup |S_n - S|` sur l'intervalle, morceau par morceau s'il y en a."""
    ecart = sp.piecewise_fold(partielle - sp.sympify(limite))
    if not isinstance(ecart, sp.Piecewise):
        return _plus_grand_ecart(ecart, variable, intervalle)
    morceaux = ecart.as_expr_set_pairs(intervalle)
    return sp.Max(*(_plus_grand_ecart(e, variable, ens) for e, ens in morceaux))


def _plus_grand_ecart(ecart: sp.Expr, variable: sp.Symbol, ensemble: sp.Set) -> sp.Expr:
    """`sup |ecart|` sur un ensemble ou l'expression n'a qu'une forme."""
    valeurs = function_range(ecart, variable, ensemble)
    return sp.Max(sp.Abs(valeurs.inf), sp.Abs(valeurs.sup))


def est_majoree_par(
    terme: sp.Expr, majorant: object, variable: sp.Symbol, intervalle: sp.Set
) -> bool:
    """Vrai si `|f_n(x)| <= u_n` sur tout l'intervalle."""
    return bool(distance_uniforme(terme, 0, variable, intervalle) <= majorant)


def serie_majorante_convergente(majorant: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si la serie numerique `somme u_n`, depuis `n = 1`, converge."""
    return bool(sp.Sum(majorant, (indice, 1, sp.oo)).is_convergent())
