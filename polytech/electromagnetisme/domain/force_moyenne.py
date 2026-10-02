"""Ce que mesure un quotient de differences d'energie.

Les enonces du chapitre IV.2 donnent souvent une **variation** de
capacite ou d'inductance sur un deplacement fini, et non une derivee. Le
quotient `Delta W / Delta x` qu'on en tire n'est pas la force en un
point : c'est la force **moyenne** sur le deplacement, et il l'est
exactement, quelle que soit la loi suivie.

Il ne se confond avec la force instantanee que si l'energie est affine
en la position. Pour un condensateur plan, c'est le cas d'un glissement
lateral -- la capacite est proportionnelle au recouvrement -- et ce
n'est **pas** le cas d'un ecartement, ou elle varie en `1/e`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    force_a_courant_constant,
)


def quotient_des_differences(
    energie: sp.Expr,
    debut: sp.Expr,
    fin: sp.Expr,
) -> sp.Expr:
    """`(W(fin) - W(debut)) / (fin - debut)` : la forme du manuscrit."""
    if sp.simplify(fin - debut) == 0:
        msg = "deplacement nul : le quotient des differences y divise par zero"
        raise ValueError(msg)
    variation = energie.subs(DEPLACEMENT, fin) - energie.subs(DEPLACEMENT, debut)
    return sp.simplify(variation / (fin - debut))


def force_moyenne(energie: sp.Expr, debut: sp.Expr, fin: sp.Expr) -> sp.Expr:
    """La moyenne de `+dW/dx` entre les deux positions.

    Calculee en **integrant** la force instantanee, et non par le
    quotient : c'est ce qui rend l'egalite des deux verifiable au lieu
    d'etre posee.
    """
    if sp.simplify(fin - debut) == 0:
        msg = "deplacement nul : la moyenne n'y est pas definie"
        raise ValueError(msg)
    force = force_a_courant_constant(energie)
    return sp.simplify(sp.integrate(force, (DEPLACEMENT, debut, fin)) / (fin - debut))


def ecart_relatif_sur_l_ecartement(ecartement: sp.Expr, pas: sp.Expr) -> sp.Expr:
    """De combien la force moyenne s'ecarte de la force au depart.

    Pour un condensateur plan dont on fait varier l'**ecartement**, de
    `e` a `e + pas` : la capacite est en `1/e`, la force au depart en
    `1/e^2`, et le rapport rendu est `(F_depart - F_moyenne) / F_depart`.
    Ni la tension ni la constante de la capacite n'y survivent.
    """
    constante = sp.Symbol("k_capacite", positive=True)
    energie = constante / DEPLACEMENT
    depart = force_a_courant_constant(energie).subs(DEPLACEMENT, ecartement)
    moyenne = force_moyenne(energie, ecartement, ecartement + pas)
    return sp.simplify((depart - moyenne) / depart)
