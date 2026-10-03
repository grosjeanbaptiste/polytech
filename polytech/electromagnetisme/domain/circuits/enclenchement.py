"""L'enclenchement d'un circuit du premier ordre sur un regime quelconque.

`x(t) = x_r(t) + (x(0) - x_r(0)) e^{-t/tau}` : la solution particuliere
-- le regime force -- plus la solution homogene, dont la constante se
fixe par la condition initiale. C'est la methode du cours, et elle ne
suppose rien du regime : un palier redonne la reponse du premier ordre,
un cosinus donne l'enclenchement en alternatif.

Le raccord porte la **difference** entre la valeur initiale et celle du
regime a l'instant zero. Oublier la premiere revient a supposer le
circuit au repos ; oublier la seconde, a supposer que le regime part de
zero -- ce qu'un cosinus ne fait que pour une phase particuliere.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS


def enclenchement(
    regime: sp.Expr,
    valeur_initiale: sp.Expr,
    constante_de_temps: sp.Expr,
) -> sp.Expr:
    """Le regime force, plus le raccord a la condition initiale."""
    if sp.simplify(constante_de_temps) == 0:
        msg = "constante de temps nulle : aucun transitoire a raccorder"
        raise ValueError(msg)
    ecart = valeur_initiale - sp.sympify(regime).subs(TEMPS, 0)
    return regime + ecart * sp.exp(-TEMPS / constante_de_temps)
