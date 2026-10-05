"""Les graphes de fluence et la regle de Mason.

Syllabus de signaux et systemes, section 4.2. Diviser haut et bas de
`H(p)` par `a_n p^n` ne laisse que des puissances negatives de `p`,
c'est-a-dire des integrateurs (4.22) :

`H(p) = (beta_(n-1) p^-1 + ... + beta_0 p^-n)
        / (1 + alpha_(n-1) p^-1 + ... + alpha_0 p^-n)`,

avec `alpha_i = a_i / a_n` et `beta_i = b_i / a_n`. Le graphe se dessine
alors avec trois operations : additionner, multiplier par un scalaire,
integrer.

La regle de Mason lit la transmittance d'un graphe sans boucle disjointe
(4.25) : la somme des parcours, divisee par un moins la somme des
boucles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from collections.abc import Sequence


def regle_de_mason(parcours: Sequence[sp.Expr], boucles: Sequence[sp.Expr]) -> sp.Expr:
    """`H = somme P_i / (1 - somme B_j)`."""
    if not parcours:
        msg = "aucun parcours de l'entree a la sortie : la transmittance est nulle"
        raise ValueError(msg)
    return sp.cancel(sum(parcours, sp.S.Zero) / (1 - sum(boucles, sp.S.Zero)))


def coefficients_normalises(
    transmittance: sp.Expr,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """`(alpha_(n-1) ... alpha_0)` et `(beta_(n-1) ... beta_0)`."""
    numerateur, denominateur = (
        sp.Poly(e, P) for e in sp.fraction(sp.cancel(transmittance))
    )
    ordre = denominateur.degree()
    if numerateur.degree() >= ordre:
        msg = "degre du numerateur non inferieur a celui du denominateur"
        raise ValueError(msg)
    tete, *suite = denominateur.all_coeffs()
    hauts = [sp.S.Zero] * (ordre - 1 - numerateur.degree()) + numerateur.all_coeffs()
    return (tuple(a / tete for a in suite), tuple(b / tete for b in hauts))


def transmittance_du_graphe(
    *, retours: Sequence[sp.Expr], directs: Sequence[sp.Expr]
) -> sp.Expr:
    """La transmittance du graphe canonique, par la regle de Mason.

    Le `k`-ieme integrateur ferme une boucle de gain `retours[k]` et ouvre
    un parcours de gain `directs[k]`.
    """
    if len(retours) != len(directs):
        msg = "le graphe canonique porte autant de retours que de gains directs"
        raise ValueError(msg)
    integrateurs = [P ** -(rang + 1) for rang in range(len(retours))]
    parcours = [gain * i for gain, i in zip(directs, integrateurs, strict=True)]
    boucles = [gain * i for gain, i in zip(retours, integrateurs, strict=True)]
    return regle_de_mason(parcours, boucles)
