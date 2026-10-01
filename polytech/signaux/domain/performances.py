"""Les performances d'une boucle fermee, telles que le cours les mesure.

Ancrage direct sur `Exs_08_Performances_Reponses` : la feuille publie
`|M(0)| = -3 dB` et `y(infini) = 0,707`, puis `D = 87 %` et
`y_max = 1,32`. Les deux couples se recoupent exactement -- `10^(-3/20)`
vaut `0,7079`, et `0,707 x 1,87` vaut `1,322` -- et ce sont ces deux
relations que ce module porte.

Ce qu'il ne porte **pas**, et c'est delibere : le passage du pic de
resonance au depassement. Le cours le lit sur les abaques de
Black-Nichols, qui sont un abaque et non une formule ; le seul lien en
forme close est celui du second ordre, et il est expose comme tel. Deduire
`D` d'un pic par la formule du second ordre sur un systeme d'ordre trois
serait une approximation non declaree.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.harmonique import depuis_les_decibels

CENT = 100


def gain_statique_depuis_decibels(decibels: sp.Expr) -> sp.Expr:
    """`|M(0)|` en dB donne la valeur finale de la reponse indicielle."""
    return depuis_les_decibels(decibels)


def maximum_de_la_reponse(valeur_finale: sp.Expr, depassement: sp.Expr) -> sp.Expr:
    """`y_max = y(infini) (1 + D)`, avec `D` en fraction et non en pourcent."""
    return sp.simplify(valeur_finale * (1 + depassement))


def depassement_du_second_ordre(amortissement: sp.Expr) -> sp.Expr:
    """`D = exp(-pi zeta / racine(1 - zeta^2))`, formule du second ordre."""
    zeta = sp.sympify(amortissement)
    if not sp.simplify(zeta > 0) or not sp.simplify(zeta < 1):
        msg = (
            f"amortissement hors de ]0, 1[ : {zeta} -- le depassement n'est "
            f"defini que pour un second ordre oscillant"
        )
        raise ValueError(msg)
    return sp.exp(-sp.pi * zeta / sp.sqrt(1 - zeta**2))


def amortissement_du_second_ordre(depassement: sp.Expr) -> sp.Expr:
    """L'inverse de la relation precedente, pour remonter de `D` a `zeta`."""
    logarithme = sp.log(sp.sympify(depassement))
    return sp.simplify(-logarithme / sp.sqrt(sp.pi**2 + logarithme**2))


def en_pourcent(fraction: sp.Expr) -> sp.Expr:
    """Le depassement tel que la feuille de reponses l'ecrit."""
    return sp.simplify(CENT * fraction)
