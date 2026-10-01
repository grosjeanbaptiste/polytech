"""La mise a l'echelle d'impedance, les transpositions, et le temps de groupe.

Un prototype passe-bas normalise -- coupure a 1 rad/s, terminaisons de
1 ohm -- devient n'importe quel filtre par deux operations que le fascicule
separe soigneusement.

La **mise a l'echelle d'impedance** multiplie les resistances et les
inductances par `k` et divise les capacites par `k`. La planche le dit en
capitales : « Beware: the gain of the filter has not changed ». C'est
exactement ce qui la rend gratuite -- elle deplace les impedances sans
toucher au transfert.

Les **transpositions de frequence** changent la nature du filtre par une
substitution sur la variable de Laplace. `s -> 1/s` donne le passe-haut, et
c'est une **involution** : l'appliquer deux fois rend le filtre de depart.
`s -> (s^2 + w0^2) / (B s)` donne le passe-bande, et son inverse le
coupe-bande.

Le **temps de groupe** `-d(arg H)/d omega` est ce qui justifie la famille
de Bessel : un filtre dont il varie deforme un signal numerique, en
retardant ses harmoniques inegalement.
"""

from __future__ import annotations

import sympy as sp


def mise_a_l_echelle_d_impedance(
    facteur: float,
    resistance: float,
    inductance: float,
    capacite: float,
) -> tuple[float, float, float]:
    """`(k R, k L, C / k)` : l'echelle bouge, le transfert ne bouge pas."""
    if facteur <= 0:
        msg = f"facteur d'echelle nul ou negatif : {facteur}"
        raise ValueError(msg)
    return facteur * resistance, facteur * inductance, capacite / facteur


def vers_passe_haut(transfert: sp.Expr, variable: sp.Expr) -> sp.Expr:
    """`s -> 1/s`. Involution : deux applications rendent le filtre de depart."""
    return sp.cancel(sp.simplify(transfert.subs(variable, 1 / variable)))


def _substitution_de_bande(
    variable: sp.Expr,
    centre: sp.Expr,
    largeur: sp.Expr,
) -> sp.Expr:
    """`(s^2 + w0^2) / (B s)`, la substitution du passe-bande."""
    if sp.simplify(largeur) == 0:
        msg = "largeur de bande nulle : la transposition diviserait par zero"
        raise ValueError(msg)
    if sp.simplify(centre) == 0:
        msg = "pulsation centrale nulle : la bande n'a pas de centre"
        raise ValueError(msg)
    return (variable**2 + centre**2) / (largeur * variable)


def vers_passe_bande(
    transfert: sp.Expr,
    variable: sp.Expr,
    centre: sp.Expr,
    largeur: sp.Expr,
) -> sp.Expr:
    """Le passe-bande centre sur `centre`, de largeur `largeur`."""
    substitution = _substitution_de_bande(variable, centre, largeur)
    return sp.cancel(sp.simplify(transfert.subs(variable, substitution)))


def vers_coupe_bande(
    transfert: sp.Expr,
    variable: sp.Expr,
    centre: sp.Expr,
    largeur: sp.Expr,
) -> sp.Expr:
    """Le coupe-bande : l'inverse de la substitution du passe-bande."""
    substitution = _substitution_de_bande(variable, centre, largeur)
    return sp.cancel(sp.simplify(transfert.subs(variable, 1 / substitution)))


def temps_de_groupe(
    transfert: sp.Expr,
    variable: sp.Expr,
    pulsation: sp.Expr,
) -> sp.Expr:
    """`-d(arg H(j omega))/d omega`, en secondes.

    Un retard **constant** ne deforme pas : il translate. C'est l'ideal que
    la famille de Bessel approche, et que Butterworth sacrifie a la raideur.
    """
    frequentiel = transfert.subs(variable, sp.I * pulsation)
    argument = sp.arg(sp.simplify(frequentiel))
    return sp.simplify(-sp.diff(argument, pulsation))
