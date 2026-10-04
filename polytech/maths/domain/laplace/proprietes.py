"""Les proprietes de la transformee de Laplace (planches 12 a 14, 35 a 42).

Chacune traduit une operation sur `f(t)` en une operation sur `F(p)` :

* produit par `e^(alpha t)` : `F(p - alpha)` ;
* derivee d'ordre `n` : `p^n F - p^(n-1) f(0+) - ... - f^(n-1)(0+)` ;
* integrale `int_0^t f` : `F / p`, et `F / p^n` a l'ordre `n` ;
* retard `f(t - h)`, `h > 0` : `e^(-ph) F` ;
* fonction periodique de periode `T`, de motif `g` : `G / (1 - e^(-pT))` ;
* produit de convolution : `F G`.

Le sinus et le cosinus se transforment par la formule d'Euler, a partir
de la seule transformee de l'exponentielle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import IMAGE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def du_produit_par_une_exponentielle(image: sp.Expr, alpha: object) -> sp.Expr:
    """`L[e^(alpha t) f] = F(p - alpha)`."""
    return image.subs(IMAGE, IMAGE - sp.sympify(alpha))


def de_la_derivee_n(image: sp.Expr, initiales: Sequence[object]) -> sp.Expr:
    """`L(D^n f)`, `n` etant le nombre de valeurs `f(0+), f'(0+), ...` donnees."""
    ordre = len(initiales)
    retranche = sum(
        (IMAGE ** (ordre - 1 - k) * sp.sympify(v) for k, v in enumerate(initiales)),
        sp.Integer(0),
    )
    return IMAGE**ordre * image - retranche


def de_l_integrale(image: sp.Expr, ordre: int = 1) -> sp.Expr:
    """`L(I^n f) = F / p^n`."""
    return image / IMAGE**ordre


def du_retard(image: sp.Expr, retard: object) -> sp.Expr:
    """`L(tau_h f) = e^(-ph) F`, pour un retard `h > 0`."""
    duree = sp.sympify(retard)
    if duree.is_negative:
        msg = f"retard positif attendu, {duree} recu : f(t - h) ne serait plus causale"
        raise ValueError(msg)
    return sp.exp(-IMAGE * duree) * image


def d_une_fonction_periodique(image_du_motif: sp.Expr, periode: object) -> sp.Expr:
    """`F = G / (1 - e^(-pT))`."""
    return image_du_motif / (1 - sp.exp(-IMAGE * sp.sympify(periode)))


def de_la_convolution(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    """`L(f * g) = L(f) L(g)`."""
    return premiere * seconde


def par_la_formule_d_euler(
    fonction: Callable[[sp.Expr], sp.Expr],
    pulsation: object,
) -> sp.Expr:
    """`L(sin wt)` ou `L(cos wt)`, a partir de `L(e^(at)) = 1/(p - a)`."""
    w = sp.sympify(pulsation)
    plus, moins = 1 / (IMAGE - sp.I * w), 1 / (IMAGE + sp.I * w)
    if fonction is sp.sin:
        return sp.simplify((plus - moins) / (2 * sp.I))
    if fonction is sp.cos:
        return sp.simplify((plus + moins) / 2)
    msg = f"formule d'Euler : sinus ou cosinus attendu, {fonction} recu"
    raise ValueError(msg)
