"""La meilleure approximation d'une fonction de `L^2` : la serie de Fourier generalisee.

Pour une famille **orthonormee** `phi_1, ..., phi_n`, la combinaison
`somme a_k phi_k` la plus proche de `f` est sa projection orthogonale :
`a_k = c_k = <f, phi_k>`, les **coefficients de Fourier**. En effet

    d^2(f, somme a_k phi_k) = ||f||^2 + somme |a_k - c_k|^2 - somme |c_k|^2,

minimale quand chaque `a_k` vaut `c_k`. Il reste alors
`somme |c_k|^2 <= ||f||^2` (inegalite de Bessel) ; l'egalite demande une
famille complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.fourier.espace_l2 import EspaceL2


def est_orthonormee(famille: Sequence[sp.Expr], espace: EspaceL2) -> bool:
    """Vrai si `<phi_i, phi_k>` vaut 1 pour `i = k` et 0 sinon."""
    return all(
        sp.simplify(espace.produit_scalaire(f, g) - int(i == k)) == 0
        for i, f in enumerate(famille)
        for k, g in enumerate(famille)
    )


def orthonormaliser(
    famille: Sequence[sp.Expr], espace: EspaceL2
) -> tuple[sp.Expr, ...]:
    """Le procede de Gram-Schmidt : retirer les projections, puis normer."""
    base: list[sp.Expr] = []
    for fonction in famille:
        projections = (espace.produit_scalaire(fonction, e) * e for e in base)
        reste = sp.simplify(fonction - sum(projections, sp.Integer(0)))
        base.append(sp.simplify(reste / espace.norme(reste)))
    return tuple(base)


def coefficients_de_fourier(
    fonction: sp.Expr, famille: Sequence[sp.Expr], espace: EspaceL2
) -> tuple[sp.Expr, ...]:
    """`c_k = <f, phi_k>`."""
    return tuple(espace.produit_scalaire(fonction, phi) for phi in famille)


def meilleure_approximation(
    fonction: sp.Expr, famille: Sequence[sp.Expr], espace: EspaceL2
) -> sp.Expr:
    """`somme c_k phi_k`, la projection de `f` sur la famille."""
    coefficients = coefficients_de_fourier(fonction, famille, espace)
    termes = (c * phi for c, phi in zip(coefficients, famille, strict=True))
    return sp.simplify(sum(termes, sp.Integer(0)))


def carre_de_la_distance(
    fonction: sp.Expr,
    alphas: Sequence[object],
    famille: Sequence[sp.Expr],
    espace: EspaceL2,
) -> sp.Expr:
    """`||f||^2 + somme |a_k - c_k|^2 - somme |c_k|^2`, pour une famille orthonormee."""
    coefficients = coefficients_de_fourier(fonction, famille, espace)
    ecarts = sum(
        sp.Abs(sp.sympify(a) - c) ** 2
        for a, c in zip(alphas, coefficients, strict=True)
    )
    energie = sum(sp.Abs(c) ** 2 for c in coefficients)
    return sp.simplify(espace.norme(fonction) ** 2 + ecarts - energie)
