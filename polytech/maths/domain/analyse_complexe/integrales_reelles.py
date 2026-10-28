"""Une integrale reelle par les residus : `int de 0 a 2 pi de dt / (a + cos t)`.

Avec `z = e^(it)`, `cos t = (z + 1/z)/2` et `dt = dz / (iz)` : l'integrale
est `oint f dz` sur le cercle unite, `f = -i / (z^2/2 + a z + 1/2)`. Pour
`a > 1` un seul des deux poles est dans le disque.
"""

from __future__ import annotations

import mpmath
import sympy as sp

from polytech.maths.domain.analyse_complexe.residus import integrale_par_les_residus

_Z = sp.Symbol("z")


def _exiger_a_superieur_a_un(a: object) -> None:
    if not sp.sympify(a) > 1:
        msg = f"l'integrale demande a > 1 : {a}"
        raise ValueError(msg)


def poles_de_1_sur_a_plus_cosinus(a: object) -> tuple[sp.Expr, sp.Expr]:
    """`(alpha, beta)` : le pole du disque unite, puis celui de l'exterieur."""
    _exiger_a_superieur_a_un(a)
    racine = sp.sqrt(sp.sympify(a) ** 2 - 1)
    return -sp.sympify(a) + racine, -sp.sympify(a) - racine


def integrale_de_1_sur_a_plus_cosinus(a: object) -> sp.Expr:
    """La valeur par le theoreme des residus."""
    alpha, _ = poles_de_1_sur_a_plus_cosinus(a)
    fonction = -sp.I / (_Z**2 / 2 + sp.sympify(a) * _Z + sp.Rational(1, 2))
    return integrale_par_les_residus(fonction, _Z, (alpha,))


def quadrature_de_1_sur_a_plus_cosinus(a: object) -> float:
    """Le second chemin : la quadrature de l'integrale reelle."""
    _exiger_a_superieur_a_un(a)
    valeur = float(sp.sympify(a))
    return float(
        mpmath.quad(
            lambda t: 1 / (valeur + mpmath.cos(t)), [0, mpmath.pi, 2 * mpmath.pi]
        )
    )
