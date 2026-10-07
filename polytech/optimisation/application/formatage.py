r"""Mise en forme des valeurs pour le document.

L'optimisation lineaire travaille en nombres **exacts** : un tableau
simplexe en flottants derive. Un rationnel est donc publie tel quel --
un entier en chiffres, une fraction en `\frac` --, et un point comme la
liste de ses coordonnees. Tout se compose en mode mathematique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def exact(valeur: object) -> str:
    r"""Le LaTeX d'une valeur exacte : `-19`, `\frac{7}{2}`, `6 M`, `\infty`."""
    expression = sp.nsimplify(valeur, rational=True)
    if expression.is_Rational and expression < 0 and not expression.is_Integer:
        oppose: str = sp.latex(-expression)
        return f"-{oppose}"
    rendu: str = sp.latex(expression)
    return rendu


def arrondi(valeur: object, decimales: int) -> str:
    r"""Une valeur approchee, en chiffres : a composer dans `\num`."""
    if decimales < 0:
        msg = f"nombre de decimales negatif : {decimales}"
        raise ValueError(msg)
    return f"{float(sp.sympify(valeur)):.{decimales}f}"


def point(coordonnees: Sequence[object]) -> str:
    r"""`(a,\ b,\ c)` : un point, coordonnees exactes."""
    if not coordonnees:
        msg = "point sans coordonnee"
        raise ValueError(msg)
    return "(" + r",\ ".join(exact(c) for c in coordonnees) + ")"


def intervalle(bas: object, haut: object) -> str:
    r"""`a \le \cdot \le b`, une borne infinie etant omise."""
    basse, haute = sp.sympify(bas), sp.sympify(haut)
    if basse == -sp.oo and haute == sp.oo:
        msg = "intervalle sans borne : rien a publier"
        raise ValueError(msg)
    if basse == -sp.oo:
        return rf"\le {exact(haute)}"
    if haute == sp.oo:
        return rf"\ge {exact(basse)}"
    return rf"\in [{exact(basse)},\ {exact(haute)}]"
