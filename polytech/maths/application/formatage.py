r"""Mise en forme des valeurs et des formules pour le document.

Les nombres sont publies **bruts**, a point decimal : c'est la syntaxe
d'entree de siunitx. Les formules sont composees par `sympy` dans le
LaTeX que le document inclut en mode mathematique, avec deux choix qui
suivent l'usage du cours : le logarithme neperien s'ecrit `\ln`, et les
fonctions reciproques `\arcsin`, `\arctan` ; une matrice se met entre
parentheses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    import numpy as np


def nombre(valeur: float, decimales: int) -> str:
    """Formate avec ce nombre de decimales, zeros de fin retires."""
    if decimales < 0:
        msg = f"nombre de decimales negatif : {decimales}"
        raise ValueError(msg)
    rendu = f"{valeur:.{decimales}f}"
    rendu = rendu.rstrip("0").rstrip(".") if "." in rendu else rendu
    return "0" if rendu == "-0" else rendu


def coordonnees(abscisses: np.ndarray, ordonnees: np.ndarray, decimales: int) -> str:
    """`(x,y) (x,y) ...`, la liste que l'option `coordinates` de pgfplots lit."""
    if len(abscisses) != len(ordonnees):
        msg = f"longueurs differentes : {len(abscisses)} et {len(ordonnees)}"
        raise ValueError(msg)
    return " ".join(
        f"({nombre(float(x), decimales)},{nombre(float(y), decimales)})"
        for x, y in zip(abscisses, ordonnees, strict=True)
    )


def _latex(expression: sp.Basic) -> str:
    rendu: str = sp.latex(
        expression,
        ln_notation=True,
        inv_trig_style="full",
        mat_delim="(",
        imaginary_unit="j",
    )
    return rendu


def formule(expression: sp.Basic) -> str:
    r"""Le LaTeX d'une expression exacte, dans les notations du cours.

    Une somme qui porte une integrale non evaluee -- l'etape d'une
    integration par parties -- s'ecrit le terme integre d'abord, puis
    l'integrale, avec le signe resolu : `u v + \int f` plutot que
    `- \int (-f) + u v`.
    """
    integrales = sorted(expression.atoms(sp.Integral), key=sp.default_sort_key)
    if not integrales or not isinstance(expression, sp.Add):
        return _latex(expression)
    reste, morceaux = expression, []
    for integrale in integrales:
        coefficient = expression.coeff(integrale)
        reste -= coefficient * integrale
        integrande = sp.simplify(coefficient * integrale.function)
        signe = "-" if integrande.could_extract_minus_sign() else "+"
        variable = _latex(integrale.variables[0])
        morceaux.append(rf"{signe} \int {_latex(abs_signe(integrande))}\, d{variable}")
    return " ".join([_latex(sp.simplify(reste)), *morceaux])


def abs_signe(expression: sp.Expr) -> sp.Expr:
    """L'expression, son signe moins de tete retire s'il y en a un."""
    return -expression if expression.could_extract_minus_sign() else expression
