"""Les autres ecritures d'une serie de Fourier, et les sommes qu'elle donne.

Mathematiques appliquees, 8.6. `series.py` ecrit la serie en cosinus et
sinus ; le cours en donne deux autres, egales a la premiere :

    somme, pour n de -N a N, des c_n e^(j n w x)        (forme complexe),
    c_0 + somme A_n cos(n w x + phi_n)                  (amplitude et phase).

Et Parseval sous sa forme generale, pour `f` a valeurs complexes :
`somme |c_n|^2 = ||f||^2`.

Une serie de Fourier **somme des series numeriques** (remarque 8.4). En
un point, `f(x_0) - c_0 = somme (a_n cos + b_n sin)` ; par Parseval,
`||f||^2 - c_0^2 = somme A_n^2 / 2`. Chaque identite est rendue par son
terme general et sa somme : `t^2` sur `[0, 1[` donne ainsi `pi^2 / 6`,
puis `pi^4 / 90`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from polytech.maths.domain.fourier.series import SerieDeFourier


def forme_complexe(serie: SerieDeFourier, ordre: int) -> sp.Expr:
    """`somme c_n e^(j n w x)`, pour `n` de `-ordre` a `ordre`."""
    angle = sp.I * serie.pulsation * VARIABLE
    termes = (
        serie.coefficient(n) * sp.exp(n * angle) for n in range(-ordre, ordre + 1)
    )
    return sum(termes, sp.Integer(0))


def forme_en_amplitude_et_phase(serie: SerieDeFourier, ordre: int) -> sp.Expr:
    """`c_0 + somme A_n cos(n w x + phi_n)`, jusqu'a `n = ordre`."""
    angle = serie.pulsation * VARIABLE
    harmoniques = (
        serie.amplitude(n) * sp.cos(n * angle + serie.phase(n))
        for n in range(1, ordre + 1)
    )
    return serie.coefficient(0) + sum(harmoniques, sp.Integer(0))


def parseval_complexe(serie: SerieDeFourier, ordre: int) -> sp.Expr:
    """`somme |c_n|^2`, pour `n` de `-ordre` a `ordre`."""
    carres = (sp.Abs(serie.coefficient(n)) ** 2 for n in range(-ordre, ordre + 1))
    return sp.simplify(sum(carres, sp.Integer(0)))


def identite_en_un_point(
    serie: SerieDeFourier, point: object, rang: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """`(terme, somme)` : `somme, n >= 1, du terme = f(point) - c_0`."""
    angle = rang * serie.pulsation * sp.sympify(point)
    terme = serie.a(rang) * sp.cos(angle) + serie.b(rang) * sp.sin(angle)
    return sp.simplify(terme), serie.valeur_en(point) - serie.coefficient(0)


def identite_de_parseval(
    serie: SerieDeFourier, rang: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """`(terme, somme)` : `somme, n >= 1, de A_n^2 / 2 = ||f||^2 - c_0^2`."""
    terme = (serie.a(rang) ** 2 + serie.b(rang) ** 2) / 2
    reste = serie.norme_au_carre() - serie.coefficient(0) ** 2
    return sp.simplify(terme), sp.simplify(reste)
