r"""Ce que les corriges du recueil d'algebre ont en commun : l'ecriture.

Le recueil -- `Mathematique/SYLLABUS/Syllabus_Algebre.pdf` -- ecrit l'unite
imaginaire `i`. `formatage.formule` l'ecrit `j` ; les macros du recueil
passent donc par `ecriture`, qui suit le syllabus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.formatage import nombre
from polytech.maths.application.macros_exs_01 import NOMBRES

if TYPE_CHECKING:
    from collections.abc import Iterable

DECIMALES = 3


def ecriture(expression: sp.Basic) -> str:
    """Le LaTeX d'une expression exacte, dans les notations du syllabus."""
    rendu: str = sp.latex(
        expression, ln_notation=True, inv_trig_style="full", mat_delim="("
    )
    return rendu


def liste(expressions: Iterable[sp.Basic]) -> str:
    r"""Les expressions a la suite, separees par `,\;`."""
    return r",\; ".join(ecriture(e) for e in expressions)


def nom(prefixe: str, numero: int, suffixe: str = "") -> str:
    """`AlgUnDixHuitC` pour l'exercice 18 c) : un nom de macro n'a pas de chiffre."""
    if not 1 <= numero <= len(NOMBRES):
        msg = f"numero d'exercice hors de la table : {numero}"
        raise ValueError(msg)
    return f"{prefixe}{NOMBRES[numero - 1]}{suffixe[:1].upper()}{suffixe[1:]}"


def points(complexes: Iterable[sp.Expr]) -> str:
    """`(x,y) (x,y) ...` : les affixes, pour une figure du plan de Gauss."""
    valeurs = [complex(sp.N(z)) for z in complexes]
    return " ".join(
        f"({nombre(v.real, DECIMALES)},{nombre(v.imag, DECIMALES)})" for v in valeurs
    )
