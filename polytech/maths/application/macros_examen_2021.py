r"""Macros du corrige de l'examen de janvier 2021.

Le prefixe est `ExamVingtEtUn`, et un nom de macro n'a pas de chiffre :
`\ExamVingtEtUnSolutionSept` est la solution de l'equation (7).
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import examen_2021_analyse as an
from polytech.maths.application import examen_2021_systemes as sy
from polytech.maths.application.courbes_examens import solution_8_de_2021
from polytech.maths.application.formatage import formule
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE

_TEMPS = sp.Symbol("t", real=True)
"""L'enonce ecrit `y(t)` ; le domaine resout en sa variable, `x`."""
_NOMS = {7: "Sept", 8: "Huit", 9: "Neuf"}


def _triplet(solution: tuple[sp.Expr, ...]) -> str:
    return formule(sp.Matrix([solution]))


def _analyse() -> dict[str, str]:
    dix, onze, douze = an.i_deux()
    macros = {
        "ExamVingtEtUnEtapeI": formule(an.etape_i()),
        "ExamVingtEtUnPrimitiveI": formule(an.primitive_i()),
        "ExamVingtEtUnTermesJ": formule(sp.Add(*an.termes_j(), evaluate=False)),
        "ExamVingtEtUnPrimitiveJ": formule(an.primitive_j()),
        "ExamVingtEtUnPrimitiveK": formule(an.primitive_k()),
        "ExamVingtEtUnIDeuxDix": formule(dix),
        "ExamVingtEtUnIDeuxOnze": formule(onze),
        "ExamVingtEtUnIDeuxDouze": formule(douze),
    }
    for numero, nom in _NOMS.items():
        generale = an.generale_de(numero).subs(VARIABLE, _TEMPS)
        solution = an.solution_de(numero).subs(VARIABLE, _TEMPS)
        macros[f"ExamVingtEtUnGenerale{nom}"] = formule(generale)
        macros[f"ExamVingtEtUnSolution{nom}"] = formule(solution)
    macros["ExamVingtEtUnCourbeHuit"] = solution_8_de_2021()
    return macros


def _systemes() -> dict[str, str]:
    return {
        "ExamVingtEtUnQuatreGenerique": _triplet(sy.solution_4_generique()),
        "ExamVingtEtUnQuatreEnUn": _triplet(sy.solution_4_en(sp.Integer(1))),
        "ExamVingtEtUnCinqDeterminant": formule(sy.determinant_5()),
        "ExamVingtEtUnCinqEnMoinsUn": _triplet(sy.solution_5_en(sp.Integer(-1))),
        "ExamVingtEtUnCinqEnZero": _triplet(sy.solution_5_en(sp.Integer(0))),
    }


def macros_de_l_examen_2021() -> dict[str, str]:
    """Toutes les formules que le corrige de janvier 2021 compose."""
    return {**_analyse(), **_systemes()}
