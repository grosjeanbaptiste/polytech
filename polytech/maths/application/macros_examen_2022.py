r"""Macros du corrige de l'examen de janvier 2022, prefixe `ExamVingtDeux`."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import courbes_examens as courbes
from polytech.maths.application import examen_2022 as ex
from polytech.maths.application.formatage import formule, nombre
from polytech.maths.domain.symboles import TEMPS, VARIABLE

_ECHANGE = {VARIABLE: TEMPS, TEMPS: VARIABLE}
"""L'enonce integre en `t` et resout en `y(x)` ; le domaine fait l'inverse."""


def _enonce(expression: sp.Expr) -> sp.Expr:
    return expression.xreplace(_ECHANGE)


def macros_de_l_examen_2022() -> dict[str, str]:
    """Toutes les formules que le corrige de janvier 2022 compose."""
    etape, primitive, valeur = ex.integrale()
    compatible, solution = ex.systeme()
    image, fractions, reponse = ex.equation()
    return {
        "ExamVingtDeuxEtape": formule(_enonce(etape)),
        "ExamVingtDeuxPrimitive": formule(_enonce(primitive)),
        "ExamVingtDeuxIntegrale": formule(_enonce(valeur)),
        "ExamVingtDeuxCompatible": formule(compatible),
        "ExamVingtDeuxDroite": formule(sp.Matrix([solution])),
        "ExamVingtDeuxAire": nombre(courbes.aire_2022(), 4),
        "ExamVingtDeuxCourbe": courbes.integrande_2022(),
        "ExamVingtDeuxImage": formule(sp.factor(image)),
        "ExamVingtDeuxFractions": formule(fractions),
        "ExamVingtDeuxSolution": formule(_enonce(reponse)),
    }
