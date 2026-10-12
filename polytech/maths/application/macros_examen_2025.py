r"""Macros du corrige de l'examen de janvier 2025, prefixe `ExamVingtCinq`."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import courbes_examens as courbes
from polytech.maths.application import examen_2025 as ex
from polytech.maths.application.formatage import formule, nombre


def macros_de_l_examen_2025() -> dict[str, str]:
    """Toutes les formules que le corrige de janvier 2025 compose."""
    termes, primitive, valeur = ex.integrale()
    solutions = ex.systeme()
    image, fractions, reponse = ex.equation()
    return {
        "ExamVingtCinqTermes": formule(sp.Add(*termes, evaluate=False)),
        "ExamVingtCinqPrimitive": formule(primitive),
        "ExamVingtCinqIntegrale": formule(valeur),
        "ExamVingtCinqEnZero": formule(sp.Matrix([solutions[sp.Integer(0)]])),
        "ExamVingtCinqEnUnDemi": formule(sp.Matrix([solutions[sp.Rational(1, 2)]])),
        "ExamVingtCinqAire": nombre(courbes.aire_2025(), 4),
        "ExamVingtCinqCourbe": courbes.integrande_2025(),
        "ExamVingtCinqImage": formule(sp.factor(image)),
        "ExamVingtCinqFractions": formule(fractions),
        "ExamVingtCinqSolution": formule(reponse),
    }
