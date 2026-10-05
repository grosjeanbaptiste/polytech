"""La discussion complete de `a x^2 + b x + c = 0`.

Le syllabus (exercice 3.4) : « examinez toutes les possibilites (racines
reelles, complexes, doubles, infinite ou inexistence de solutions) ». C'est
un exercice de structures alternatives : l'ordre des tests compte, et le
cas `a = 0` doit etre ecarte avant de diviser par `2a`.

1. `a = 0` et `b = 0` : l'equation se reduit a `c = 0` -- toute valeur
   convient si `c` est nul, aucune sinon ;
2. `a = 0` seulement : premier degre, `x = -c/b` ;
3. sinon le discriminant `b^2 - 4ac` decide : deux racines reelles, une
   racine double, ou deux racines complexes conjuguees.

Le discriminant est celui du cours de mathematique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from polytech.maths.domain.algebre.racines import discriminant


class Nature(Enum):
    """Ce que la discussion peut conclure."""

    INFINITE_DE_SOLUTIONS = "infinite de solutions"
    AUCUNE_SOLUTION = "aucune solution"
    PREMIER_DEGRE = "premier degre"
    DEUX_RACINES_REELLES = "deux racines reelles"
    RACINE_DOUBLE = "racine double"
    DEUX_RACINES_COMPLEXES = "deux racines complexes"


@dataclass(frozen=True, slots=True)
class Discussion:
    """La nature de l'equation, et ses racines par ordre croissant."""

    nature: Nature
    racines: tuple[complex, ...] = ()


def discuter_le_second_degre(a: float, b: float, c: float) -> Discussion:
    """Toutes les possibilites, dans l'ordre ou un programme doit les tester."""
    if a == 0 and b == 0:
        nature = Nature.INFINITE_DE_SOLUTIONS if c == 0 else Nature.AUCUNE_SOLUTION
        return Discussion(nature)
    if a == 0:
        return Discussion(Nature.PREMIER_DEGRE, (-c / b,))
    delta = discriminant(a, b, c)
    if delta == 0:
        return Discussion(Nature.RACINE_DOUBLE, (-b / (2 * a),))
    if delta > 0:
        ecart = math.sqrt(delta) / abs(2 * a)
        centre = -b / (2 * a)
        return Discussion(Nature.DEUX_RACINES_REELLES, (centre - ecart, centre + ecart))
    imaginaire = math.sqrt(-delta) / abs(2 * a)
    conjuguees = (complex(-b / (2 * a), -imaginaire), complex(-b / (2 * a), imaginaire))
    return Discussion(Nature.DEUX_RACINES_COMPLEXES, conjuguees)
