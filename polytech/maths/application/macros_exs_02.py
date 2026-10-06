r"""Macros des exercices 2 de mathematique : equations differentielles.

Trois seances, trois modules ; celui-ci les reunit. Le prefixe est
`MathDeux`, et un nom de macro n'a pas de chiffre : `\MathDeuxSolutionSept`
est la solution de la septieme equation de l'exercice I.1.
"""

from __future__ import annotations

from polytech.maths.application.macros_a_exs_02 import macros_de_la_seance_un
from polytech.maths.application.macros_b_exs_02 import macros_de_la_seance_deux
from polytech.maths.application.macros_c_exs_02 import macros_de_la_seance_trois


def macros_des_exercices_deux() -> dict[str, str]:
    """Toutes les formules et les courbes que le corrige des exercices 2 compose."""
    return {
        **macros_de_la_seance_un(),
        **macros_de_la_seance_deux(),
        **macros_de_la_seance_trois(),
    }
