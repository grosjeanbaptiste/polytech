"""Solution maximale, solution globale (planches 29 a 31).

Une solution `y~` **prolonge** `y` si elle est definie sur un intervalle
plus grand et coincide avec `y` la ou `y` existe. Une solution est
**maximale** si elle n'admet aucun prolongement, **globale** si elle est
definie sur tout l'intervalle considere. Toute solution globale est
maximale ; la reciproque est fausse.

Pour `y' = y^2` : `y = 0` est globale, mais `y = -1/(t + C)` n'existe que
sur `]-oo, -C[` ou sur `]-C, +oo[` -- deux solutions maximales, aucune
globale. C'est la « duree de vie » d'une solution.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.util import continuous_domain

from polytech.maths.domain.differentielles.poles_periodiques import poles_dans
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE


def _domaine(solution: sp.Expr, etude: sp.Interval | None) -> sp.Set:
    if solution.atoms(sp.tan, sp.cot):
        if etude is None:
            msg = "tan/cot : poles periodiques, intervalle d'etude requis"
            raise ValueError(msg)
        return etude - poles_dans(solution, etude)
    domaine = continuous_domain(solution, VARIABLE, sp.S.Reals)
    return domaine if etude is None else domaine.intersect(etude)


def intervalles_de_definition(
    solution: sp.Expr, etude: sp.Interval | None = None
) -> tuple[sp.Set, ...]:
    """Les intervalles maximaux sur lesquels l'expression definit une solution.

    `etude` restreint le resultat ; elle est **obligatoire et bornee** si
    l'expression contient `tan` ou `cot` (leurs poles sont une infinite).
    """
    domaine = _domaine(sp.sympify(solution), etude)
    morceaux = domaine.args if isinstance(domaine, sp.Union) else (domaine,)
    return tuple(sorted(morceaux, key=lambda intervalle: intervalle.inf))


def est_globale(solution: sp.Expr) -> bool:
    """Vrai si la solution est definie sur toute la droite reelle."""
    return intervalles_de_definition(solution) == (sp.S.Reals,)
