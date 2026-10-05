"""Les approximations successives, menees jusqu'a la convergence.

`contraction.iterer` applique une fonction un nombre de fois donne ; ici
on repete `x <- g(x)` jusqu'a ce que deux iteres successifs ne se
distinguent plus. La suite converge quand `g` est contractante pres de
la solution, et sa convergence est alors lineaire.

Le cours de thermique l'emploie pour le coefficient global de
transmission (jeu 10, planche 18 ; tome 2, page 108) : les coefficients
dependent des temperatures de paroi, qui dependent d'eux. « Rares sont
les cas ou cette methode simple n'est pas rapidement convergente. »
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

ITERATIONS = 200


def approximations_successives(
    fonction: Callable[[float], float],
    depart: float,
    *,
    tolerance: float,
    iterations: int = ITERATIONS,
) -> tuple[float, int]:
    """Le point fixe de `fonction` et le nombre de tours qu'il a fallu."""
    courant = depart
    for tour in range(1, iterations + 1):
        suivant = fonction(courant)
        if abs(suivant - courant) <= tolerance:
            return suivant, tour
        courant = suivant
    msg = f"pas de convergence en {iterations} iterations"
    raise ValueError(msg)
