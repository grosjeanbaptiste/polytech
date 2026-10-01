"""Les symboles du contexte, declares une fois pour toutes.

Les declarer ici evite que deux modules forgent deux symboles homonymes
que sympy tiendrait pour differents : une simplification qui ne simplifie
rien, et sans message.

Les hypotheses portees par chaque symbole ne sont pas decoratives. `x`
reel laisse sympy simplifier `sqrt(x**2)` en `Abs(x)` et non en `x` ;
`p` de partie reelle positive est ce qui rend la transformee de Laplace
convergente ; `n` entier positif est ce qui autorise les recurrences.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

VARIABLE: Final = sp.Symbol("x", real=True)
"""La variable d'integration et d'algebre."""

TEMPS: Final = sp.Symbol("t", real=True, nonnegative=True)
"""Le temps, pour les EDO et la transformee de Laplace."""

IMAGE: Final = sp.Symbol("p", positive=True)
"""La variable de Laplace. Le cours la note `p`, et non `s`."""

CONSTANTE: Final = sp.Symbol("C", real=True)
"""La constante d'integration, que le cours ecrit explicitement."""

ENTIER: Final = sp.Symbol("n", integer=True, positive=True)
"""L'indice des recurrences et des puissances."""
