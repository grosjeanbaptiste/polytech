"""Les symboles du contexte, declares une fois pour toutes.

Le cours travaille en coordonnees cartesiennes et cylindriques, et les
exercices passent de l'une a l'autre -- l'exercice « Statique 1.1 » part
de la divergence cartesienne puis l'ecrit en cylindriques. Les deux jeux
coexistent donc, et porter `rho` pour le rayon cylindrique et non pour une
masse volumique est une convention du cours qu'il faut tenir.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

ABSCISSE: Final = sp.Symbol("x", real=True)
ORDONNEE: Final = sp.Symbol("y", real=True)
COTE: Final = sp.Symbol("z", real=True)
CARTESIENNES: Final = (ABSCISSE, ORDONNEE, COTE)

RAYON: Final = sp.Symbol("rho", positive=True)
"""Le rayon cylindrique. Le cours le note `rho`, et non `r`."""

AZIMUT: Final = sp.Symbol("theta", real=True)
CYLINDRIQUES: Final = (RAYON, AZIMUT, COTE)

TEMPS: Final = sp.Symbol("t", real=True)
PULSATION: Final = sp.Symbol("omega", positive=True)
