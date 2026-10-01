"""Les symboles de la conduction, declares une fois pour toutes.

Les declarer ici, et non a l'endroit de chaque calcul, evite que deux
modules forgent deux symboles homonymes que sympy tiendrait pour
differents -- une simplification qui ne simplifie rien, sans message.

`x` monte du sol (x = 0) vers la surface exposee (x = e). Ce sens est le
seul de la bibliotheque : la densite de flux dans le sens de x vaut
`-lambda dt/dx`, et les deux conditions aux limites s'ecrivent donc avec
des signes opposes.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

ABSCISSE: Final = sp.Symbol("x", nonnegative=True)
"""Position dans l'epaisseur, comptee depuis la face basse."""

RAYON: Final = sp.Symbol("r", positive=True)
"""Rayon courant, pour la conduction a symetrie spherique."""

CONSTANTES: Final = (sp.Symbol("C_1", real=True), sp.Symbol("C_2", real=True))
"""Les deux constantes des deux integrations successives."""

TEMPERATURE: Final = sp.Function("t")
"""La temperature comme fonction inconnue, avant integration."""
