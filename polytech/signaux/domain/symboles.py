"""Les symboles du contexte, declares une fois pour toutes.

Le cours note la variable de Laplace `p` et la pulsation `omega`. Les
declarer ici evite que deux modules en forgent deux versions que sympy
tiendrait pour differentes.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

IMAGE: Final = sp.Symbol("p")
"""La variable de Laplace, telle que le cours la note."""

PULSATION: Final = sp.Symbol("omega", positive=True)
"""La pulsation, en rad/s."""

TEMPS: Final = sp.Symbol("t", nonnegative=True)
"""Le temps, pour les reponses temporelles."""

GAIN: Final = sp.Symbol("K", positive=True)
"""Le gain reglable de la boucle, dont depend la stabilite."""
