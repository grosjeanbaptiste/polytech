"""Les symboles et les donnees de la seance 3.1.

Les noms sont ceux du recueil. L'exercice 1 est le seul numerique : ses
cinq valeurs sont portees en rationnels exacts, de sorte que les
decimales du manuscrit -- 0,226 et 21,24 -- se retrouvent par arrondi et
non par accumulation d'arrondis.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

R, R1, R2 = sp.symbols("R R_1 R_2", positive=True)
L1, L2, L12 = sp.symbols("L_1 L_2 L_12", positive=True)
C = sp.Symbol("C", positive=True)
TENSION = sp.Symbol("E", positive=True)
DUREE = sp.Symbol("T", positive=True)
"""La duree de l'impulsion rectangulaire de l'exercice 7."""

FEM_DU_RELAIS: Final = sp.Integer(12)
"""`E = 12 V`."""

RESISTANCE_INTERNE: Final = sp.Rational(1, 10)
"""`R_s = 0,1 ohm`."""

RESISTANCE_DE_LIGNE: Final = sp.Integer(3)
"""`R = 3 ohms`."""

RESISTANCE_DE_LA_BOBINE: Final = sp.Integer(50)
"""`R_b = 50 ohms`."""

INDUCTANCE_DE_LA_BOBINE: Final = sp.Rational(5, 2)
"""`L = 2,5 H`."""

SEUIL_DU_RELAIS: Final = sp.Rational(5, 100)
"""« Quelque 0,05 A » : le courant a partir duquel le relais s'enclenche."""
