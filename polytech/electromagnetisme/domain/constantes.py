"""Les constantes du vide, et la relation qui les lie a la vitesse de la lumiere.

Les trois sont declarees independamment, et c'est volontaire :
`c = 1 / racine(epsilon_0 mu_0)` devient alors une **verification** et non
une definition. Si l'une des trois etait mal transcrite, le test qui
recoupe les trois tomberait -- ce qu'une definition circulaire n'aurait
jamais permis de voir.
"""

from __future__ import annotations

from typing import Final

PERMITTIVITE_DU_VIDE: Final = 8.8541878128e-12
"""`epsilon_0`, en F/m."""

PERMEABILITE_DU_VIDE: Final = 1.25663706212e-6
"""`mu_0`, en H/m."""

VITESSE_DE_LA_LUMIERE: Final = 2.99792458e8
"""`c`, en m/s. Valeur exacte par definition du metre."""

CHARGE_ELEMENTAIRE: Final = 1.602176634e-19
"""`e`, en coulombs."""
