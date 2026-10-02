"""Les donnees de l'exercice 6, et elles ne viennent que de la transcription.

Un seul endroit les porte, pour qu'une donnee mal lue se corrige une fois
et se propage partout.

La transcription est celle du scan manuscrit, sous
`transcriptions/Electromagnetisme/EXS/Exs_03_Stationnaire_CircuitsMagnetiques_06.md`
-- une page dont l'OCR automatique n'avait produit que du charabia.

Les sections sont donnees en centimetres dans l'enonce et converties ici
une fois pour toutes : melanger les unites dans le texte est l'erreur que
cette conversion unique previent.
"""

from __future__ import annotations

import math
from typing import Final

SPIRES: Final = 1000
"""`N`, le nombre de spires de l'enroulement."""

INDUCTION_H: Final = 1.6
"""`B_H`, en teslas. Donnee de l'enonce pour la branche a entrefer."""

CHAMP_LU_H: Final = 70.0
"""`H_H`, en A/m, **lu sur la courbe** du materiau pour cette induction."""

RAYON_H: Final = 0.16
"""Le rayon moyen du contour H, en metres."""

RAYON_T: Final = 0.175
"""Le rayon moyen du contour T, en metres."""

SECTION_H: Final = 2e-2 * 4e-2
"""`S_H = 2 x 4` cm, en metres carres."""

SECTION_T: Final = 1e-2 * 4e-2
"""`S_T = 1 x 4` cm, en metres carres."""

INDUCTION_T: Final = 0.5
"""`B_T`, en teslas, **lu sur la courbe** pour le champ `H_T` trouve."""

LONGUEUR_H: Final = 2 * math.pi * RAYON_H
"""`l_H`, la longueur du contour H, en metres."""

LONGUEUR_T: Final = 2 * math.pi * RAYON_T
"""`l_T`, la longueur du contour T, en metres."""

DECIMALES_AFFICHEES: Final = 2
DECIMALES_FINES: Final = 5
"""Cinq decimales pour le courant : `0,07037` A est la precision du manuscrit."""
