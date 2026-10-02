"""Les donnees de l'exercice 8, et elles ne viennent que de la transcription.

Un seul endroit les porte, pour qu'une donnee mal lue se corrige une fois
et se propage partout -- texte, calculs et recoupements.

La transcription est celle du scan manuscrit, sous
`transcriptions/Electromagnetisme/EXS/Exs_03_Stationnaire_Thevenin_08.md`,
relue contre le scan. Ces sept nombres sont tout l'enonce.
"""

from __future__ import annotations

from typing import Final

PREMIERE_TENSION: Final = 20.0
"""`E_1`, en volts."""

SECONDE_TENSION: Final = 10.0
"""`E_2`, en volts."""

PREMIERE_RESISTANCE: Final = 1.0
"""`R_1`, en ohms."""

SECONDE_RESISTANCE: Final = 5.0
"""`R_2`, en ohms."""

COURANT_INJECTE: Final = 1.0
"""`I`, la source de courant de la maille, en amperes."""

CHARGE: Final = 10.0
"""`R'`, la charge de la question 2, en ohms."""

COURANT_EN_PARALLELE: Final = 3.0
"""`I'`, la source de courant en parallele de la charge, en amperes."""

DECIMALES_AFFICHEES: Final = 2
"""Deux decimales : `E_TH = 19,17` V est la precision du manuscrit."""

DECIMALES_FINES: Final = 4
"""Pour les valeurs ou l'arrondi a deux decimales masquerait l'exactitude."""
