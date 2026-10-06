"""Mathematique, exercices 1, section 1.2 : les dix-neuf integrales, par numero.

Fichier de donnees : le registre qui associe a chaque numero de la feuille
la fonction qui l'etablit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.maths.application import definies_a_exs_01 as da
from polytech.maths.application import definies_b_exs_01 as db

if TYPE_CHECKING:
    from collections.abc import Callable

    from polytech.maths.application.newton_exs_01 import Definie

INTEGRALES: Final[dict[int, Callable[[], Definie]]] = {
    1: da.integrale_1,
    2: da.integrale_2,
    3: da.integrale_3,
    4: da.integrale_4,
    5: da.integrale_5,
    6: da.integrale_6,
    7: da.integrale_7,
    8: da.integrale_8,
    9: da.integrale_9,
    10: da.integrale_10,
    11: db.integrale_11,
    12: db.integrale_12,
    13: db.integrale_13,
    14: db.integrale_14,
    15: db.integrale_15,
    16: db.integrale_16,
    17: db.integrale_17,
    18: db.integrale_18,
    19: db.integrale_19,
}
