"""Mathematique, exercices 1, section 1.1 : les vingt primitives, par numero.

Fichier de donnees : le registre qui associe a chaque numero de la feuille
la fonction qui le resout. Les lettres distinguent les integrales d'un
meme numero, dans l'ordre ou la feuille les ecrit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.maths.application import primitives_a_exs_01 as pa
from polytech.maths.application import primitives_b_exs_01 as pb
from polytech.maths.application.resolution_exs_01 import A, B, Resolution

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["PRIMITIVES", "A", "B", "formes_linearisees", "primitive_du_12_sur_r"]

formes_linearisees = pa.formes_linearisees
primitive_du_12_sur_r = pb.primitive_du_12_sur_r

PRIMITIVES: Final[dict[str, Callable[[], Resolution]]] = {
    "1": pa.exercice_1,
    "2a": lambda: pa.exercice_2()[0],
    "2b": lambda: pa.exercice_2()[1],
    "2c": lambda: pa.exercice_2()[2],
    "2d": lambda: pa.exercice_2()[3],
    "3": pa.exercice_3,
    "4a": lambda: pa.exercice_4()[0],
    "4b": lambda: pa.exercice_4()[1],
    "4c": lambda: pa.exercice_4()[2],
    "5": pa.exercice_5,
    "6": pa.exercice_6,
    "7": pa.exercice_7,
    "8": pb.exercice_8,
    "9a": lambda: pb.exercice_9()[0],
    "9b": lambda: pb.exercice_9()[1],
    "10": pb.exercice_10,
    "11": pb.exercice_11,
    "12": pb.exercice_12,
    "13": pb.exercice_13,
    "14": pb.exercice_14,
}
