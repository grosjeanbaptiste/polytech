"""Les systemes nodaux des six circuits du TP 4 d'electronique.

Aucune matrice n'est ecrite ici : les circuits sont des listes d'elements
(`netlistes_tp_04`), et le domaine (`netliste`) les assemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.electronique.application.netlistes_tp_04 import (
    CASCODE,
    CIRCUIT_1,
    CIRCUIT_2,
    CIRCUIT_3,
    DEPHASEUR,
    SOURCE_COMMUNE,
)
from polytech.electronique.domain.netliste import resoudre, systeme_nodal_modifie

if TYPE_CHECKING:
    import sympy as sp


def systeme_du_circuit_1() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 2.1, premier circuit : `[V_1, V_2, I_1]`."""
    return systeme_nodal_modifie(CIRCUIT_1, 2)


def solution_du_circuit_1() -> sp.Matrix:
    """Les tensions des deux noeuds, puis le courant de la source."""
    return resoudre(CIRCUIT_1, 2)


def systeme_du_circuit_2() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 2.1, deuxieme circuit : `[V_1, ..., V_4, I_1]`."""
    return systeme_nodal_modifie(CIRCUIT_2, 4)


def solution_du_circuit_2() -> sp.Matrix:
    """Les tensions des quatre noeuds, puis le courant de la source."""
    return resoudre(CIRCUIT_2, 4)


def systeme_du_circuit_3() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 2.1, troisieme circuit : `[V_1, V_2, V_3, I_y, I_x]`."""
    return systeme_nodal_modifie(CIRCUIT_3, 3)


def solution_du_circuit_3() -> sp.Matrix:
    """Les tensions des trois noeuds, puis `I_y` et `I_x`."""
    return resoudre(CIRCUIT_3, 3)


def systeme_du_dephaseur() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 3.1 : trois cellules `C`-`R` et la charge `R_L`."""
    return systeme_nodal_modifie(DEPHASEUR, 4)


def systeme_de_la_source_commune() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 3.2 : le schema petits signaux du MOSFET degenere."""
    return systeme_nodal_modifie(SOURCE_COMMUNE, 4)


def systeme_de_la_cascode() -> tuple[sp.Matrix, sp.Matrix]:
    """Exercice 3.3 : deux transistors empiles."""
    return systeme_nodal_modifie(CASCODE, 4)
