"""Ecrire une expression lineaire et une contrainte comme le cours les imprime.

`7 u_1 - 4 u_2 + 5 u_3` : les termes nuls omis, le coefficient 1 tu, les
signes lies aux termes. Les explications en ont besoin pour dire une
ligne d'un programme dans la forme ou l'etudiant la lit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy as sp

    from polytech.optimisation.domain.contraintes import Contrainte


def ecrire_lineaire(coefficients: Sequence[sp.Expr], lettre: str) -> str:
    """`7 u_1 - 4 u_2 + 5 u_3` : les termes nuls omis, les signes lies."""
    texte = ""
    for rang, a in enumerate(coefficients, start=1):
        if a == 0:
            continue
        signe = ("-" if a < 0 else "") if not texte else (" - " if a < 0 else " + ")
        facteur = "" if abs(a) == 1 else f"{abs(a)} "
        texte += f"{signe}{facteur}{lettre}_{rang}"
    return texte or "0"


def ecrire_contrainte(contrainte: Contrainte, lettre: str) -> str:
    """Le premier membre, le sens, le second membre."""
    membre = ecrire_lineaire(contrainte.coefficients, lettre)
    return f"{membre} {contrainte.sens.value} {contrainte.second_membre}"
