"""L'optimum d'un programme lineaire, exige fini.

`resoudre` rend une issue -- optimale, non bornee, impossible -- et ne
porte une valeur que dans le premier cas. Un corrige qui annonce un
optimum en veut un : cette fonction le rend, ou dit pourquoi il n'y en a
pas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.formulation import ProgrammeGeneral

Point = tuple["sp.Expr", ...]


def optimum_fini(programme: ProgrammeGeneral) -> tuple[sp.Expr, Point]:
    """La valeur et le point optimaux ; leve si le programme n'en a pas."""
    solution = resoudre(programme)
    if solution.valeur is None or solution.point is None:
        msg = f"pas d'optimum fini : {solution.issue.value}"
        raise ValueError(msg)
    return solution.valeur, solution.point
