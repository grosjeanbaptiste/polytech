"""Un systeme lineaire, et les deux questions que le cours lui pose.

Le resoudre, et dire **a quelle condition** un systeme homogene admet
autre chose que la solution nulle -- c'est la question 2.3 du fascicule,
posee sur des systemes parametriques.

La solution est verifiee par substitution, et non par confiance : c'est le
seul controle qui ne depende d'aucune lecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class SystemeLineaire:
    """La matrice du systeme et son second membre."""

    matrice: sp.Matrix
    second_membre: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse un second membre dont la hauteur n'est pas celle du systeme."""
        if self.second_membre.rows != self.matrice.rows:
            msg = (
                f"second membre de hauteur {self.second_membre.rows} pour "
                f"{self.matrice.rows} equations"
            )
            raise ValueError(msg)

    def est_homogene(self) -> bool:
        """Vrai si le second membre est nul : la solution nulle est acquise."""
        return bool(self.second_membre == sp.zeros(self.matrice.rows, 1))

    def rang(self) -> int:
        """Le rang de la matrice, dont depend la dimension des solutions."""
        return int(self.matrice.rank())

    def admet_des_solutions_non_nulles(self) -> bool:
        """Vrai si le noyau n'est pas reduit a zero.

        Pour un systeme homogene, c'est la condition du fascicule : le
        rang doit etre strictement inferieur au nombre d'inconnues.
        """
        return self.rang() < int(self.matrice.cols)

    def solution(self) -> sp.Matrix:
        """Une solution du systeme, par la pseudo-resolution de sympy."""
        return sp.Matrix(self.matrice.solve(self.second_membre))

    def solution_verifiee(self) -> bool:
        """Vrai si la solution trouvee satisfait vraiment le systeme."""
        ecart = sp.simplify(self.matrice * self.solution() - self.second_membre)
        return bool(ecart == sp.zeros(self.matrice.rows, 1))
