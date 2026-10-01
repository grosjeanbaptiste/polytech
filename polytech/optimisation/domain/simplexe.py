"""L'algorithme simplexe : le tableau, le changement de base, les arrets.

Le chapitre II enseigne le geste -- choisir la colonne entrante au
coefficient favorable, la ligne sortante au plus petit rapport, pivoter --
et c'est ce geste qui est expose, iteration par iteration. Un corrige pas
a pas doit pouvoir montrer chaque tableau ; rendre seulement l'optimum
priverait le lecteur de tout ce que le chapitre apprend.

Les calculs sont exacts : les coefficients sont des rationnels sympy, et
non des flottants. Un simplexe en flottants accumule l'erreur de pivot et
finit par declarer realisable une base qui ne l'est pas -- c'est le defaut
classique des implementations naives, et il n'a pas sa place ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.forme_standard import ProgrammeLineaire

ITERATIONS_MAXIMALES = 200
"""Garde-fou : un cyclage ferait tourner la boucle sans fin."""


@dataclass(frozen=True, slots=True)
class Iteration:
    """L'etat du tableau a une iteration, tel que le corrige l'affiche."""

    base: tuple[int, ...]
    tableau: sp.Matrix
    valeur: sp.Expr


@dataclass(frozen=True, slots=True)
class Simplexe:
    """Le simplexe primal, sur un programme en forme standard."""

    programme: ProgrammeLineaire

    def _depart(self) -> tuple[list[int], sp.Matrix]:
        """La base triviale des variables d'ecart ; leve si `b` a un negatif.

        Le simplexe **primal** part d'une base realisable. Si un second
        membre est negatif, la base triviale ne l'est pas, et il faut le
        dual simplexe -- le dire ici evite de pivoter sur une base
        infaisable et de rendre un « optimum » qui n'en est pas un.
        """
        negatifs = [b for b in self.programme.seconds_membres if b < 0]
        if negatifs:
            msg = (
                f"second membre negatif {negatifs} : la base triviale n'est pas "
                f"realisable, employer le dual simplexe"
            )
            raise ValueError(msg)
        decalage = self.programme.variables
        base = [decalage + rang for rang in range(self.programme.equations)]
        tableau = self.programme.matrice_etendue().row_join(
            self.programme.seconds_membres,
        )
        return base, tableau

    def _couts_reduits(self, base: list[int], tableau: sp.Matrix) -> sp.Matrix:
        """`c - c_B B^-1 A` : les coefficients qui disent s'il faut continuer."""
        total = self.programme.variables + self.programme.equations
        couts = sp.zeros(total, 1)
        for rang in range(self.programme.variables):
            couts[rang, 0] = self.programme.objectif[rang, 0]
        duals = (
            sp.Matrix(
                [couts[colonne, 0] for colonne in base],
            ).T
            * tableau[:, :total]
        )
        return (couts.T - duals).T

    def iterations(self) -> tuple[Iteration, ...]:
        """Toutes les iterations, de la base triviale a l'optimum."""
        base, tableau = self._depart()
        total = self.programme.variables + self.programme.equations
        etapes: list[Iteration] = []
        for _ in range(ITERATIONS_MAXIMALES):
            valeur = sum(
                self.programme.objectif[colonne, 0] * tableau[rang, total]
                for rang, colonne in enumerate(base)
                if colonne < self.programme.variables
            )
            etapes.append(Iteration(tuple(base), tableau.copy(), sp.simplify(valeur)))
            couts = self._couts_reduits(base, tableau)
            entrante = next(
                (c for c in range(total) if couts[c, 0] > 0 and c not in base),
                None,
            )
            if entrante is None:
                return tuple(etapes)
            rapports = [
                (tableau[rang, total] / tableau[rang, entrante], rang)
                for rang in range(self.programme.equations)
                if tableau[rang, entrante] > 0
            ]
            if not rapports:
                msg = f"programme non borne : la colonne {entrante} ne limite rien"
                raise ValueError(msg)
            _, sortante = min(rapports)
            tableau[sortante, :] /= tableau[sortante, entrante]
            for rang in range(self.programme.equations):
                if rang != sortante:
                    facteur = tableau[rang, entrante]
                    tableau[rang, :] -= facteur * tableau[sortante, :]
            base[sortante] = entrante
        msg = f"le simplexe n'a pas converge en {ITERATIONS_MAXIMALES} iterations"
        raise ValueError(msg)

    def optimum(self) -> tuple[sp.Matrix, sp.Expr]:
        """Le point optimal et sa valeur, au terme des iterations."""
        derniere = self.iterations()[-1]
        total = self.programme.variables + self.programme.equations
        point = sp.zeros(self.programme.variables, 1)
        for rang, colonne in enumerate(derniere.base):
            if colonne < self.programme.variables:
                point[colonne, 0] = derniere.tableau[rang, total]
        return point, derniere.valeur
