"""Un programme lineaire, sa forme standard, et ses sommets.

Le chapitre I s'intitule « Caracterisation algebrique des sommets de D »,
et c'est exactement ce que ce module porte : un sommet du domaine
realisable est une **solution de base realisable**, c'est-a-dire le choix
de `m` colonnes parmi `n + m` dont la sous-matrice est inversible et dont
la solution est positive.

Enumerer les bases est couteux -- c'est tout l'interet du simplexe, qui ne
les visite pas toutes -- mais c'est **exact**, et cela donne au simplexe un
recoupement independant : les deux doivent tomber sur le meme optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp


@dataclass(frozen=True, slots=True)
class ProgrammeLineaire:
    """`max c.x` sous `A x <= b`, `x >= 0`.

    Aucune condition de signe sur `b` : elle n'appartient pas au
    **probleme** mais a l'**algorithme**. Le simplexe primal a besoin
    d'une base de depart realisable, donc de `b >= 0`, et c'est lui qui
    l'exige. Le dual d'un programme a `b >= 0` a justement un second
    membre negatif -- c'est precisement pourquoi le dual simplexe existe,
    et l'interdire ici aurait rendu le chapitre V inexprimable.
    """

    objectif: sp.Matrix
    contraintes: sp.Matrix
    seconds_membres: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse les dimensions incoherentes et un second membre negatif."""
        if self.contraintes.cols != self.objectif.rows:
            msg = (
                f"objectif de dimension {self.objectif.rows} pour "
                f"{self.contraintes.cols} variables"
            )
            raise ValueError(msg)
        if self.contraintes.rows != self.seconds_membres.rows:
            msg = (
                f"second membre de dimension {self.seconds_membres.rows} pour "
                f"{self.contraintes.rows} contraintes"
            )
            raise ValueError(msg)

    @property
    def variables(self) -> int:
        """Le nombre de variables de decision, `n`."""
        return int(self.contraintes.cols)

    @property
    def equations(self) -> int:
        """Le nombre de contraintes, `m`."""
        return int(self.contraintes.rows)

    def matrice_etendue(self) -> sp.Matrix:
        """`[A | I]` : la forme standard, variables d'ecart comprises."""
        return self.contraintes.row_join(sp.eye(self.equations))

    def valeur(self, point: sp.Matrix) -> sp.Expr:
        """`c.x` au point donne."""
        return sp.simplify((self.objectif.T * point)[0, 0])

    def est_realisable(self, point: sp.Matrix) -> bool:
        """Vrai si `A x <= b` et `x >= 0`."""
        if any(valeur < 0 for valeur in point):
            return False
        ecarts = self.seconds_membres - self.contraintes * point
        return all(ecart >= 0 for ecart in ecarts)

    def sommets(self) -> tuple[sp.Matrix, ...]:
        """Les solutions de base realisables, en variables de decision."""
        etendue = self.matrice_etendue()
        total = self.variables + self.equations
        trouves: list[sp.Matrix] = []
        for base in combinations(range(total), self.equations):
            sous_matrice = etendue[:, list(base)]
            if sous_matrice.det() == 0:
                continue
            valeurs = sous_matrice.solve(self.seconds_membres)
            if any(valeur < 0 for valeur in valeurs):
                continue
            point = sp.zeros(self.variables, 1)
            for rang, colonne in enumerate(base):
                if colonne < self.variables:
                    point[colonne, 0] = valeurs[rang]
            if point not in trouves:
                trouves.append(point)
        return tuple(trouves)

    def optimum_par_enumeration(self) -> tuple[sp.Matrix, sp.Expr]:
        """Le meilleur sommet et sa valeur ; leve si le domaine est vide."""
        sommets = self.sommets()
        if not sommets:
            msg = "domaine realisable vide : aucune solution de base realisable"
            raise ValueError(msg)
        meilleur = max(sommets, key=self.valeur)
        return meilleur, self.valeur(meilleur)
