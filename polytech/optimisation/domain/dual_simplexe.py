"""L'algorithme dual simplexe, et ce qui le distingue du primal.

Le primal part d'une base **realisable** et ameliore l'objectif. Le dual
simplexe part d'une base **dualement** realisable -- tous les couts
reduits favorables -- mais primalement infaisable, et retablit la
faisabilite. C'est ce qui le rend utile quand une contrainte est ajoutee
apres coup : la base precedente reste dualement realisable, et il n'y a
pas a tout recommencer. C'est exactement la situation du chapitre XIV,
ou chaque branchement ajoute une contrainte.

Le choix des lignes et colonnes est donc **inverse** de celui du primal :
la ligne sortante est celle dont le second membre est le plus negatif, et
la colonne entrante est choisie par un rapport sur les couts reduits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.forme_standard import ProgrammeLineaire

ITERATIONS_MAXIMALES = 200


@dataclass(frozen=True, slots=True)
class DualSimplexe:
    """Le dual simplexe, sur un tableau et une base donnes."""

    programme: ProgrammeLineaire
    base: tuple[int, ...]
    tableau: sp.Matrix

    @property
    def colonne_du_second_membre(self) -> int:
        """L'indice de la derniere colonne, celle de `b`."""
        return self.programme.variables + self.programme.equations

    def ligne_sortante(self) -> int | None:
        """La ligne au second membre le plus negatif, ou rien si realisable."""
        colonne = self.colonne_du_second_membre
        negatifs = [
            (self.tableau[rang, colonne], rang)
            for rang in range(self.programme.equations)
            if self.tableau[rang, colonne] < 0
        ]
        return min(negatifs)[1] if negatifs else None

    def cout_reduit(self, colonne: int) -> sp.Expr:
        """`c_B B^-1 a_j - c_j` ; les variables d'ecart ont un cout nul."""
        couts = [*self.programme.objectif, *([0] * self.programme.equations)]
        de_base = sum(
            (
                couts[b] * self.tableau[rang, colonne]
                for rang, b in enumerate(self.base)
            ),
            sp.Integer(0),
        )
        return de_base - couts[colonne]

    def colonne_entrante(self, sortante: int) -> int:
        """Le rapport dual : le plus petit `|cout reduit / coefficient|`.

        Sur les seuls coefficients negatifs de la ligne sortante. C'est ce
        rapport qui garde la base duale admissible (chapitre V.1, cas 2.b) ;
        choisir sur le coefficient seul retablissait l'admissibilite sans
        garantir l'optimum.
        """
        candidates = [
            (abs(self.cout_reduit(c) / self.tableau[sortante, c]), c)
            for c in range(self.colonne_du_second_membre)
            if self.tableau[sortante, c] < 0 and c not in self.base
        ]
        if not candidates:
            msg = (
                f"primal infaisable : la ligne {sortante} n'a aucun coefficient "
                f"negatif, donc aucune colonne ne peut y entrer"
            )
            raise ValueError(msg)
        return min(candidates)[1]

    def pivoter(self) -> DualSimplexe:
        """Une iteration : un pivot, et la base mise a jour."""
        sortante = self.ligne_sortante()
        if sortante is None:
            msg = "base deja primalement realisable : rien a pivoter"
            raise ValueError(msg)
        entrante = self.colonne_entrante(sortante)
        tableau = self.tableau.copy()
        tableau[sortante, :] /= tableau[sortante, entrante]
        for rang in range(self.programme.equations):
            if rang != sortante:
                tableau[rang, :] -= tableau[rang, entrante] * tableau[sortante, :]
        base = list(self.base)
        base[sortante] = entrante
        return DualSimplexe(
            programme=self.programme,
            base=tuple(base),
            tableau=tableau,
        )

    def retablir_la_faisabilite(self) -> DualSimplexe:
        """Pivote jusqu'a ce que tous les seconds membres soient positifs."""
        courant = self
        for _ in range(ITERATIONS_MAXIMALES):
            if courant.ligne_sortante() is None:
                return courant
            courant = courant.pivoter()
        msg = f"le dual simplexe n'a pas converge en {ITERATIONS_MAXIMALES} iterations"
        raise ValueError(msg)

    def point(self) -> sp.Matrix:
        """Le point correspondant a la base courante."""
        valeurs = sp.zeros(self.programme.variables, 1)
        for rang, colonne in enumerate(self.base):
            if colonne < self.programme.variables:
                valeurs[colonne, 0] = self.tableau[rang, self.colonne_du_second_membre]
        return valeurs
