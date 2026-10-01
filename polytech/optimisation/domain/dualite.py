"""La dualite : le programme dual, et les trois theoremes du chapitre IV.

Le dual de `max c.x` sous `A x <= b`, `x >= 0` est `min b.y` sous
`A^T y >= c`, `y >= 0`. Trois enonces en decoulent, et tous trois sont
**verifiables**, ce qui en fait les meilleurs tests possibles :

* la dualite **faible** : toute paire realisable verifie `c.x <= b.y` ;
* la dualite **forte** : a l'optimum, les deux valeurs sont egales ;
* les **ecarts complementaires** : pour chaque contrainte, ou l'ecart
  primal est nul, ou la variable duale l'est.

Le troisieme est le plus utile en pratique : il permet de lire la solution
d'un probleme sur l'autre, ce que le chapitre appelle « interpretation ».
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.optimisation.domain.forme_standard import ProgrammeLineaire


def dual_de(primal: ProgrammeLineaire) -> ProgrammeLineaire:
    """Le dual, ecrit lui aussi comme une maximisation.

    `min b.y` sous `A^T y >= c` devient `max (-b).y` sous `(-A^T) y <= -c`,
    ce qui le ramene a la forme que `ProgrammeLineaire` connait. Le signe
    est porte ici, une fois, plutot qu'a chaque exercice.
    """
    return ProgrammeLineaire(
        objectif=-primal.seconds_membres,
        contraintes=-primal.contraintes.T,
        seconds_membres=-primal.objectif,
    )


@dataclass(frozen=True, slots=True)
class PaireDuale:
    """Un primal et un point realisable de chacun des deux programmes."""

    primal: ProgrammeLineaire
    point_primal: sp.Matrix
    point_dual: sp.Matrix

    def valeur_primale(self) -> sp.Expr:
        """`c.x`."""
        return self.primal.valeur(self.point_primal)

    def valeur_duale(self) -> sp.Expr:
        """`b.y`, la borne que le dual fournit."""
        return sp.simplify((self.primal.seconds_membres.T * self.point_dual)[0, 0])

    def ecart_de_dualite(self) -> sp.Expr:
        """`b.y - c.x`, nul si et seulement si les deux points sont optimaux."""
        return sp.simplify(self.valeur_duale() - self.valeur_primale())

    def verifie_la_dualite_faible(self) -> bool:
        """Vrai si `c.x <= b.y` : la borne duale encadre bien le primal."""
        return bool(self.ecart_de_dualite() >= 0)

    def ecarts_primaux(self) -> sp.Matrix:
        """`b - A x` : l'ecart de chaque contrainte primale."""
        return self.primal.seconds_membres - self.primal.contraintes * self.point_primal

    def verifie_les_ecarts_complementaires(self) -> bool:
        """Vrai si, pour chaque contrainte, l'ecart ou la variable duale est nul."""
        paires = zip(self.ecarts_primaux(), self.point_dual, strict=True)
        return all(sp.simplify(ecart * duale) == 0 for ecart, duale in paires)
