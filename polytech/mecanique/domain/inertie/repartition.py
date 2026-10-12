"""Une repartition de masse : ce sur quoi tout moment s'integre.

Les moments de la geometrie des masses sont tous des sommes « etendues a
tout le systeme, des masses des points, multipliees par une certaine
puissance des coordonnees de ces points » : la masse a l'ordre zero, le
centre de masse a l'ordre un, l'inertie a l'ordre deux. Une repartition
porte donc ce qu'il faut pour integrer un poids quelconque : la position
d'un point en fonction des parametres, l'element de masse et les bornes.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
section 6.1.1, page 103, relations (6.4) a (6.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur

type Borne = tuple[sp.Symbol, sp.Expr, sp.Expr]


@dataclass(frozen=True, slots=True)
class Repartition:
    """`dm = element * d(parametres)` au point `position`, entre les `bornes`."""

    position: Vecteur
    element: sp.Expr
    bornes: tuple[Borne, ...]

    def integrale(self, poids: sp.Expr) -> sp.Expr:
        """`integrale de poids dm`, etendue a tout le systeme."""
        return sp.simplify(sp.integrate(poids * self.element, *self.bornes))

    def masse(self) -> sp.Expr:
        """Le moment d'ordre zero, relation (6.4)."""
        return self.integrale(sp.Integer(1))

    def centre_de_masse(self) -> Vecteur:
        """Les moments du premier ordre, relation (6.5) : `m x_G = integrale x dm`."""
        masse = self.masse()
        if masse == 0:
            msg = "repartition sans masse : le centre de masse n'existe pas"
            raise ValueError(msg)
        x, y, z = (sp.simplify(self.integrale(c) / masse) for c in self.position)
        return x, y, z
