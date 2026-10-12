"""Le torseur d'un systeme de forces, et le changement de pole.

Les forces qui agissent sur un solide forment un **torseur**. Ses
coordonnees en un pole `O` sont sa resultante et son moment resultant :

    R = somme F_i,        M_O = somme OA_i ^ F_i.

Connaissant ces coordonnees en un pole, le moment en tout autre point se
deduit sans refaire la somme : `M_B = M_A + R ^ AB`. Deux torseurs sont
**equivalents** s'ils ont memes coordonnees en un pole ; un solide est en
equilibre si et seulement si le torseur des forces exterieures est
equivalent a zero.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
section 1.6.3, pages 16 a 18, relations (1.13) a (1.21).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.produit_vectoriel import produit_vectoriel
from polytech.mecanique.domain.statique.forces import (
    Force,
    de_l_espace,
    difference,
    moment,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur


def _somme(vecteurs: Iterable[Vecteur]) -> Vecteur:
    total: list[sp.Expr] = [sp.Integer(0)] * 3
    for vecteur in vecteurs:
        total = [sp.expand(t + v) for t, v in zip(total, vecteur, strict=True)]
    x, y, z = total
    return x, y, z


@dataclass(frozen=True, slots=True)
class Torseur:
    """Les coordonnees d'un torseur : resultante, et moment en un pole."""

    resultante: Vecteur
    moment: Vecteur
    pole: Vecteur

    @classmethod
    def des_forces(cls, forces: Sequence[Force], pole: Sequence[object]) -> Torseur:
        """Relations (1.13) et (1.14) : la somme des forces et de leurs moments."""
        if not forces:
            msg = "torseur sans force : il n'y a rien a sommer"
            raise ValueError(msg)
        return cls(
            resultante=_somme(force.vecteur for force in forces),
            moment=_somme(moment(force, pole) for force in forces),
            pole=de_l_espace(pole),
        )

    def en(self, pole: Sequence[object]) -> Torseur:
        """Le meme torseur en un autre pole : `M_B = M_A + R ^ AB`, (1.16)."""
        nouveau = de_l_espace(pole)
        transport = produit_vectoriel(self.resultante, difference(nouveau, self.pole))
        return Torseur(self.resultante, _somme((self.moment, transport)), nouveau)

    def est_equivalent_a(self, autre: Torseur) -> bool:
        """Memes coordonnees en un meme pole, relation (1.20)."""
        ramene = autre.en(self.pole)
        return self.resultante == ramene.resultante and self.moment == ramene.moment

    def est_nul(self) -> bool:
        """Resultante nulle et moment nul en un pole, relation (1.21)."""
        zero = (sp.Integer(0),) * 3
        return self.resultante == zero and self.moment == zero
