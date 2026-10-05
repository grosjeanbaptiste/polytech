"""Les varietes lineaires affines : le translate d'un sous-espace vectoriel.

Les notes de cours de 2011 (annexe 2, section D) : l'image d'un sous-espace
`W` par la translation de vecteur `Y` est la variete lineaire affine
`w_Y = {X + Y | X dans W}`. `W` est son **sous-espace directeur**, et sa
dimension est celle de `W` : une droite pour 1, un hyperplan pour `n - 1`.

* `w_Y` n'est un sous-espace que si elle contient le vecteur nul ;
* tout point de `w_Y` la definit aussi bien que `Y` ;
* deux varietes sont **paralleles** quand elles ont meme sous-espace
  directeur ; elles sont alors egales ou disjointes ;
* une intersection non vide de varietes est une variete, de directeur
  l'intersection des directeurs ;
* `k + 1` points engendrent une variete de dimension au plus `k`, et sont
  **affinement independants** quand elle vaut `k`.

L'ensemble des solutions de `A X = b` en est une, de directeur le noyau de
`A` : c'est la « forme generale » du syllabus d'algebre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import extraire_une_base
from polytech.maths.domain.algebre.rang_exact import rang_par_echelonnement
from polytech.maths.domain.algebre.resolution import par_echelonnement_simple
from polytech.maths.domain.algebre.systemes import SystemeLineaire

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.familles_de_vecteurs import Famille, Vecteur


@dataclass(frozen=True, slots=True)
class VarieteAffine:
    """Un point, et des generateurs du sous-espace directeur."""

    point: Vecteur
    directeur: Famille

    def __post_init__(self) -> None:
        """Refuse un generateur qui n'est pas de l'espace du point."""
        if any(len(vecteur) != len(self.point) for vecteur in self.directeur):
            msg = "un generateur du directeur n'est pas de la taille du point"
            raise ValueError(msg)

    @classmethod
    def des_solutions(cls, systeme: SystemeLineaire) -> VarieteAffine:
        """L'ensemble des solutions de `A X = b`, de directeur le noyau de `A`."""
        ensemble = par_echelonnement_simple(systeme)
        return cls(ensemble.particuliere, ensemble.directions)

    @classmethod
    def engendree_par(cls, points: Famille) -> VarieteAffine:
        """La plus petite variete qui contient les points."""
        if not points:
            msg = "aucun point : il n'y a pas de variete a engendrer"
            raise ValueError(msg)
        origine = sp.Matrix(list(points[0]))
        ecarts = [tuple(sp.Matrix(list(p)) - origine) for p in points[1:]]
        return cls(tuple(points[0]), tuple(ecarts))

    @staticmethod
    def sont_affinement_independants(points: Famille) -> bool:
        """`k + 1` points qui engendrent une variete de dimension `k`."""
        return VarieteAffine.engendree_par(points).dimension() == len(points) - 1

    def _colonnes(self) -> sp.Matrix:
        base = extraire_une_base(self.directeur)
        lignes = sp.Matrix([list(v) for v in base]) if base else sp.zeros(0, 0)
        return lignes.T if base else sp.zeros(len(self.point), 0)

    def dimension(self) -> int:
        """Celle du sous-espace directeur."""
        return int(self._colonnes().cols)

    def contient(self, vecteur: Vecteur) -> bool:
        """Vrai si `vecteur - point` est dans le sous-espace directeur."""
        ecart = sp.Matrix(list(vecteur)) - sp.Matrix(list(self.point))
        colonnes = self._colonnes()
        jointes = rang_par_echelonnement(colonnes.row_join(ecart))
        return jointes == rang_par_echelonnement(colonnes)

    def est_un_sous_espace(self) -> bool:
        """Vrai si elle contient le vecteur nul."""
        return self.contient((0,) * len(self.point))

    def est_parallele_a(self, autre: VarieteAffine) -> bool:
        """Meme sous-espace directeur."""
        miens, siens = self._colonnes(), autre._colonnes()
        ensemble = rang_par_echelonnement(miens.row_join(siens))
        return bool(miens.cols == ensemble == siens.cols)

    def est_egale_a(self, autre: VarieteAffine) -> bool:
        """Parallele, et passant par un point de l'autre."""
        return self.est_parallele_a(autre) and self.contient(autre.point)

    def est_une_droite(self) -> bool:
        """De dimension 1."""
        return self.dimension() == 1

    def est_un_hyperplan(self) -> bool:
        """De dimension `n - 1`."""
        return self.dimension() == len(self.point) - 1

    def intersection(self, autre: VarieteAffine) -> VarieteAffine | None:
        """La variete commune, ou `None` si les deux sont disjointes."""
        miens, siens = self._colonnes(), autre._colonnes()
        ecart = sp.Matrix(list(autre.point)) - sp.Matrix(list(self.point))
        jointes = miens.row_join(-siens)
        if rang_par_echelonnement(jointes.row_join(ecart)) != jointes.rank():
            return None
        commun = par_echelonnement_simple(SystemeLineaire(jointes, ecart))
        depart = sp.Matrix(list(self.point))
        point = depart + miens * sp.Matrix(list(commun.particuliere[: miens.cols]))
        directions = [
            tuple(miens * sp.Matrix(list(d[: miens.cols]))) for d in commun.directions
        ]
        return VarieteAffine(tuple(point), tuple(directions))
