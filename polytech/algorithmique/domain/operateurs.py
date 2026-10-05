"""La surcharge des operateurs sur un point du plan.

Le cours (seance 4, « Surcharge des operateurs ») : « et si A et B etaient
des objets ? » -- la surcharge redefinit `==`, `+`, `-` pour un type du
programme. Ecrire `a == b` revient a appeler `a.operator==(b)`.

Le point des planches en est l'exemple : deux points sont egaux quand
leurs deux coordonnees le sont, leur somme et leur difference se font
coordonnee par coordonnee. Une recherche dans un tableau de points n'a
alors plus a comparer les champs un a un.

La meme classe porte le milieu et la distance de la planche
« Encapsulation » : `M = ((x1 + x2)/2, (y1 + y2)/2)` et
`d = sqrt(dx^2 + dy^2)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.algorithmique.domain.flux_de_sortie import ecrire_un_reel


@dataclass(frozen=True, slots=True)
class Point:
    """Un point du plan ; l'egalite est celle de ses deux coordonnees."""

    x: float
    y: float

    def __add__(self, autre: Point) -> Point:
        """`operator+` : la somme coordonnee par coordonnee."""
        return Point(self.x + autre.x, self.y + autre.y)

    def __sub__(self, autre: Point) -> Point:
        """`operator-` : la difference coordonnee par coordonnee."""
        return Point(self.x - autre.x, self.y - autre.y)

    def milieu(self, autre: Point) -> Point:
        """Le milieu du segment qui joint les deux points."""
        return Point((self.x + autre.x) / 2, (self.y + autre.y) / 2)

    def distance(self, autre: Point) -> float:
        """La longueur du segment qui joint les deux points."""
        return math.hypot(self.x - autre.x, self.y - autre.y)


@dataclass(frozen=True, slots=True)
class Vecteur3d:
    """Un vecteur a trois composantes, nulles par defaut."""

    x: float = 0
    y: float = 0
    z: float = 0

    def __add__(self, autre: Vecteur3d) -> Vecteur3d:
        """`operator+` : la somme composante par composante."""
        return Vecteur3d(self.x + autre.x, self.y + autre.y, self.z + autre.z)

    def __sub__(self, autre: Vecteur3d) -> Vecteur3d:
        """`operator-` : la difference composante par composante."""
        return Vecteur3d(self.x - autre.x, self.y - autre.y, self.z - autre.z)

    def __str__(self) -> str:
        """`< composante1, composante2, composante3>`, comme `affiche`."""
        composantes = (ecrire_un_reel(float(c)) for c in (self.x, self.y, self.z))
        return f"< {', '.join(composantes)}>"
