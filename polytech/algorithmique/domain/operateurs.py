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
