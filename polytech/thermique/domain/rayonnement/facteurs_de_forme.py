"""Les facteurs d'angle, et les deux lois qui les contraignent.

Le formulaire ne les donne qu'en abaques : trois pages de courbes pour les
rectangles paralleles et perpendiculaires, les ecrans de tubes, les
rangees. Une bibliotheque ne peut pas lire un abaque -- mais elle peut
verifier que les valeurs **lues** sont coherentes, et c'est la tout ce
qu'il est honnete de lui demander ici.

Deux lois suffisent a cela. La **reciprocite**, `S_i F_ij = S_j F_ji`, et
la **sommation**, `somme_j F_ij = 1` pour une enceinte fermee. Un facteur
mal lu sur un abaque viole presque toujours l'une des deux, et c'est le
seul garde-fou disponible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TOLERANCE = 1e-9
"""Les facteurs viennent d'abaques : l'egalite exacte n'a pas de sens."""


@dataclass(frozen=True, slots=True)
class Enceinte:
    """Les surfaces d'une enceinte et les facteurs lus entre elles."""

    surfaces: tuple[float, ...]
    facteurs: dict[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse une surface nulle et un facteur hors de [0, 1]."""
        for aire in self.surfaces:
            if aire <= 0:
                msg = f"surface nulle ou negative interdite : {aire} m^2"
                raise ValueError(msg)
        hors = {c: f for c, f in self.facteurs.items() if not 0 <= f <= 1}
        if hors:
            msg = f"facteur d'angle hors de [0, 1] : {hors}"
            raise ValueError(msg)

    def reciproque(
        self, depart: int, arrivee: int, tolerance: float = TOLERANCE
    ) -> bool:
        """Vrai si `S_i F_ij = S_j F_ji` pour ce couple."""
        aller = self.surfaces[depart] * self.facteurs[depart, arrivee]
        retour = self.surfaces[arrivee] * self.facteurs[arrivee, depart]
        return abs(aller - retour) <= tolerance * max(abs(aller), 1.0)

    def facteur_deduit(self, depart: int, arrivee: int) -> float:
        """`F_ji` deduit de `F_ij` par reciprocite, pour eviter une lecture."""
        return (
            self.surfaces[arrivee]
            * self.facteurs[arrivee, depart]
            / self.surfaces[depart]
        )

    def somme_depuis(self, depart: int) -> float:
        """`somme_j F_ij`, qui doit valoir un sur une enceinte fermee."""
        return sum(f for (i, _), f in self.facteurs.items() if i == depart)

    def est_fermee_depuis(self, depart: int, tolerance: float = TOLERANCE) -> bool:
        """Vrai si la sommation est satisfaite depuis cette surface."""
        return abs(self.somme_depuis(depart) - 1.0) <= tolerance


def facteur_entre_deux_plaques_paralleles(cote: float, distance: float) -> float:
    """Le facteur d'angle entre deux carres identiques, face a face.

    Une des « expressions analytiques » que le fascicule donne en exemple,
    et la plus utile : elle tend vers **un** quand les plaques se
    rapprochent -- chacune ne voit plus que l'autre -- ce qui est
    exactement l'hypothese du cas tabule « deux plans gris infinis
    paralleles ».
    """
    if distance <= 0:
        msg = f"distance nulle ou negative : {distance} m"
        raise ValueError(msg)
    if cote <= 0:
        msg = f"cote nul ou negatif : {cote} m"
        raise ValueError(msg)
    rapport = cote / distance
    carre = rapport**2
    racine = math.sqrt(1 + carre)
    logarithme = math.log(math.sqrt((1 + carre) ** 2 / (1 + 2 * carre)))
    arcs = rapport * racine * math.atan(rapport / racine) - rapport * math.atan(rapport)
    return min(1.0, (2 / (math.pi * carre)) * (logarithme + 2 * arcs))


def aire_d_echange_direct(aire: float, facteur: float) -> float:
    """`S_i F_ij`, en metres carres : l'aire d'echange direct.

    Le fascicule emploie cette forme pour les echanges totaux parce
    qu'elle rend la **reciprocite** visible : `S_i F_ij = S_j F_ji`, donc
    l'aire d'echange direct est symetrique alors que le facteur d'angle ne
    l'est pas.
    """
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    if not 0 <= facteur <= 1:
        msg = f"facteur d'angle hors de [0, 1] : {facteur}"
        raise ValueError(msg)
    return aire * facteur
