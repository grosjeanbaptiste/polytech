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
