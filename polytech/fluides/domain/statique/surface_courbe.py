"""La poussee hydrostatique sur une surface courbe : la porte cylindrique.

Chapitre II des planches (exercice de la planche 19). Un cylindre de
rayon `R` repose au fond d'un bassin rempli a la hauteur `n` et s'articule
a mi-hauteur, cote eau. Son quart inferieur, entre la charniere et le
point bas, est mouille.

Sur une surface courbe, la poussee se decompose :

- sa composante **horizontale** est celle qui s'exerce sur la projection
  verticale de la surface, `rho g h_C A`, `h_C` etant la profondeur du
  centre de cette projection ;
- sa composante **verticale** est la poussee sur la projection
  horizontale, diminuee du poids du bloc de fluide compris entre la
  surface courbe et ses deux projections.

Chaque force elementaire etant normale au cylindre, la resultante passe
par son axe. Les grandeurs sont donnees par metre de longueur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique


@dataclass(frozen=True, slots=True)
class PorteCylindrique:
    """Le rayon, le niveau d'eau au-dessus du fond et le fluide."""

    rayon: float
    niveau: float
    masse_volumique: MasseVolumique
    pesanteur: float = PESANTEUR

    def __post_init__(self) -> None:
        """Refuse un cylindre sans rayon, ou dont la charniere emerge."""
        if self.rayon <= 0:
            msg = f"rayon nul ou negatif : {self.rayon} m"
            raise ValueError(msg)
        if self.niveau < self.rayon:
            msg = f"niveau inferieur au rayon : {self.niveau} m pour {self.rayon} m"
            raise ValueError(msg)

    def _poids_volumique(self) -> float:
        return self.masse_volumique.kg_par_m3 * self.pesanteur

    def poussee_horizontale(self) -> float:
        """`rho g (n - R / 2) R`, en newtons par metre."""
        centre = self.niveau - self.rayon / 2
        return self._poids_volumique() * centre * self.rayon

    def poussee_verticale(self) -> float:
        """`rho g [n R - R^2 (1 - pi / 4)]`, vers le haut, en newtons par metre."""
        bloc = self.rayon**2 * (1 - math.pi / 4)
        return self._poids_volumique() * (self.niveau * self.rayon - bloc)

    def resultante(self) -> float:
        """La grandeur de la poussee, en newtons par metre."""
        return math.hypot(self.poussee_horizontale(), self.poussee_verticale())

    def inclinaison(self) -> float:
        """L'angle de la poussee sur l'horizontale, en radians."""
        return math.atan2(self.poussee_verticale(), self.poussee_horizontale())

    def poids_a_l_ouverture(self) -> float:
        """Le poids du cylindre que la poussee equilibre autour de la charniere.

        La resultante passe par l'axe, a la distance `R` de la charniere :
        `F_R sin(theta) R = W R`.
        """
        return self.resultante() * math.sin(self.inclinaison())
