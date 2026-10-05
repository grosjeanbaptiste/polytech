"""Ce que la forme `Nu = C Re^n Pr^m` dit du coefficient de convection forcee.

Planche de synthese du jeu 09 (planche 33). Les correlations de
convection forcee prennent presque toutes cette forme, avec `n` entre un
tiers et 0,8 et `m` voisin d'un tiers. Comme `Nu = K l / lambda` et
`Re = u_0 l / nu`, il en decoule, pour un fluide donne :

* `K ~ u_0^n` : le coefficient croit avec la vitesse, mais moins vite
  qu'elle, puisque `n < 1` ;
* `K ~ l^(n - 1)` : il decroit quand la dimension augmente.

La meme correlation s'ecrit `j_H = Nu / (Re Pr^1/3) = C Re^(n - 1)`.
"""

from __future__ import annotations

from dataclasses import dataclass


def _exiger_un_rapport(rapport: float) -> None:
    if rapport <= 0:
        msg = f"rapport nul ou negatif : {rapport}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Correlation:
    """`Nu = C Re^n Pr^m`, pour une geometrie fixee."""

    coefficient: float
    exposant: float
    exposant_de_prandtl: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        """La planche pose `n < 1` : c'est ce qui fixe le sens des deux effets."""
        if not 0 < self.exposant < 1:
            msg = f"exposant de Reynolds hors de ]0, 1[ : {self.exposant}"
            raise ValueError(msg)

    def nusselt(self, reynolds: float, prandtl: float) -> float:
        """`C Re^n Pr^m`."""
        regime = reynolds**self.exposant * prandtl**self.exposant_de_prandtl
        return float(self.coefficient * regime)

    def facteur_j(self, reynolds: float) -> float:
        """`j_H = C Re^(n - 1)`, quand `m` vaut un tiers."""
        return float(self.coefficient * reynolds ** (self.exposant - 1))

    def effet_de_la_vitesse(self, rapport: float) -> float:
        """Ce par quoi `K` est multiplie quand `u_0` l'est par `rapport`."""
        _exiger_un_rapport(rapport)
        return float(rapport**self.exposant)

    def effet_de_la_dimension(self, rapport: float) -> float:
        """Ce par quoi `K` est multiplie quand `l` l'est par `rapport`."""
        _exiger_un_rapport(rapport)
        return float(rapport ** (self.exposant - 1))
