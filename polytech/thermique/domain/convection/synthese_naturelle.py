"""Ce que la forme `Nu = C (Gr Pr)^n` dit du coefficient de convection naturelle.

Planche de synthese du jeu 10 (planche 9). Les correlations de
convection naturelle prennent cette forme, avec `n` entre un quart et un
tiers. Comme `Gr = beta g theta_0 l^3 / nu^2` et `Nu = K l / lambda` :

* `K ~ theta_0^n` : le coefficient depend de l'ecart de temperature, et
  la relation entre le flux et cet ecart n'est donc **pas** lineaire,
  `phi ~ Dt^(1 + n)` -- la loi de Newton ne vaut pas ;
* `K ~ l^(3n - 1)` : il decroit quand la dimension augmente, et n'en
  depend plus du tout quand `n` vaut un tiers.

`K` depend aussi de `g` : sans pesanteur, pas de convection naturelle.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPOSANTS = (0.25, 1.0 / 3.0)
"""« n ~ 0,25 ... 1/3 » : du laminaire au turbulent."""


def _exiger_un_rapport(rapport: float) -> None:
    if rapport <= 0:
        msg = f"rapport nul ou negatif : {rapport}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CorrelationNaturelle:
    """`Nu = C (Gr Pr)^n`, pour une geometrie fixee."""

    coefficient: float
    exposant: float

    def __post_init__(self) -> None:
        """La planche borne `n` : c'est ce qui donne `3n - 1 <= 0`."""
        if not EXPOSANTS[0] <= self.exposant <= EXPOSANTS[1]:
            msg = f"exposant hors de [1/4, 1/3] : {self.exposant}"
            raise ValueError(msg)

    def nusselt(self, rayleigh: float) -> float:
        """`C (Gr Pr)^n`."""
        return float(self.coefficient * rayleigh**self.exposant)

    def effet_de_l_ecart(self, rapport: float) -> float:
        """Ce par quoi `K` est multiplie quand `theta_0` l'est par `rapport`."""
        _exiger_un_rapport(rapport)
        return float(rapport**self.exposant)

    def effet_sur_le_flux(self, rapport: float) -> float:
        """Ce par quoi `phi` est multiplie : `rapport^(1 + n)`."""
        return rapport * self.effet_de_l_ecart(rapport)

    def effet_de_la_dimension(self, rapport: float) -> float:
        """Ce par quoi `K` est multiplie quand `l` l'est par `rapport`."""
        _exiger_un_rapport(rapport)
        return float(rapport ** (3 * self.exposant - 1))
