"""Une vitre selective, et le rayonnement entre deux plans gris.

**La vitre.** Ses proprietes spectrales sont en marche d'escalier : sous
la coupure, elle transmet `tau_c` et reflechit `rho` ; au-dessus, elle ne
transmet plus rien et absorbe tout ce qu'elle ne reflechit pas. Ses
proprietes globales se ponderent donc par la fraction d'energie que la
source emet sous la coupure, `k(lambda_c T)` :

    tau(T)   = tau_c k
    alpha(T) = (1 - rho - tau_c) k + (1 - rho)(1 - k)

Au soleil (6000 K), presque tout est emis sous la coupure : la vitre
transmet. A la temperature d'un capteur (300 a 350 K), presque rien : la
vitre absorbe, donc emet comme un corps presque noir. C'est l'effet de
serre, et c'est tout le principe du capteur.

**Deux plans gris.** Le flux qu'ils echangent, rapporte a leur ecart de
temperature, donne un coefficient `K_ray` qui s'additionne a celui de la
convection dans un schema en resistances.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.thermique.domain.rayonnement.corps_noir import (
    STEFAN_BOLTZMANN,
    fraction_emise_sous,
)


@dataclass(frozen=True, slots=True)
class VitreSelective:
    """La marche spectrale : transmission sous la coupure, reflexion partout."""

    transmission_courte: float
    reflexion: float
    coupure: float

    def __post_init__(self) -> None:
        """Refuse une transmission que la reflexion ne laisse pas."""
        if self.transmission_courte + self.reflexion > 1:
            msg = (
                f"transmission {self.transmission_courte} plus reflexion "
                f"{self.reflexion} depasse un"
            )
            raise ValueError(msg)

    def _fraction(self, temperature: float) -> float:
        return fraction_emise_sous(self.coupure, temperature)

    def transmission(self, temperature: float) -> float:
        """La transmission globale pour une source a cette temperature."""
        return self.transmission_courte * self._fraction(temperature)

    def absorption(self, temperature: float) -> float:
        """L'absorption globale, egale a l'emissivite a cette temperature."""
        k = self._fraction(temperature)
        courte = 1 - self.reflexion - self.transmission_courte
        return courte * k + (1 - self.reflexion) * (1 - k)


def coefficient_entre_plans_gris(
    premiere_emissivite: float,
    seconde_emissivite: float,
    premiere_temperature: float,
    seconde_temperature: float,
) -> float:
    """`sigma (T1^4 - T2^4) / ((1/eps1 + 1/eps2 - 1)(T1 - T2))`, par m2 et par K."""
    facteur = 1 / premiere_emissivite + 1 / seconde_emissivite - 1
    if premiere_temperature == seconde_temperature:
        return 4 * STEFAN_BOLTZMANN * premiere_temperature**3 / facteur
    ecart_des_puissances = premiere_temperature**4 - seconde_temperature**4
    ecart = premiere_temperature - seconde_temperature
    return STEFAN_BOLTZMANN * ecart_des_puissances / (facteur * ecart)
