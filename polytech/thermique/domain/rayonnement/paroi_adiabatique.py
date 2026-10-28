"""Trois surfaces dont une adiabatique : l'echange entre les deux autres.

Le cas du four du TP 2 : des resistances (1), une tole (2), des parois
isolees (3). Une paroi adiabatique ne garde rien ; elle reradie tout ce
qu'elle recoit et sert de **relais**. Dans le schema en resistances, elle
ajoute au chemin direct `1/S1S2` un chemin en parallele par elle-meme,
`1/S1S3 + 1/S2S3`. L'echange entre corps noirs vaut donc

    S1S2_noir = S1S2 + 1 / (1/S1S3 + 1/S2S3),

et les deux resistances de surface `rho/(eps S)` s'y ajoutent en serie.

Les radiosites `J1`, `J2` se lisent ensuite sur le schema, et l'emittance
de la paroi est leur moyenne ponderee par ses deux aires d'echange : c'est
le bilan nul au noeud qu'elle forme.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.thermique.domain.rayonnement.corps_noir import (
    STEFAN_BOLTZMANN,
    emittance,
)


def _resistance_de_surface(aire: float, emissivite: float) -> float:
    """`rho / (eps S)`, nulle pour une surface noire ; leve hors de ]0, 1]."""
    if not 0 < emissivite <= 1:
        msg = f"emissivite hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)
    return (1 - emissivite) / (emissivite * aire)


@dataclass(frozen=True, slots=True)
class EnceinteAParoiAdiabatique:
    """Deux surfaces grises actives, et les aires d'echange de l'enceinte."""

    aire_1: float
    emissivite_1: float
    temperature_1: float
    aire_2: float
    emissivite_2: float
    temperature_2: float
    echange_12: float
    echange_13: float
    echange_23: float

    def echange_noir(self) -> float:
        """L'echange direct, plus le relais par la paroi s'il existe."""
        if self.echange_13 <= 0 or self.echange_23 <= 0:
            return self.echange_12
        return self.echange_12 + 1 / (1 / self.echange_13 + 1 / self.echange_23)

    def _resistances(self) -> tuple[float, float]:
        return (
            _resistance_de_surface(self.aire_1, self.emissivite_1),
            _resistance_de_surface(self.aire_2, self.emissivite_2),
        )

    def echange_gris(self) -> float:
        """`1 / (rho1/eps1S1 + rho2/eps2S2 + 1/S1S2_noir)`."""
        premiere, seconde = self._resistances()
        return 1 / (premiere + seconde + 1 / self.echange_noir())

    def flux(self) -> float:
        """Le flux net de 1 vers 2, par unite de longueur si les aires le sont."""
        ecart = emittance(self.temperature_1) - emittance(self.temperature_2)
        return self.echange_gris() * ecart

    def radiosites(self) -> tuple[float, float]:
        """`J1` et `J2`, lues sur les deux resistances de surface."""
        premiere, seconde = self._resistances()
        flux = self.flux()
        return (
            emittance(self.temperature_1) - flux * premiere,
            emittance(self.temperature_2) + flux * seconde,
        )

    def flux_pour_la_radiosite_2(self, radiosite_2: float) -> float:
        """`(J1 - J2) S1S2_noir` : le flux que le schema impose pour ce `J2`."""
        j1, _ = self.radiosites()
        return (j1 - radiosite_2) * self.echange_noir()

    def temperature_de_la_paroi(self) -> float:
        """`sigma T3^4 = (S1S3 J1 + S2S3 J2) / (S1S3 + S2S3)` ; leve sans paroi."""
        total = self.echange_13 + self.echange_23
        if total <= 0:
            msg = "aucune paroi : ses deux aires d'echange sont nulles"
            raise ValueError(msg)
        premiere, seconde = self.radiosites()
        moyenne = (self.echange_13 * premiere + self.echange_23 * seconde) / total
        return float((moyenne / STEFAN_BOLTZMANN) ** 0.25)
