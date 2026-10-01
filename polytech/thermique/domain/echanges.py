"""Les echanges de surface, et la resistance qu'ils opposent au flux.

Le coefficient d'echange global regroupe convection et rayonnement : le
corrige officiel du TP 1 l'ecrit `K_glob = K_cv + K_ray`, et l'enonce ne
donne que la somme. La bibliotheque ne separe donc pas ce que l'enonce ne
separe pas -- inventer la part radiative serait inventer une donnee.

Un coefficient **nul** est licite et signifie « adiabatique ». C'est tout
l'interet : le cas (a) du TP n'est pas un autre probleme, c'est K = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.thermique.domain.validation import (
    exiger_fini,
    exiger_strictement_positif,
)


@dataclass(frozen=True, slots=True)
class CoefficientDEchange:
    """Coefficient d'echange global de surface, en W/(m^2 K)."""

    watts_par_m2_k: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_fini(self.watts_par_m2_k, "coefficient d'echange")
        if self.watts_par_m2_k < 0:
            msg = f"coefficient d'echange negatif interdit : {self.watts_par_m2_k}"
            raise ValueError(msg)

    @property
    def est_adiabatique(self) -> bool:
        """Vrai si cette paroi ne laisse passer aucun flux."""
        return self.watts_par_m2_k == 0.0


@dataclass(frozen=True, slots=True)
class DensiteDeFlux:
    """Densite de flux de chaleur, en W/m^2 : le signe en dit le sens."""

    watts_par_m2: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_fini(self.watts_par_m2, "densite de flux")


@dataclass(frozen=True, slots=True)
class ResistanceThermique:
    """Resistance au transfert, en K/W : l'ecart qu'il faut par watt."""

    kelvins_par_watt: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_strictement_positif(self.kelvins_par_watt, "resistance thermique")

    def en_serie_avec(self, autre: ResistanceThermique) -> ResistanceThermique:
        """Deux resistances traversees par le meme flux s'additionnent."""
        return ResistanceThermique(
            kelvins_par_watt=self.kelvins_par_watt + autre.kelvins_par_watt,
        )
