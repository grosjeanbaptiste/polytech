"""Les proprietes que l'enonce donne : conduire, generer, s'evaporer.

Trois d'entre elles sont des denominateurs quelque part dans le TP 1 : la
conductivite divise le flux pour donner un gradient, l'epaisseur divise
une resistance, la chaleur latente divise un flux pour donner un debit.
Un zero y est une division par zero deguisee, pas une donnee.

La generation volumique, elle, peut etre nulle : le reservoir d'hydrogene
n'a aucune source interne, et `q = 0` y est la donnee de l'enonce.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.thermique.domain.validation import (
    exiger_fini,
    exiger_strictement_positif,
)


@dataclass(frozen=True, slots=True)
class Conductivite:
    """Conductivite thermique, en W/(m K) -- le lambda de la loi de Fourier."""

    watts_par_m_k: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_strictement_positif(self.watts_par_m_k, "conductivite")


@dataclass(frozen=True, slots=True)
class Epaisseur:
    """Epaisseur traversee par le flux, en metres."""

    metres: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_strictement_positif(self.metres, "epaisseur")


@dataclass(frozen=True, slots=True)
class ChaleurLatente:
    """Chaleur latente de vaporisation, en J/kg."""

    joules_par_kg: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_strictement_positif(self.joules_par_kg, "chaleur latente")


@dataclass(frozen=True, slots=True)
class GenerationVolumique:
    """Puissance degagee par unite de volume, en W/m^3.

    Nulle est licite : c'est le cas du reservoir d'hydrogene.
    """

    watts_par_m3: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_fini(self.watts_par_m3, "generation volumique")
        if self.watts_par_m3 < 0:
            msg = f"generation volumique negative interdite : {self.watts_par_m3}"
            raise ValueError(msg)
