"""Objets-valeur du cours II.1 : pression, masse volumique, hauteur.

Ce sont des *value objects* au sens DDD : immuables, sans identite, egaux par
leurs attributs. Toute construction invalide leve -- aucune valeur par defaut
ne vient masquer une donnee aberrante.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

PESANTEUR: Final = 9.81
"""Acceleration de la pesanteur retenue par le cours, en m/s^2."""


def _exiger_fini(valeur: float, grandeur: str) -> None:
    if not math.isfinite(valeur):
        msg = f"{grandeur} : valeur non finie ({valeur})"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Pression:
    """Pression absolue, en pascals (cours II.1)."""

    pascals: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        _exiger_fini(self.pascals, "pression")
        if self.pascals < 0:
            msg = f"pression absolue negative interdite : {self.pascals} Pa"
            raise ValueError(msg)

    @property
    def kilopascals(self) -> float:
        """Meme pression exprimee en kPa."""
        return self.pascals / 1_000.0

    def relative_a(self, reference: Pression) -> Ecart:
        """Pression relative : ecart a une reference (cours II.1)."""
        return Ecart(pascals=self.pascals - reference.pascals)


@dataclass(frozen=True, slots=True)
class Ecart:
    """Difference de pression, en pascals : peut etre negative."""

    pascals: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        _exiger_fini(self.pascals, "ecart de pression")

    @property
    def kilopascals(self) -> float:
        """Meme ecart exprime en kPa."""
        return self.pascals / 1_000.0


@dataclass(frozen=True, slots=True)
class MasseVolumique:
    """Masse volumique rho, en kg/m^3.

    Le cours distingue la masse volumique (dimensionnee) de la densite
    relative (sans dimension) : voir :meth:`densite_relative`.
    """

    kg_par_m3: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        _exiger_fini(self.kg_par_m3, "masse volumique")
        if self.kg_par_m3 <= 0:
            msg = f"masse volumique non strictement positive : {self.kg_par_m3}"
            raise ValueError(msg)

    def densite_relative(self, reference: MasseVolumique) -> float:
        """Densite relative, sans dimension."""
        return self.kg_par_m3 / reference.kg_par_m3


@dataclass(frozen=True, slots=True)
class Hauteur:
    """Hauteur de colonne, en metres, comptee positivement."""

    metres: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        _exiger_fini(self.metres, "hauteur")
        if self.metres < 0:
            msg = f"hauteur negative interdite : {self.metres} m"
            raise ValueError(msg)
