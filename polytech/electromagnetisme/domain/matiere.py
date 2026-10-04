"""Paragraphes 1.3 et 1.4 du syllabus : les charges que porte la matiere.

Un echantillon de metal contient `N = Z rho V N_A / M` protons, et autant
d'electrons : sa charge positive `Q+ = N e` et sa charge negative `Q-`
se compensent exactement. Si chaque atome perd `n` electrons, la charge
mobile vaut `Delta Q = -n Q+ / Z` -- petite devant `Q+`, et c'est elle
seule que l'electricite met en mouvement.

Le compte des atomes n'est pas de l'electricite : il vient de la chimie
(`polytech.chemistry`), qui porte l'atome, la masse molaire et le nombre
d'Avogadro. Ce module y ajoute la seule chose qui soit electrique, la
charge elementaire.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
exemples 1.3.2 et 1.4.1, pages 1-4 a 1-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.chemistry.domain.matiere import nombre_d_entites, quantite_de_matiere
from polytech.electromagnetisme.domain.constantes import CHARGE_ELEMENTAIRE

if TYPE_CHECKING:
    from polytech.chemistry.domain.atomes import Atome


@dataclass(frozen=True, slots=True)
class Echantillon:
    """Un volume d'un corps pur : son atome, sa masse volumique, sa masse molaire."""

    atome: Atome
    masse_volumique: float
    masse_molaire: float
    volume: float

    def __post_init__(self) -> None:
        """Refuse un echantillon qui ne contiendrait aucune matiere."""
        if self.masse_volumique <= 0 or self.volume <= 0:
            msg = (
                "echantillon sans matiere : masse volumique "
                f"{self.masse_volumique}, volume {self.volume}"
            )
            raise ValueError(msg)

    @property
    def atomes(self) -> float:
        """`rho V N_A / M` : le nombre d'atomes de l'echantillon."""
        masse = self.masse_volumique * self.volume
        return nombre_d_entites(quantite_de_matiere(masse, self.masse_molaire))

    @property
    def protons(self) -> float:
        """`N = Z` fois le nombre d'atomes."""
        return self.atome.protons * self.atomes

    @property
    def charge_positive(self) -> float:
        """`Q+ = N e`, en coulombs."""
        return self.protons * CHARGE_ELEMENTAIRE

    @property
    def charge_negative(self) -> float:
        """`Q- = -Q+` : les atomes sont neutres."""
        return -self.atome.electrons * self.atomes * CHARGE_ELEMENTAIRE

    def charge_mobile(self, electrons_perdus: int) -> float:
        """`Delta Q = -n Q+ / Z`, si chaque atome perd `n` electrons."""
        if not 0 <= electrons_perdus <= self.atome.electrons:
            msg = (
                f"un atome de {self.atome.symbole} ne perd pas plus d'electrons "
                f"qu'il n'en a : {electrons_perdus} pour {self.atome.electrons}"
            )
            raise ValueError(msg)
        return -electrons_perdus * self.atomes * CHARGE_ELEMENTAIRE
