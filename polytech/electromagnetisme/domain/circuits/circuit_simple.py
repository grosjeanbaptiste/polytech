"""Le circuit electrique simple : une source, une ligne, un recepteur.

Le syllabus d'electricite generale (13.15). Un tube de courant ferme sur
lui-meme, sans ramification : la source 1-2 (`E_s`, resistance interne
`R_s`), les deux conducteurs de liaison (`R'_l` et `R''_l`), le recepteur
3-4 (`E_r`, `R_r`). Le courant vaut
`I = (E_s - E_r) / (R_s + R'_l + R''_l + R_r)` (13.15.1).

La tension aux bornes de la source est sa force electromotrice diminuee
de la chute de tension interne (13.15.2) ; celle du recepteur s'en deduit
par la chute de tension en ligne (13.15.5). Le bilan (13.15.10) : la
puissance convertie par le recepteur est celle de la force electromotrice
de la source, moins les trois effets Joule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.circuits.troncon import Troncon


@dataclass(frozen=True, slots=True)
class CircuitSimple:
    """La source, le recepteur et les deux resistances de la ligne."""

    source: Troncon
    recepteur: Troncon
    ligne: tuple[float, float]

    def __post_init__(self) -> None:
        """Refuse un recepteur plus fort que la source, et un circuit sans frein."""
        if self.recepteur.fem > self.source.fem:
            msg = (
                f"roles inverses : E_r = {self.recepteur.fem} V depasse "
                f"E_s = {self.source.fem} V"
            )
            raise ValueError(msg)
        if self._resistance() <= 0:
            msg = "circuit sans resistance : le courant n'y serait pas borne"
            raise ValueError(msg)

    def _resistance(self) -> float:
        return self.source.resistance + sum(self.ligne) + self.recepteur.resistance

    def courant(self) -> float:
        """`I = (E_s - E_r) / (R_s + R'_l + R''_l + R_r)` (13.15.1)."""
        return (self.source.fem - self.recepteur.fem) / self._resistance()

    def tension_de_la_source(self) -> float:
        """`U_21 = E_s - R_s I` (13.15.2) : la tension en charge."""
        return self.source.tension(-self.courant())

    def chute_en_ligne(self) -> float:
        """`(R'_l + R''_l) I`."""
        return sum(self.ligne) * self.courant()

    def tension_du_recepteur(self) -> float:
        """`U_34 = U_21 - (R'_l + R''_l) I` (13.15.5)."""
        return self.tension_de_la_source() - self.chute_en_ligne()

    def puissance_fournie(self) -> float:
        """`P'_12 = U_21 I = E_s I - R_s I^2` (13.15.4)."""
        return self.tension_de_la_source() * self.courant()

    def puissance_de_la_ligne(self) -> float:
        """`(R'_l + R''_l) I^2` (13.15.6)."""
        return self.chute_en_ligne() * self.courant()

    def puissance_du_recepteur(self) -> float:
        """`P_34 = U_34 I = E_r I + R_r I^2` (13.15.8)."""
        return self.tension_du_recepteur() * self.courant()

    def puissance_utile(self) -> float:
        """`E_r I = U_34 I - R_r I^2` (13.15.9)."""
        return self.recepteur.puissance_convertie(self.courant())
