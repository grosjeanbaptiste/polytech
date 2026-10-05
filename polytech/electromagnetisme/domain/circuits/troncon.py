"""Un troncon de circuit : sa force electromotrice et sa resistance.

Le syllabus d'electricite generale (13.8, 13.11, 13.13 et 13.14). Entre
ses bornes 1 et 2, un troncon de tube de courant suit la loi d'Ohm
integree `U_12 = E_21 + R_12 I_12` (13.8.10) ; a vide, `U_12 = E_21`
(13.8.15). C'est l'ecriture `U = E + R I` (13.14.1a), en convention
recepteur.

Il **recoit** la puissance electrique `P_12 = U_12 I_12` (13.13.4), qu'il
convertit pour une part sous forme chimique, thermique ou mecanique,
`P_e = E_21 I_12` (13.13.1), et dissipe pour l'autre par effet Joule,
`P_j = R_12 I_12^2` (13.13.2). Si `U_12` depasse `E_21`, le courant va de
1 vers 2 et le troncon est un recepteur ; sinon il est une source, et sa
puissance recue est negative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Troncon:
    """Le troncon 1-2, par `E_21` en volts et `R_12` en ohms."""

    fem: float
    resistance: float

    def __post_init__(self) -> None:
        """Refuse une resistance negative ; nulle, elle est un fil ideal."""
        if self.resistance < 0:
            msg = f"resistance negative : {self.resistance} ohm"
            raise ValueError(msg)

    def tension(self, courant: float) -> float:
        """`U_12 = E_21 + R_12 I_12` (13.8.10)."""
        return self.fem + self.resistance * courant

    def courant(self, tension: float) -> float:
        """`I_12 = (U_12 - E_21) / R_12`."""
        if self.resistance == 0:
            msg = "troncon sans resistance : sa tension ne fixe pas son courant"
            raise ValueError(msg)
        return (tension - self.fem) / self.resistance

    def puissance_convertie(self, courant: float) -> float:
        """`P_e = E_21 I_12` (13.13.1) : sous forme non electrique, hors Joule."""
        return self.fem * courant

    def puissance_de_joule(self, courant: float) -> float:
        """`P_j = R_12 I_12^2` (13.13.2)."""
        return self.resistance * courant**2

    def puissance_recue(self, courant: float) -> float:
        """`P_12 = U_12 I_12` (13.13.4)."""
        return self.tension(courant) * courant

    def est_recepteur(self, courant: float) -> bool:
        """Vrai si la puissance electrique recue est positive."""
        return self.puissance_recue(courant) > 0
