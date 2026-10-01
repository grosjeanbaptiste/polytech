"""Thevenin et Norton : deux ecritures du meme dipole.

Ancrage direct sur l'exercice 1 du fascicule, dont la transcription donne
tous les chiffres : une source de `110 V` en serie avec `R = 1 ohm`, le
tout en parallele avec une source de courant de `10 A`. A vide la tension
vaut `120 V` -- et le detail qui fait l'exercice est que le courant de la
source traverse `R` **en sens oppose**, d'ou un `+10` et non un `-10`.

Les deux representations sont equivalentes, et la bibliotheque le verifie
plutot que de l'affirmer : convertir dans un sens puis dans l'autre doit
rendre le dipole de depart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Thevenin:
    """Un dipole vu comme une source de tension en serie avec une resistance."""

    tension_a_vide: float
    resistance: float

    def __post_init__(self) -> None:
        """Refuse une resistance nulle : la conversion en Norton diviserait par zero."""
        if self.resistance <= 0:
            msg = (
                f"resistance de Thevenin nulle ou negative : {self.resistance} ohm "
                f"-- le dipole n'a pas d'equivalent de Norton"
            )
            raise ValueError(msg)

    def courant_de_court_circuit(self) -> float:
        """`I_cc = U_vide / R`."""
        return self.tension_a_vide / self.resistance

    def en_norton(self) -> Norton:
        """L'ecriture duale, meme resistance."""
        return Norton(
            courant_de_court_circuit=self.courant_de_court_circuit(),
            resistance=self.resistance,
        )

    def tension_sur(self, charge: float) -> float:
        """La tension aux bornes d'une charge resistive donnee."""
        if charge < 0:
            msg = f"resistance de charge negative : {charge} ohm"
            raise ValueError(msg)
        return self.tension_a_vide * charge / (self.resistance + charge)


@dataclass(frozen=True, slots=True)
class Norton:
    """Un dipole vu comme une source de courant en parallele avec une resistance."""

    courant_de_court_circuit: float
    resistance: float

    def __post_init__(self) -> None:
        """Refuse une resistance nulle."""
        if self.resistance <= 0:
            msg = f"resistance de Norton nulle ou negative : {self.resistance} ohm"
            raise ValueError(msg)

    def tension_a_vide(self) -> float:
        """`U_vide = I_cc R`."""
        return self.courant_de_court_circuit * self.resistance

    def en_thevenin(self) -> Thevenin:
        """L'ecriture duale, meme resistance."""
        return Thevenin(
            tension_a_vide=self.tension_a_vide(),
            resistance=self.resistance,
        )


def source_de_tension_et_source_de_courant(
    tension: float,
    resistance: float,
    courant: float,
) -> Thevenin:
    """Le dipole de l'exercice 1 : les deux sources reunies en un Thevenin.

    A vide, le courant de la source de courant n'a que la branche `R` pour
    passer, et il la traverse en sens oppose : la tension a vide vaut donc
    `U + R I` et non `U - R I`. C'est tout l'enonce.
    """
    return Thevenin(
        tension_a_vide=tension + resistance * courant,
        resistance=resistance,
    )
