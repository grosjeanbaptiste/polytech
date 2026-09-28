"""Du prototype normalise au filtre reel : mise a l'echelle et transposition.

Un prototype en echelle est tabule pour une coupure a 1 rad/s et des
terminaisons de 1 ohm. Deux operations en font un filtre reel : la mise a
l'echelle d'impedance, qui multiplie les inductances et divise les
capacites, puis la transposition passe-bas vers passe-haut, qui echange la
nature de chaque element.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrototypeNormalise:
    """Les elements normalises d'un prototype, et l'impedance visee.

    `elements` alterne inductance serie et capacite shunt, en commencant par
    une inductance : c'est la convention des tables du cours.
    """

    elements: tuple[float, ...]
    impedance: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if not self.elements:
            msg = "prototype vide : aucun element a mettre a l'echelle"
            raise ValueError(msg)
        if self.impedance <= 0:
            msg = f"impedance non positive : {self.impedance}"
            raise ValueError(msg)
        if any(g <= 0 for g in self.elements):
            msg = f"element non positif dans le prototype : {self.elements}"
            raise ValueError(msg)

    def en_passe_bas(self, pulsation_de_coupure: float) -> tuple[float, ...]:
        """Rend les elements du passe-bas, dans l'ordre du prototype.

        Ici la nature ne change pas : seule l'echelle bouge. Une inductance
        serie reste une inductance, une capacite shunt reste une capacite.
        Les rangs pairs sont donc des inductances, les rangs impairs des
        capacites -- l'inverse exact de :meth:`en_passe_haut`.
        """
        if pulsation_de_coupure <= 0:
            msg = f"pulsation de coupure non positive : {pulsation_de_coupure}"
            raise ValueError(msg)
        return tuple(
            g * self.impedance / pulsation_de_coupure
            if rang % 2 == 0
            else g / (pulsation_de_coupure * self.impedance)
            for rang, g in enumerate(self.elements)
        )

    def en_passe_haut(self, pulsation_de_coupure: float) -> tuple[float, ...]:
        """Rend les elements du passe-haut, dans l'ordre du prototype.

        La transposition echange la nature : une inductance serie devient une
        capacite serie, une capacite shunt devient une inductance shunt. Le
        premier element etant une inductance, les rangs pairs donnent des
        capacites et les rangs impairs des inductances.
        """
        if pulsation_de_coupure <= 0:
            msg = f"pulsation de coupure non positive : {pulsation_de_coupure}"
            raise ValueError(msg)
        return tuple(
            1 / (pulsation_de_coupure * self.impedance * g)
            if rang % 2 == 0
            else self.impedance / (pulsation_de_coupure * g)
            for rang, g in enumerate(self.elements)
        )


@dataclass(frozen=True, slots=True)
class PrototypeButterworth(PrototypeNormalise):
    """Un prototype de Butterworth : plat dans la bande, sans ondulation."""


@dataclass(frozen=True, slots=True)
class PrototypeDeChebyshev(PrototypeNormalise):
    """Un prototype de Chebyshev : plus raide, au prix d'une ondulation.

    Sa table depend de l'ondulation toleree dans la bande passante ; la
    mise a l'echelle et la transposition, elles, ne s'en soucient pas --
    c'est pourquoi les deux prototypes partagent le meme parent.
    """
