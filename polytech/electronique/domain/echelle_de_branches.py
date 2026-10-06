"""Une echelle dont chaque barreau est une branche, entre source et charge.

Apres une transformation de frequence, un element du prototype devient
une *branche* : une inductance, une capacite, ou les deux
(`conversions_d_elements`). L'echelle se resout alors comme celle du
prototype, par ses matrices de chaine. Les places alternent -- bras,
derivation, bras -- et seule la premiere est a dire.

C'est ce qui permet de verifier un filtre dimensionne sur le circuit
lui-meme, et non sur la formule qui l'a produit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.electronique.domain.abcd import (
    cascade,
    derivation,
    gain_en_tension,
    serie,
)

if TYPE_CHECKING:
    from polytech.electronique.domain.conversions_d_elements import Branche


@dataclass(frozen=True, slots=True)
class EchelleTerminee:
    """Des branches en echelle, entre une resistance de source et une charge."""

    branches: tuple[Branche, ...]
    serie_d_abord: bool
    source: float
    charge: float

    def __post_init__(self) -> None:
        """Refuse une echelle vide ou des terminaisons sans sens."""
        if not self.branches:
            msg = "echelle vide : aucune branche a mettre en cascade"
            raise ValueError(msg)
        if self.source < 0 or self.charge <= 0:
            msg = f"terminaisons invalides : {self.source} et {self.charge} ohm"
            raise ValueError(msg)

    def transfert(self, variable: sp.Expr) -> sp.Expr:
        """`V_charge / V_source`, resistances de source et de charge comprises."""
        etages = [serie(sp.nsimplify(self.source, rational=True))]
        dans_le_bras = self.serie_d_abord
        for branche in self.branches:
            impedance = branche.impedance(variable)
            etages.append(serie(impedance) if dans_le_bras else derivation(impedance))
            dans_le_bras = not dans_le_bras
        charge = sp.nsimplify(self.charge, rational=True)
        return sp.cancel(sp.together(gain_en_tension(cascade(*etages), charge)))

    def module(self, pulsation: float) -> float:
        """`|V_charge / V_source|` a cette pulsation, en rad/s."""
        point = sp.I * sp.nsimplify(pulsation, rational=True)
        return abs(complex(self.transfert(point)))
