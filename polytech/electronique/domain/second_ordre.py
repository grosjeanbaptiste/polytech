"""Le passe-bas canonique du second ordre, tel que le cours l'ecrit.

`G omega_0^2 / (s^2 + (omega_0/Q) s + omega_0^2)` : trois nombres, et tout
le comportement en decoule. Les poles, en particulier, ne dependent que de
`Q` pour leur nature -- reels ou complexes conjugues -- et jamais de
`omega_0`, qui ne fait que les mettre a l'echelle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

LAPLACE = sp.Symbol("s")
"""Variable de Laplace, nommee comme dans l'enonce."""

QUALITE_CRITIQUE = sp.Rational(1, 2)
"""En deca, le discriminant est positif et les deux poles sont reels."""


def qualite_de_coupure_a_trois_decibels() -> float:
    """Le `Q` qui place la coupure a -3 dB exactement sur `omega_0`.

    A cette pulsation le module vaut `G Q` et le gain continu `G` : leur
    rapport est donc `Q`, et -3 dB en amplitude valent `1/racine(2)`.
    """
    return 1 / math.sqrt(2)


@dataclass(frozen=True, slots=True)
class PasseBasCanonique:
    """Un filtre defini par son gain continu, sa pulsation et sa qualite."""

    gain: sp.Expr
    pulsation: sp.Expr
    qualite: sp.Expr

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.pulsation == 0:
            msg = "pulsation nulle : le filtre n'a pas de reference"
            raise ValueError(msg)
        if self.qualite == 0:
            msg = "facteur de qualite nul : le terme en s serait infini"
            raise ValueError(msg)

    def transfert(self, variable: sp.Expr = LAPLACE) -> sp.Expr:
        """La forme canonique elle-meme."""
        return (self.gain * self.pulsation**2) / (
            variable**2 + (self.pulsation / self.qualite) * variable + self.pulsation**2
        )

    def gain_continu(self) -> sp.Expr:
        """`H(j0)`, obtenu en annulant la variable."""
        return sp.simplify(self.transfert(sp.Integer(0)))

    def reponse_a_la_pulsation_propre(self) -> sp.Expr:
        """`H(j omega_0)` : la partie reelle du denominateur s'y annule."""
        return sp.simplify(self.transfert(sp.I * self.pulsation))

    def poles(self) -> tuple[sp.Expr, sp.Expr]:
        """Les deux racines du denominateur, sans supposer leur nature."""
        racines = sp.solve(
            sp.denom(sp.together(self.transfert(LAPLACE))),
            LAPLACE,
        )
        if len(racines) == 1:
            return (racines[0], racines[0])
        return (racines[0], racines[1])

    def poles_sont_reels(self) -> bool:
        """Vrai quand le discriminant est positif ou nul, soit `Q <= 1/2`."""
        return all(sp.im(pole) == 0 for pole in self.poles())

    def est_stable(self) -> bool:
        """Vrai quand les deux poles sont a partie reelle strictement negative."""
        return all(bool(sp.re(pole) < 0) for pole in self.poles())
