"""La quantification d'un convertisseur : codes, niveaux, saturation, ouverture.

Chapitre 4, partie III, planches 23 a 32. Un convertisseur de `n` bits
couvrant `[-V_r, V_r]` rend l'un de `2^n` codes, espaces du pas `LSB =
2 V_r / (2^n - 1)`. Il arrondit au niveau le plus proche, donc se trompe
d'un demi-pas au plus, et **sature** au-dela de sa pleine echelle : c'est
pourquoi le gain de la chaine se regle pour approcher la pleine echelle
sans l'atteindre.

Un convertisseur reel regarde le signal pendant un temps d'ouverture
`t_A` : l'echantillon rendu vaut a peu pres la moyenne du signal sur cette
duree, et non sa valeur a un instant.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class Quantificateur:
    """Un convertisseur de `bits` bits sur la plage `[-reference, reference]`."""

    reference: float
    bits: int

    def __post_init__(self) -> None:
        """Refuse une plage vide ou une resolution nulle."""
        if self.reference <= 0:
            msg = f"tension de reference nulle ou negative : {self.reference} V"
            raise ValueError(msg)
        if self.bits < 1:
            msg = f"resolution inferieure a un bit : {self.bits}"
            raise ValueError(msg)

    def code_maximal(self) -> int:
        """`2^n - 1` : le code de la borne haute."""
        return (1 << self.bits) - 1

    def pas(self) -> float:
        """`LSB = 2 V_r / (2^n - 1)`."""
        return 2 * self.reference / self.code_maximal()

    def sature(self, tension: float) -> bool:
        """Vrai si la tension sort de la pleine echelle."""
        return abs(tension) > self.reference

    def code(self, tension: float) -> int:
        """Le code du niveau le plus proche, borne par la saturation."""
        brut = round((tension + self.reference) / self.pas())
        return min(max(brut, 0), self.code_maximal())

    def niveau(self, code: int) -> float:
        """La tension que ce code represente."""
        if not 0 <= code <= self.code_maximal():
            msg = f"code hors de [0, {self.code_maximal()}] : {code}"
            raise ValueError(msg)
        return -self.reference + code * self.pas()


def echantillon_a_ouverture(
    signal: sp.Expr, temps: sp.Symbol, *, instant: object, ouverture: object
) -> sp.Expr:
    """La moyenne du signal sur `[instant, instant + t_A]`."""
    duree = sp.nsimplify(ouverture, rational=True)
    if not duree.is_positive:
        msg = f"temps d'ouverture nul ou negatif : {ouverture} s"
        raise ValueError(msg)
    debut = sp.nsimplify(instant, rational=True)
    return sp.integrate(signal, (temps, debut, debut + duree)) / duree
