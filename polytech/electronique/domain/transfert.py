"""Le diviseur de tension R / (R_L + L) du sujet, et son diagramme de Bode.

La sortie est prise aux bornes de la bobine reelle -- sa resistance serie
`R_L` en serie avec son inductance `L` --, l'ensemble etant alimente a
travers `R`. Ecrite sous forme normalisee, la fonction de transfert exhibe
directement son zero et son pole, ce que le corrige fait a la main.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

S = sp.Symbol("s")
"""Variable de Laplace, nommee comme dans l'enonce."""


@dataclass(frozen=True, slots=True)
class DiviseurRL:
    """Une resistance serie, puis une bobine reelle dont on prend la tension."""

    resistance_serie: sp.Expr
    resistance_bobine: sp.Expr
    inductance: sp.Expr

    def fonction_de_transfert(self) -> sp.Expr:
        """Rend `H(s)` sous la forme normalisee `(s + z)/(s + p)`.

        Le diviseur donne `(R_L + sL)/(R + R_L + sL)` ; diviser haut et bas
        par `L` fait apparaitre le zero et le pole sans calcul de plus.
        """
        return (S + self.resistance_bobine / self.inductance) / (
            S + (self.resistance_serie + self.resistance_bobine) / self.inductance
        )

    def zero(self) -> sp.Expr:
        """La pulsation ou le numerateur s'annule : `-R_L / L`."""
        return -self.resistance_bobine / self.inductance

    def pole(self) -> sp.Expr:
        """La pulsation ou le denominateur s'annule : `-(R + R_L) / L`."""
        return -(self.resistance_serie + self.resistance_bobine) / self.inductance

    def gain_continu(self) -> sp.Expr:
        """Gain en continu : la bobine est un court-circuit, reste un pont."""
        return self.resistance_bobine / (self.resistance_serie + self.resistance_bobine)

    def gain_continu_en_decibels(self) -> float:
        """Le meme gain, en decibels de tension."""
        return 20 * math.log10(float(self.gain_continu()))

    def gain_infini_en_decibels(self) -> float:
        """A frequence infinie la bobine domine : tout passe, donc 0 dB."""
        return 0.0
