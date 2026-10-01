"""Le pas de quantification d'un convertisseur analogique-numerique.

Le corrige ecrit le pas exact -- pleine echelle sur `2^n - 1` -- puis
l'approxime par `2^n`. Les deux sont ici, parce que l'approximation est une
affirmation comme une autre : elle doit pouvoir se chiffrer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class ConvertisseurAnalogiqueNumerique:
    """Un convertisseur, defini par sa resolution et sa pleine echelle."""

    bits: int
    pleine_echelle_volts: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.bits <= 0:
            msg = f"resolution non strictement positive : {self.bits} bits"
            raise ValueError(msg)
        if self.pleine_echelle_volts <= 0:
            msg = f"pleine echelle non positive : {self.pleine_echelle_volts} V"
            raise ValueError(msg)

    def niveaux(self) -> int:
        """Nombre de niveaux representables : deux puissance le nombre de bits."""
        return int(2**self.bits)

    def pas_exact(self) -> float:
        """Pas de quantification exact : la pleine echelle sur `2^n - 1`.

        Les `2^n` niveaux delimitent `2^n - 1` intervalles ; c'est entre les
        bornes extremes que la pleine echelle se repartit.
        """
        return self.pleine_echelle_volts / (self.niveaux() - 1)

    def pas_approche(self) -> float:
        """Le meme pas, approxime par `2^n`, comme le fait le corrige."""
        return self.pleine_echelle_volts / self.niveaux()


class EchantillonnageDeLEtendue(Enum):
    """Les trois cas que le fascicule montre sur un convertisseur quatre bits."""

    CORRECT = "etendue correctement couverte"
    DEPASSEE = "etendue depassee, signal ecrete"
    SOUS_EXPLOITEE = "etendue non couverte, bits gaspilles"


FRACTION_MINIMALE_UTILE = 0.25
"""En dessous, le signal n'emploie pas assez de l'etendue.

Un quart, c'est deux bits perdus sur l'etendue. Le seuil est choisi et
dit ; le fascicule montre le cas sans le chiffrer.
"""


def etat_de_l_echantillonnage(
    amplitude: float,
    etendue: float,
) -> EchantillonnageDeLEtendue:
    """Lequel des trois cas du fascicule ce couple realise.

    Les deux fautes sont symetriques : depasser l'etendue ecrete le
    signal, ne pas la couvrir gaspille des bits. Un dimensionnement doit
    eviter les deux, et c'est pourquoi le gain de la chaine se calcule au
    lieu de se choisir.
    """
    if etendue <= 0:
        msg = f"etendue nulle ou negative : {etendue}"
        raise ValueError(msg)
    if amplitude > etendue:
        return EchantillonnageDeLEtendue.DEPASSEE
    if amplitude < FRACTION_MINIMALE_UTILE * etendue:
        return EchantillonnageDeLEtendue.SOUS_EXPLOITEE
    return EchantillonnageDeLEtendue.CORRECT


def erreur_de_temps_d_ouverture(frequence: float, temps: float) -> float:
    """`2 pi f t_a`, en fraction de pleine echelle.

    L'analogie que le fascicule file : un temps de pose trop long floute un
    sujet rapide, un temps d'ouverture trop long floute un signal rapide.
    L'erreur croit avec la frequence du signal, pas avec son amplitude.
    """
    if frequence < 0 or temps < 0:
        msg = f"frequence {frequence} Hz ou temps {temps} s negatif"
        raise ValueError(msg)
    return 2 * math.pi * frequence * temps


def temps_d_ouverture_maximal(frequence: float, bits: int) -> float:
    """Le temps qui tient l'erreur sous le **demi**-pas de quantification.

    Demi et non entier : au-dela, l'erreur d'ouverture depasse l'erreur de
    quantification et le convertisseur perd un bit utile. C'est le critere
    de dimensionnement du fascicule.
    """
    if bits <= 0:
        msg = f"nombre de bits nul ou negatif : {bits}"
        raise ValueError(msg)
    if frequence <= 0:
        msg = f"frequence nulle ou negative : {frequence} Hz"
        raise ValueError(msg)
    demi_pas = 1 / float(2 ** (bits + 1))
    return demi_pas / (2 * math.pi * frequence)
