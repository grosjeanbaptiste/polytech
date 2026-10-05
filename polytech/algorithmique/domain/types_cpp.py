"""Les types de donnees du C++ et l'arithmetique qu'ils imposent.

Le cours (partie 1, planches 11 a 26) : chaque type entier a une plage de
valeurs fixee par son nombre de bits -- de `-2^(n-1)` a `2^(n-1) - 1` s'il
est signe, de 0 a `2^n - 1` sinon. Trois regles en decoulent, et chacune
est un piege classique :

* la division de deux entiers est **entiere**, et tronque vers zero :
  `7/2` vaut 3. Il suffit d'un operande reel pour qu'elle soit reelle ;
* `%` ne s'applique qu'aux entiers, et son resultat a le signe du
  dividende ;
* affecter un reel a un entier **tronque**, et un resultat trop grand
  **deborde** : il se replie dans la plage.

Python ne fait rien de tout cela -- sa division entiere arrondit vers le
bas, ses entiers ne debordent pas. Ce module dit donc ce qu'un programme
C++ doit afficher, pour qu'un corrige ait quelque chose a prouver.

Le `int` y est sur 32 bits, comme sur les machines du cours ; la planche
11 donne aussi sa plage sur 16 bits.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TypeEntier:
    """Un type entier : son nombre de bits, et s'il porte un signe."""

    bits: int
    signe: bool

    @property
    def minimum(self) -> int:
        """La plus petite valeur representable."""
        return -(1 << (self.bits - 1)) if self.signe else 0

    @property
    def maximum(self) -> int:
        """La plus grande valeur representable."""
        return (1 << (self.bits - 1)) - 1 if self.signe else (1 << self.bits) - 1

    def contient(self, valeur: int) -> bool:
        """Vrai si la valeur tient dans le type sans deborder."""
        return self.minimum <= valeur <= self.maximum

    def ranger(self, valeur: int) -> int:
        """La valeur telle que le type la garde : repliee modulo `2^n`."""
        repliee = valeur % (1 << self.bits)
        return repliee - (1 << self.bits) if repliee > self.maximum else repliee


TYPES_ENTIERS: Final = {
    "char": TypeEntier(8, signe=True),
    "unsigned char": TypeEntier(8, signe=False),
    "short int": TypeEntier(16, signe=True),
    "unsigned short int": TypeEntier(16, signe=False),
    "int": TypeEntier(32, signe=True),
    "unsigned int": TypeEntier(32, signe=False),
    "long int": TypeEntier(32, signe=True),
    "unsigned long int": TypeEntier(32, signe=False),
}
"""Le tableau de la planche 11, pour ce qui est des entiers."""

TYPES_REELS: Final = ("float", "double", "long double")


def en_simple_precision(valeur: float) -> float:
    """Le reel tel qu'un `float` le garde : arrondi sur 32 bits."""
    (arrondi,) = struct.unpack("f", struct.pack("f", valeur))
    return float(arrondi)


def tronquer(valeur: float) -> int:
    """La conversion d'un reel en entier : la partie decimale tombe."""
    return int(valeur)


def _quotient_entier(dividende: int, diviseur: int) -> int:
    """Le quotient tronque vers zero, sans passer par les flottants."""
    quotient = abs(dividende) // abs(diviseur)
    return quotient if (dividende < 0) == (diviseur < 0) else -quotient


def diviser(dividende: float, diviseur: float) -> float:
    """`a / b` : entiere et tronquee vers zero entre deux entiers, reelle sinon."""
    if diviseur == 0:
        msg = "division par zero : le resultat n'est pas defini"
        raise ZeroDivisionError(msg)
    if isinstance(dividende, int) and isinstance(diviseur, int):
        return _quotient_entier(dividende, diviseur)
    return dividende / diviseur


def modulo(dividende: float, diviseur: float) -> int:
    """`a % b` : le reste de la division entiere, du signe du dividende."""
    if not (isinstance(dividende, int) and isinstance(diviseur, int)):
        msg = "modulo de deux entiers seulement : % refuse les reels"
        raise TypeError(msg)
    if diviseur == 0:
        msg = "division par zero : le reste n'est pas defini"
        raise ZeroDivisionError(msg)
    return dividende - _quotient_entier(dividende, diviseur) * diviseur


def affecter(valeur: float, type_cible: str) -> float:
    """La valeur telle que la garde une variable du type donne."""
    if type_cible in TYPES_REELS:
        return float(valeur)
    if type_cible not in TYPES_ENTIERS:
        msg = f"type inconnu : {type_cible}"
        raise ValueError(msg)
    return TYPES_ENTIERS[type_cible].ranger(tronquer(valeur))


def caractere_decale(caractere: str, ecart: int) -> str:
    """`c + ecart`, un `char` etant un petit entier : son code ASCII."""
    code = ord(caractere) + ecart
    if not TYPES_ENTIERS["char"].contient(code) or code < 0:
        msg = f"code {code} hors de la plage des caracteres ASCII"
        raise ValueError(msg)
    return chr(code)


def est_vrai(valeur: float) -> bool:
    """Toute valeur non nulle est vraie, la valeur nulle est fausse."""
    return valeur != 0
