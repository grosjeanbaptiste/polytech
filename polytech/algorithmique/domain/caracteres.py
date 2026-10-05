"""Les predicats de `<cctype>` : la nature d'un caractere.

Le cours (partie 1, « Predicats verifiant des conditions uniquement sur un
caractere ») : `isalpha(c)` est vrai si `c` est une lettre, `isdigit(c)`
un chiffre, `isspace(c)` un espace, une tabulation ou un retour a la
ligne, `ispunct(c)` un signe de ponctuation... Chaque predicat porte ici
le nom que lui donne `<cctype>`.

Ces predicats ne connaissent que l'**ASCII** : une lettre accentuee n'est
pas une lettre pour eux. Les methodes de `str` en Python disent le
contraire -- `"é".isalpha()` est vrai, `"²".isdigit()` aussi -- et ne
peuvent donc pas dire ce qu'un programme C++ affichera.
"""

from __future__ import annotations

import string
from typing import Final

_BLANCS: Final = " \t"
_ESPACES: Final = " \t\n\v\f\r"


def _code(caractere: str) -> str:
    if len(caractere) != 1:
        msg = f"un seul caractere attendu, et non {len(caractere)}"
        raise ValueError(msg)
    return caractere


def islower(caractere: str) -> bool:
    """Vrai pour une minuscule, de `a` a `z`."""
    return _code(caractere) in string.ascii_lowercase


def isupper(caractere: str) -> bool:
    """Vrai pour une majuscule, de `A` a `Z`."""
    return _code(caractere) in string.ascii_uppercase


def isalpha(caractere: str) -> bool:
    """Vrai pour une lettre, minuscule ou majuscule."""
    return islower(caractere) or isupper(caractere)


def isdigit(caractere: str) -> bool:
    """Vrai pour un chiffre, de 0 a 9."""
    return _code(caractere) in string.digits


def isxdigit(caractere: str) -> bool:
    """Vrai pour un chiffre hexadecimal : 0 a 9, `a` a `f`, `A` a `F`."""
    return _code(caractere) in string.hexdigits


def isalnum(caractere: str) -> bool:
    """Vrai pour une lettre ou un chiffre."""
    return isalpha(caractere) or isdigit(caractere)


def isspace(caractere: str) -> bool:
    """Vrai pour un espace, une tabulation, un retour a la ligne."""
    return _code(caractere) in _ESPACES


def isblank(caractere: str) -> bool:
    """Vrai pour un espace ou une tabulation horizontale."""
    return _code(caractere) in _BLANCS


def ispunct(caractere: str) -> bool:
    """Vrai pour un signe de ponctuation."""
    return _code(caractere) in string.punctuation


def toupper(caractere: str) -> str:
    """La majuscule d'une minuscule ; tout autre caractere tel quel."""
    return caractere.upper() if islower(caractere) else _code(caractere)


def _masque(caractere: str) -> str:
    if isalpha(caractere):
        return toupper(caractere)
    return " " if isspace(caractere) else "*"


def masquer(texte: str) -> str:
    """Les lettres en majuscules, les espaces en blancs, le reste en etoiles."""
    return "".join(_masque(caractere) for caractere in texte)
