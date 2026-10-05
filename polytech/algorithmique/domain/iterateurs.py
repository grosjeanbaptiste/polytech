"""Les iterateurs : parcourir un conteneur sans connaitre sa forme.

Le cours (seances 5 et 6, « Les iterateurs ») : « des pointeurs
specialises permettant d'acceder aux elements d'un conteneur. La methode
`begin()` renvoie l'iterateur du premier element, et `end()` celui de la
position suivant le dernier element ». Une plage `[first, last)` contient
donc `first` et s'arrete avant `last`.

Une position entiere tient ici lieu d'iterateur. Avancer au-dela de
`end()` est un comportement indefini en C++ -- « crash tres probable » ;
c'est ici une erreur.

Le chiffrement de la planche des iterateurs parcourt une chaine caractere
par caractere : une majuscule est decalee dans l'alphabet, en boucle ; un
blanc, une apostrophe, un point d'exclamation, `e` et `t` passent tels
quels ; tout autre caractere est remplace par son code moins 105, suivi
d'un blanc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_INTACTS: Final = " '!et"
_BASE_DES_CODES: Final = 105
_LETTRES: Final = 26


def plage(tableau: Sequence[Any], debut: int, fin: int) -> tuple[Any, ...]:
    """`vector<T> V(T + debut, T + fin)` : les elements de `[debut, fin)`."""
    if not 0 <= debut <= fin <= len(tableau):
        msg = f"plage hors du tableau : {debut} a {fin}"
        raise IndexError(msg)
    return tuple(tableau[debut:fin])


def parcours_inverse(elements: Sequence[Any]) -> tuple[Any, ...]:
    """De `rbegin()` a `rend()` : du dernier element au premier."""
    return tuple(reversed(elements))


def avancer(position: int, pas: int, taille: int) -> int:
    """`advance(it, pas)` : la position atteinte, au plus `end()`."""
    if position + pas > taille:
        msg = f"iterateur avance au-dela de end : {position + pas} pour {taille}"
        raise IndexError(msg)
    return position + pas


def _chiffre(caractere: str, decalage: int) -> str:
    if caractere.isupper():
        rang = (ord(caractere) + decalage - ord("A")) % _LETTRES
        return chr(rang + ord("A"))
    if caractere in _INTACTS:
        return caractere
    return f"{ord(caractere) - _BASE_DES_CODES} "


def chiffrer(texte: str, decalage: int) -> str:
    """Le texte de la planche, chiffre caractere par caractere."""
    return "".join(_chiffre(caractere, decalage) for caractere in texte)
