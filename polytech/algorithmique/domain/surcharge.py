"""La surcharge des fonctions et les parametres par defaut.

Le cours (partie 1, planches 122 et 123) : plusieurs fonctions peuvent
porter le meme nom si elles different par le type ou le nombre de leurs
parametres. Le compilateur choisit celle dont les parametres conviennent
le mieux aux arguments, chaque argument etant classe :

* **exact** : meme type ;
* **promotion** : `char`, `bool` ou `short` vers `int`, `float` vers
  `double` ;
* **conversion** : tout autre passage entre types numeriques.

Une fonction l'emporte si aucun de ses arguments n'est moins bien classe
que chez une autre, et qu'un au moins l'est mieux. Si aucune ne l'emporte
sur toutes, l'appel est **ambigu** et ne compile pas.

Un parametre par defaut prend sa valeur quand l'argument manque ; seuls
les derniers arguments peuvent manquer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROMOTIONS: Final = {
    ("char", "int"),
    ("bool", "int"),
    ("short", "int"),
    ("float", "double"),
}
EXACT, PROMOTION, CONVERSION = 0, 1, 2


def _rang(argument: str, parametre: str) -> int:
    if argument == parametre:
        return EXACT
    return PROMOTION if (argument, parametre) in _PROMOTIONS else CONVERSION


def _l_emporte(rangs: Sequence[int], autres: Sequence[int]) -> bool:
    paires = list(zip(rangs, autres, strict=True))
    return all(r <= a for r, a in paires) and any(r < a for r, a in paires)


def surcharge_choisie(
    signatures: Sequence[Sequence[str]], arguments: Sequence[str]
) -> int:
    """Le rang de la fonction appelee ; leve si aucune ne convient, ou deux."""
    viables = {
        rang: [_rang(a, p) for a, p in zip(arguments, signature, strict=True)]
        for rang, signature in enumerate(signatures)
        if len(signature) == len(arguments)
    }
    if not viables:
        msg = f"aucune fonction ne correspond a {len(arguments)} argument(s)"
        raise TypeError(msg)
    meilleures = [
        rang
        for rang, rangs in viables.items()
        if all(r == rang or _l_emporte(rangs, a) for r, a in viables.items())
    ]
    if not meilleures:
        msg = "appel ambigu : aucune fonction n'est meilleure que toutes les autres"
        raise TypeError(msg)
    return meilleures[0]


def arguments_completes(
    defauts: Sequence[Any], arguments: Sequence[Any]
) -> tuple[Any, ...]:
    """Les arguments que recoit la fonction, les manquants pris parmi les defauts."""
    if len(arguments) > len(defauts):
        msg = f"trop d'arguments : {len(arguments)} pour {len(defauts)} parametres"
        raise TypeError(msg)
    return (*arguments, *defauts[len(arguments) :])
