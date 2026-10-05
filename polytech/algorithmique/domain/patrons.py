"""Les patrons de fonctions : un meme code pour plusieurs types.

Le cours (seance 4, « Patrons : fonctions et classes ») : « un patron de
fonction constitue une definition generique s'appliquant a des types de
donnees differents ». `template <class T> T min(T a, T b)` vaut pour
`int`, `float`, `char` : le compilateur **deduit** `T` des arguments de
chaque appel et fabrique la fonction correspondante.

Deux consequences se calculent :

* chaque parametre de type est deduit de tous les arguments qui le
  portent ; s'ils ne s'accordent pas -- `min(2, 9.5)` -- la deduction
  echoue et l'appel ne compile pas. Aucune conversion n'est tentee ;
* le patron calcule **dans le type deduit**. La somme d'un tableau de
  `char` est un `char` : « H E L L O » fait 372, qui se replie en 116,
  le code de `t`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polytech.algorithmique.domain.types_cpp import affecter

if TYPE_CHECKING:
    from collections.abc import Sequence


def minimum(a: Any, b: Any) -> Any:  # noqa: ANN401 -- tout type ordonne
    """`if (a < b) return a; else return b;` -- pour tout type qui a un `<`."""
    return min(b, a)


def types_deduits(
    parametres: Sequence[str], arguments: Sequence[str], fixes: Sequence[str] = ()
) -> dict[str, str]:
    """Le type que prend chaque parametre de patron, d'apres ceux des arguments."""
    if len(parametres) != len(arguments):
        msg = f"nombre d'arguments inattendu : {len(arguments)} pour {len(parametres)}"
        raise TypeError(msg)
    deduits: dict[str, str] = {}
    for parametre, argument in zip(parametres, arguments, strict=True):
        if parametre in fixes:
            continue
        if deduits.setdefault(parametre, argument) != argument:
            msg = f"deduction contradictoire pour {parametre}"
            raise TypeError(msg)
    return deduits


def somme_du_patron(tableau: Sequence[float | str], type_deduit: str) -> float:
    """`T som = 0; som = som + tab[i];` : la somme, rangee dans `T` a chaque pas."""
    somme: float = 0
    for element in tableau:
        valeur = ord(element) if isinstance(element, str) else element
        somme = affecter(somme + valeur, type_deduit)
    return somme
