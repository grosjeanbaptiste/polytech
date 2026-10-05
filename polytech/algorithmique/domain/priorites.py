"""Les classes de priorite des operateurs du C++, et ce que chacun calcule.

Le syllabus (2.9.5, « Classes de priorites ») : les parentheses d'abord,
puis les unaires `!`, `+`, `-`, puis `* / %`, puis `+ -`, puis les
comparaisons `< <= > >=`, puis `== !=`, puis `&&`, puis `||`.

Deux regles du langage tiennent a ces operateurs :

* la division de deux entiers est entiere, et `%` ne prend que des
  entiers (`types_cpp`) ;
* `&&` et `||` sont **paresseux** : l'operande de droite n'est evalue que
  si celui de gauche ne suffit pas a conclure. Un operande est donc ici un
  calcul differe, que l'operateur declenche ou non.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Final

from polytech.algorithmique.domain.types_cpp import diviser, modulo

if TYPE_CHECKING:
    from collections.abc import Callable

    Calcul = Callable[[], Any]

PRIORITES: Final = (
    ("||",),
    ("&&",),
    ("==", "!="),
    ("<", "<=", ">", ">="),
    ("+", "-"),
    ("*", "/", "%"),
)
"""Les classes d'operateurs binaires, de la moins a la plus prioritaire."""

UNAIRES: Final[dict[str, Callable[[Any], Any]]] = {
    "-": operator.neg,
    "+": operator.pos,
    "!": operator.not_,
}
"""Les operateurs unaires de la classe 2, evalues de droite a gauche."""

_BINAIRES: Final[dict[str, Callable[[Any, Any], Any]]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": diviser,
    "%": modulo,
}


def calcul_binaire(operateur: str, gauche: Calcul, droite: Calcul) -> Calcul:
    """Le calcul differe de `gauche operateur droite`, paresseux pour `&&` et `||`."""
    if operateur == "&&":
        return lambda: bool(gauche()) and bool(droite())
    if operateur == "||":
        return lambda: bool(gauche()) or bool(droite())
    fonction = _BINAIRES[operateur]
    return lambda: fonction(gauche(), droite())
