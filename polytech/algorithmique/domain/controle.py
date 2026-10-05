"""Les structures de controle : ce qui s'execute, et combien de fois.

Le cours (partie 1, planches 27 a 53) :

* un `if` sans accolades ne commande que l'instruction qui le suit, et un
  `else` se rattache au `if` le plus proche ;
* `for` et `while` testent leur condition **avant** chaque tour : ils
  peuvent ne rien executer. `do while` la teste **apres** : il execute au
  moins une fois. A la sortie, la variable garde sa derniere valeur ;
* `break` quitte la boucle qui le contient, et elle seule ; `continue`
  passe au tour suivant ;
* dans un `switch`, un bloc sans `break` laisse l'execution tomber dans le
  bloc suivant.

Une boucle dont rien ne fait evoluer la condition ne s'arrete pas : ce
module la refuse au lieu de tourner sans fin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LIMITE: Final = 10_000
"""Le nombre de tours au-dela duquel une boucle est tenue pour infinie."""


class BoucleInfinie(RuntimeError):
    """La condition de la boucle ne devient jamais fausse."""


@dataclass(frozen=True, slots=True)
class Parcours:
    """Les valeurs pour lesquelles le corps s'execute, et la valeur a la sortie."""

    valeurs: tuple[Any, ...]
    sortie: Any


@dataclass(frozen=True, slots=True)
class Cas:
    """Une etiquette `case`, son bloc, et s'il se termine par `break`."""

    valeur: object
    bloc: str
    avec_break: bool = True


def _derouler(
    courante: Any,  # noqa: ANN401 -- la variable de boucle, de tout type
    tant_que: Callable[[Any], bool],
    pas: Callable[[Any], Any],
    *,
    d_abord: bool,
) -> Parcours:
    valeurs: list[Any] = []
    encore = d_abord or tant_que(courante)
    while encore:
        if len(valeurs) >= LIMITE:
            msg = f"boucle infinie : plus de {LIMITE} tours sans sortir"
            raise BoucleInfinie(msg)
        valeurs.append(courante)
        courante = pas(courante)
        encore = tant_que(courante)
    return Parcours(tuple(valeurs), courante)


def boucle_for(
    debut: Any,  # noqa: ANN401 -- la variable de boucle, de tout type
    tant_que: Callable[[Any], bool],
    pas: Callable[[Any], Any],
) -> Parcours:
    """`for` ou `while` : la condition est testee avant chaque tour."""
    return _derouler(debut, tant_que, pas, d_abord=False)


def boucle_do_while(
    debut: Any,  # noqa: ANN401 -- la variable de boucle, de tout type
    tant_que: Callable[[Any], bool],
    pas: Callable[[Any], Any],
) -> Parcours:
    """`do while` : le corps s'execute, puis la condition est testee."""
    return _derouler(debut, tant_que, pas, d_abord=True)


def jusqu_au_break(
    valeurs: Sequence[Any], arret: Callable[[Any], bool]
) -> tuple[Any, ...]:
    """Les valeurs traitees, celle qui declenche le `break` comprise."""
    for rang, valeur in enumerate(valeurs):
        if arret(valeur):
            return tuple(valeurs[: rang + 1])
    return tuple(valeurs)


def sans_les_continue(
    valeurs: Sequence[Any], saut: Callable[[Any], bool]
) -> tuple[Any, ...]:
    """Les valeurs traitees, celles que `continue` fait sauter exclues."""
    return tuple(valeur for valeur in valeurs if not saut(valeur))


def selon(valeur: object, cas: Sequence[Cas], defaut: str | None) -> tuple[str, ...]:
    """Les blocs qu'execute `switch (valeur)`, chute dans les suivants comprise."""
    blocs = [(c.bloc, c.avec_break) for c in cas]
    etiquettes = [c.valeur for c in cas]
    if defaut is not None:
        blocs.append((defaut, True))
    depart = etiquettes.index(valeur) if valeur in etiquettes else len(etiquettes)
    executes: list[str] = []
    for bloc, avec_break in blocs[depart:]:
        executes.append(bloc)
        if avec_break:
            break
    return tuple(executes)


def instructions_executees(
    condition: bool,  # noqa: FBT001 -- la condition du `if`, telle qu'evaluee
    instructions: Sequence[str],
    *,
    accolades: int,
) -> tuple[str, ...]:
    """Ce qui s'execute apres `if (condition)`, selon ce que les accolades enferment."""
    sous_condition = max(accolades, 1)
    debut = 0 if condition else sous_condition
    return tuple(instructions[debut:])


def branche_des_if_emboites(
    exterieure: bool,  # noqa: FBT001 -- la condition du premier `if`
    interieure: bool,  # noqa: FBT001 -- la condition du second `if`
    *,
    accolades: bool,
) -> str | None:
    """`if (a) if (b) alors; else sinon;` : le bloc execute, ou `None`."""
    if accolades:
        return ("alors" if interieure else None) if exterieure else "sinon"
    return ("alors" if interieure else "sinon") if exterieure else None


def premiere_branche_vraie(conditions: Sequence[bool]) -> int:
    """Le rang de la premiere condition vraie d'une cascade, ou celui du `else`."""
    return next(
        (rang for rang, vraie in enumerate(conditions) if vraie), len(conditions)
    )
