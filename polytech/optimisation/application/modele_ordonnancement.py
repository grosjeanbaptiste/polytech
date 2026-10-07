"""Modelisation, planches 87 a 89 : trois modeles de pieces sur trois machines.

Chaque modele passe sur ses machines dans un ordre impose ; une machine
ne traite qu'un modele a la fois. Pour chaque paire d'operations qui se
disputent une machine, une binaire dit laquelle passe d'abord.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

Operation = tuple[int, int]
"""Un modele et la machine sur laquelle il passe."""
GAMMES: Final = {
    1: ((1, 45), (3, 10)),
    2: ((2, 10), (1, 20), (3, 34)),
    3: ((2, 17), (3, 28)),
}
"""Planche 87 : pour chaque modele, ses machines dans l'ordre, et les durees."""
DUREES: Final = {(m, machine): d for m, gamme in GAMMES.items() for machine, d in gamme}
CONFLITS: Final[tuple[tuple[Operation, Operation], ...]] = tuple(
    paire
    for paire in itertools.combinations(sorted(DUREES, key=lambda o: (o[1], o[0])), 2)
    if paire[0][1] == paire[1][1]
)
"""Les paires d'operations sur une meme machine : une binaire chacune."""
GANTT: Final = {
    (2, 2): 0,
    (3, 2): 10,
    (2, 1): 10,
    (1, 1): 30,
    (3, 3): 27,
    (2, 3): 55,
    (1, 3): 89,
}
"""Planche 89 : le debut de chaque operation, lu sur le diagramme."""
Debuts = dict[Operation, int]


def _gammes() -> list[tuple[Operation, Operation]]:
    """Les arcs « finit avant que commence » qu'imposent les gammes."""
    arcs = []
    for modele, gamme in GAMMES.items():
        machines = [machine for machine, _ in gamme]
        arcs += [((modele, a), (modele, b)) for a, b in itertools.pairwise(machines)]
    return arcs


def au_plus_tot(ordre: tuple[int, ...]) -> tuple[int, Debuts] | None:
    """La fin et les debuts au plus tot pour cet ordre ; rien s'il se contredit."""
    arcs = _gammes()
    for (premiere, seconde), d_abord in zip(CONFLITS, ordre, strict=True):
        arcs.append((premiere, seconde) if d_abord else (seconde, premiere))
    debuts = dict.fromkeys(DUREES, 0)
    for _ in range(len(DUREES) + 1):
        avance = False
        for avant, apres in arcs:
            if debuts[avant] + DUREES[avant] > debuts[apres]:
                debuts[apres] = debuts[avant] + DUREES[avant]
                avance = True
        if not avance:
            return max(debuts[o] + DUREES[o] for o in DUREES), debuts
    return None


@cache
def _classement() -> tuple[tuple[int, tuple[int, ...], Debuts], ...]:
    """Chaque ordre sans contradiction : sa fin, lui-meme, ses debuts au plus tot."""
    lignes = []
    for ordre in itertools.product((0, 1), repeat=len(CONFLITS)):
        resultat = au_plus_tot(ordre)
        if resultat is not None:
            lignes.append((resultat[0], ordre, resultat[1]))
    return tuple(sorted(lignes, key=lambda ligne: ligne[:2]))


def ordres_possibles() -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Les ordres sans contradiction avec leur date de fin, du plus court d'abord."""
    return tuple((fin, ordre) for fin, ordre, _ in _classement())


def meilleur_ordonnancement() -> tuple[int, Debuts]:
    """La plus petite date de fin, et les debuts du premier ordre qui l'atteint."""
    fin, _, debuts = _classement()[0]
    return fin, debuts


def est_realisable(debuts: Mapping[Operation, int]) -> bool:
    """Vrai si les gammes sont respectees et qu'aucune machine n'est prise deux fois."""
    gammes = all(
        debuts[avant] + DUREES[avant] <= debuts[apres] for avant, apres in _gammes()
    )
    libres = all(
        debuts[a] + DUREES[a] <= debuts[b] or debuts[b] + DUREES[b] <= debuts[a]
        for a, b in CONFLITS
    )
    return gammes and libres


def charge_de_la_machine(machine: int) -> int:
    """Le temps total de travail de cette machine."""
    durees = [d for (_, m), d in DUREES.items() if m == machine]
    if not durees:
        msg = f"machine inconnue : {machine}"
        raise KeyError(msg)
    return sum(durees)


def arrivee_au_plus_tot(machine: int) -> int:
    """Le plus tot qu'un modele puisse se presenter devant cette machine."""
    arrivees = []
    for gamme in GAMMES.values():
        machines = [m for m, _ in gamme]
        if machine in machines:
            arrivees.append(sum(d for _, d in gamme[: machines.index(machine)]))
    return min(arrivees)
