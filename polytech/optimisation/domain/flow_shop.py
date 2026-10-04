"""Chapitre XIV.4.5 (3) : les contraintes du « flow shop » de permutation.

Les variables, dans l'ordre : `x_ij` (tache `i`, position `j`), puis
`y_jk` (position `j`, machines `k = 1 .. m - 1`), puis `z_jk` (position
`j`, machines `k = 2 .. m` -- `z_j1 = 0`, la premiere machine ne
s'arrete jamais).

La relation de la figure XIV.12 mesure de deux manieres le temps entre la
fin de la tache en position `j` sur la machine `k` et le debut de la
suivante sur la machine `k + 1` :

    z_{j+1,k} + p[j+1]k + y_{j+1,k} = y_{j,k} + p[j],k+1 + z_{j+1,k+1},

avec `p[j]k = sum_i p_ik x_ij`. Les conditions initiales -- `y_1k = 0`,
`z_{1,k+1} = sum(p[1]k', k' <= k)` -- sont discutees dans `ordonnancement`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Contrainte, Sens

if TYPE_CHECKING:
    from collections.abc import Sequence

type Durees = Sequence[Sequence[int]]


class _Indices:
    """Ou se trouve chaque variable dans le vecteur du programme."""

    def __init__(self, taches: int, machines: int) -> None:
        self.taches, self.machines = taches, machines
        self.attentes = taches * taches
        self.inactivites = self.attentes + taches * (machines - 1)
        self.total = self.inactivites + taches * (machines - 1)

    def x(self, tache: int, position: int) -> int:
        return tache * self.taches + position

    def y(self, position: int, machine: int) -> int:
        return self.attentes + position * (self.machines - 1) + machine

    def z(self, position: int, machine: int) -> int | None:
        """`None` pour la premiere machine : `z_j1 = 0` n'est pas une variable."""
        if machine == 0:
            return None
        return self.inactivites + position * (self.machines - 1) + machine - 1


def _ecrire(total: int, termes: Sequence[tuple[int | None, int]]) -> Contrainte:
    ligne = [0] * total
    for rang, coefficient in termes:
        if rang is not None:
            ligne[rang] += coefficient
    return Contrainte(tuple(ligne), Sens.EGAL, 0)


def _permutation(indices: _Indices) -> list[Contrainte]:
    n = indices.taches
    lignes = []
    for fixe in range(n):
        par_tache = [0] * indices.total
        par_position = [0] * indices.total
        for libre in range(n):
            par_tache[indices.x(fixe, libre)] = 1
            par_position[indices.x(libre, fixe)] = 1
        lignes.append(Contrainte(tuple(par_tache), Sens.EGAL, 1))
        lignes.append(Contrainte(tuple(par_position), Sens.EGAL, 1))
    return lignes


def _relation(durees: Durees, indices: _Indices, j: int, k: int) -> Contrainte:
    n = indices.taches
    termes: list[tuple[int | None, int]] = [
        (indices.z(j + 1, k), 1),
        (indices.y(j + 1, k), 1),
        (indices.y(j, k), -1),
        (indices.z(j + 1, k + 1), -1),
    ]
    termes += [(indices.x(i, j + 1), durees[i][k]) for i in range(n)]
    termes += [(indices.x(i, j), -durees[i][k + 1]) for i in range(n)]
    return _ecrire(indices.total, termes)


def _initiales(durees: Durees, indices: _Indices) -> list[Contrainte]:
    n = indices.taches
    lignes = []
    for k in range(indices.machines - 1):
        lignes.append(_ecrire(indices.total, [(indices.y(0, k), 1)]))
        termes: list[tuple[int | None, int]] = [(indices.z(0, k + 1), 1)]
        termes += [(indices.x(i, 0), -sum(durees[i][: k + 1])) for i in range(n)]
        lignes.append(_ecrire(indices.total, termes))
    return lignes


def contraintes_du_flow_shop(
    durees: Durees,
    *,
    conditions_initiales: bool,
) -> tuple[Contrainte, ...]:
    """La permutation, la relation de la figure XIV.12, les conditions initiales."""
    indices = _Indices(len(durees), len(durees[0]))
    relations = [
        _relation(durees, indices, j, k)
        for j in range(indices.taches - 1)
        for k in range(indices.machines - 1)
    ]
    initiales = _initiales(durees, indices) if conditions_initiales else []
    return (*_permutation(indices), *relations, *initiales)
