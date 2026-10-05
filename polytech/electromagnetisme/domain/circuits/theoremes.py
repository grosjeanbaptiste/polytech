"""Les theoremes des reseaux lineaires : superposition et reciprocite.

Le syllabus d'electricite generale (13.19 et 13.22). Les equations de
Kirchhoff etant lineaires, le courant d'une branche est la somme de ceux
qu'y ferait circuler chaque force electromotrice seule, les autres
annulees.

Dans un reseau passif, une force electromotrice inseree dans une branche
fait circuler dans une autre le meme courant que si l'on echangeait les
deux branches : `I_2 / E_1 = I'_1 / E'_2` (13.22.2), l'egalite des
conductances de transfert. C'est un cas de la relation generale
(13.22.1), analogue de l'identite de Gauss.

Les theoremes de Thevenin et de Norton (13.20 et 13.21) vivent dans
`equivalents` et `stationnaires`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polytech.electromagnetisme.domain.circuits.mailles import (
    Branche,
    resoudre_par_les_mailles,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import sympy as sp


def _une_seule_source(
    branches: Sequence[Branche],
    rang: int,
    fem: Any,  # noqa: ANN401 -- reel ou symbole
) -> tuple[Branche, ...]:
    return tuple(
        Branche(b.de, b.vers, b.resistance, fem if place == rang else 0)
        for place, b in enumerate(branches)
    )


def par_superposition(
    branches: Sequence[Branche], arbre: Collection[int]
) -> tuple[sp.Expr, ...]:
    """Les courants de branche, chaque force electromotrice agissant seule."""
    total = resoudre_par_les_mailles(_une_seule_source(branches, -1, 0), arbre)
    for rang, branche in enumerate(branches):
        seule = _une_seule_source(branches, rang, branche.fem)
        partiel = resoudre_par_les_mailles(seule, arbre)
        total = tuple(a + b for a, b in zip(total, partiel, strict=True))
    return total


def conductance_de_transfert(
    branches: Sequence[Branche], arbre: Collection[int], *, source: int, mesure: int
) -> sp.Expr:
    """`I_mesure / E_source` (13.22.2), en siemens, le reseau rendu passif."""
    unitaire = _une_seule_source(branches, source, 1)
    return resoudre_par_les_mailles(unitaire, arbre)[mesure]
