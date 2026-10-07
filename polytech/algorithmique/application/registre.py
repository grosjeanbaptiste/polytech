"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention que dans les autres matieres : le nom du dossier sous
`travaux/ProgrammationAlgorithmique/` designe le jeu, et chaque jeu
prefixe ses macros pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.algorithmique.application import macros_tp_2017, macros_tp_2019

if TYPE_CHECKING:
    from collections.abc import Callable

JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "tp-01-a-06-2019": macros_tp_2019.macros_des_tp_de_2019,
    "tp-04-a-06-2017": macros_tp_2017.macros_des_tp_de_2017,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
