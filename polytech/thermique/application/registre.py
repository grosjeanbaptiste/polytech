"""Les travaux de thermique au-dela du TP 1, chacun avec son jeu de macros.

Le TP 1 garde sa commande d'origine, `GenererMacros`, et rien n'en est
change. Les suivants s'inscrivent ici, comme en electromagnetisme : le
nom du dossier sous `travaux/ThermiqueFondements/` designe le jeu.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.thermique.application import macros_tp_02, macros_tp_03

if TYPE_CHECKING:
    from collections.abc import Callable

JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "tp-02-camion-refrigere": macros_tp_02.macros_du_tp_deux,
    "tp-03-capteur-solaire": macros_tp_03.macros_du_tp_trois,
}


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail ; leve s'il n'est pas inscrit."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
