"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention qu'en electromagnetisme et en signaux : le nom du dossier
sous `travaux/ElectroniqueInstrumentation/` designe le jeu, et chaque
jeu prefixe ses macros pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.electronique.application import (
    macros_annexe_tp_07,
    macros_lab_01,
    macros_lab_03,
    macros_lab_05,
    macros_tp_01,
    macros_tp_02,
    macros_tp_03,
    macros_tp_04,
    macros_tp_05,
    macros_tp_06,
    macros_tp_07,
    macros_tp_08,
)

if TYPE_CHECKING:
    from collections.abc import Callable

JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "lab-01-filtrage": macros_lab_01.macros_du_labo_un,
    "lab-03-instrumentation": macros_lab_03.macros_du_labo_trois,
    "lab-05-amplificateur-audio": macros_lab_05.macros_du_labo_cinq,
    "tp-01": macros_tp_01.macros_du_tp_un,
    "tp-02": macros_tp_02.macros_du_tp_deux,
    "tp-03": macros_tp_03.macros_du_tp_trois,
    "tp-04": macros_tp_04.macros_du_tp_quatre,
    "tp-05": macros_tp_05.macros_du_tp_cinq,
    "tp-06": macros_tp_06.macros_du_tp_six,
    "tp-07": macros_tp_07.macros_du_tp_sept,
    "tp-07-bis": macros_annexe_tp_07.macros_de_l_annexe,
    "tp-08": macros_tp_08.macros_du_tp_huit,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
