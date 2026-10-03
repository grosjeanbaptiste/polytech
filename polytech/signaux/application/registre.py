"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention qu'en electromagnetisme : le nom du dossier sous
`travaux/SignauxSystemes/` designe le jeu, et chaque jeu prefixe ses
macros pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.signaux.application import (
    macros_lab_01_circuit,
    macros_lab_01_poles,
    macros_lab_01_signaux,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _jeu_du_tp_un() -> dict[str, str]:
    """Le TP 1 : convolution, Fourier, un circuit, deux systemes."""
    return {
        **macros_lab_01_signaux.macros_des_signaux(),
        **macros_lab_01_circuit.macros_du_circuit(),
        **macros_lab_01_poles.macros_des_poles(),
    }


JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "lab-01": _jeu_du_tp_un,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
