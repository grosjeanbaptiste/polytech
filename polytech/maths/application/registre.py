"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention que dans les autres matieres : le nom du dossier sous
`travaux/Mathematique/` designe le jeu, et chaque jeu prefixe ses macros
pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.maths.application import (
    macros_examen_2021,
    macros_examen_2022,
    macros_examen_2025,
    macros_exs_01,
    macros_exs_02,
    macros_recueil_algebre_ch01,
)

if TYPE_CHECKING:
    from collections.abc import Callable

JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "algebre-01-nombres-complexes": macros_recueil_algebre_ch01.macros_du_chapitre_un,
    "examen-2021-janvier": macros_examen_2021.macros_de_l_examen_2021,
    "examen-2022-janvier": macros_examen_2022.macros_de_l_examen_2022,
    "examen-2025-janvier": macros_examen_2025.macros_de_l_examen_2025,
    "exs-01-integration-matrices": macros_exs_01.macros_des_exercices_un,
    "exs-02-equations-differentielles": macros_exs_02.macros_des_exercices_deux,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
