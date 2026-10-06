"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention qu'en electromagnetisme : le nom du dossier sous
`travaux/SignauxSystemes/` designe le jeu, et chaque jeu prefixe ses
macros pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.signaux.application import (
    macros_convolution_graphique,
    macros_cuves_boucle,
    macros_cuves_modele,
    macros_exs_03,
    macros_exs_06_black,
    macros_exs_06_bode,
    macros_exs_07,
    macros_exs_08_black,
    macros_exs_08_regimes,
    macros_exs_45,
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


def _jeu_de_l_analyse_harmonique() -> dict[str, str]:
    """La feuille « Analyse harmonique » : Bode d'un cote, Black de l'autre."""
    return {
        **macros_exs_06_bode.macros_de_bode(),
        **macros_exs_06_black.macros_de_black(),
    }


def _jeu_des_performances() -> dict[str, str]:
    """La feuille « Performances » : les regimes et marges, puis les lectures de `M`."""
    return {
        **macros_exs_08_regimes.macros_des_regimes(),
        **macros_exs_08_black.macros_de_black(),
    }


def _jeu_des_cuves() -> dict[str, str]:
    """Le TP « Reservoir » : le modele, la boucle ouverte, les boucles fermees."""
    return {
        **macros_cuves_modele.macros_du_modele(),
        **macros_cuves_boucle.macros_des_boucles(),
    }


JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "lab-01": _jeu_du_tp_un,
    "exs-03-systemes": macros_exs_03.macros_des_systemes,
    "exs-06-analyse-harmonique": _jeu_de_l_analyse_harmonique,
    "exs-07-stabilite": macros_exs_07.macros_de_la_stabilite,
    "exs-08-performances": _jeu_des_performances,
    "exs-04-05-resolus": macros_exs_45.macros_de_la_seance,
    "exs-convolution-graphique": macros_convolution_graphique.macros_de_la_convolution,
    "tp-reservoir": _jeu_des_cuves,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
