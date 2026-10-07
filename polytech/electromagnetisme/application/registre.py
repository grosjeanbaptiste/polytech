"""Quel jeu de macros appartient a quel dossier de travail.

La thermique et les fluides n'ont qu'un travail chacun : un seul jeu de
macros y suffisait. L'electromagnetisme en aura vingt et un, et un jeu
commun aurait deux defauts.

Chaque `resultats.tex` porterait les macros des vingt autres exercices,
donc le test « aucune macro publiee ne reste inutilisee » deviendrait
intenable. Et une erreur dans un exercice ferait echouer la generation
des vingt autres.

D'ou ce registre, et la convention de nommage qui va avec : les macros
d'un exercice portent son numero en prefixe -- `Huit...` pour
l'exercice 8 -- si bien que deux exercices ne peuvent pas se melanger.
Un test le verifie plutot que de compter sur la discipline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

from polytech.electromagnetisme.application import (
    macros_champ_06,
    macros_champ_07,
    macros_champ_08,
    macros_champ_12,
    macros_exercice_06,
    macros_exercice_08,
    macros_forces_06,
    macros_forces_11,
    macros_forces_15,
    macros_forces_16,
    macros_labo_01,
    macros_labo_02,
    macros_labo_03,
    macros_seance_21_autres,
    macros_seance_24,
    macros_seance_31,
    macros_seance_32,
    macros_statique_05,
    macros_statique_07,
    macros_statique_08,
    macros_statique_11,
    macros_statique_12,
    macros_thevenin,
    macros_transitoire_05,
)
from polytech.electromagnetisme.application.registre_des_interrogations import (
    JEUX_DES_INTERROGATIONS,
)


def _jeu_thevenin_norton() -> dict[str, str]:
    """L'exercice 1 : equivalences Thevenin / Norton."""
    return {**macros_thevenin.macros_donnees(), **macros_thevenin.macros_thevenin()}


def _jeu_exercice_huit() -> dict[str, str]:
    """L'exercice 8 : Thevenin avec source de courant dans la maille."""
    return {
        **macros_exercice_08.macros_donnees_huit(),
        **macros_exercice_08.macros_exercice_huit(),
    }


def _jeu_exercice_six() -> dict[str, str]:
    """L'exercice 6 : l'inductance d'un circuit magnetique a deux branches."""
    return {
        **macros_exercice_06.macros_donnees_six(),
        **macros_exercice_06.macros_exercice_six(),
    }


def _jeu_forces_six() -> dict[str, str]:
    """L'exercice 6 du chapitre IV : force sur une armature de condensateur."""
    return macros_forces_06.macros_forces_six()


JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    **JEUX_DES_INTERROGATIONS,
    "exs-04-forces-06": _jeu_forces_six,
    "exs-04-forces-11": macros_forces_11.macros_forces_onze,
    "exs-04-forces-15": macros_forces_15.macros_forces_quinze,
    "exs-04-forces-16": macros_forces_16.macros_forces_seize,
    "exs-02-statique-05": macros_statique_05.macros_statique_cinq,
    "exs-02-statique-07": macros_statique_07.macros_statique_sept,
    "exs-02-statique-08": macros_statique_08.macros_statique_huit,
    "exs-02-statique-1-1-poisson": macros_statique_11.macros_statique_un_un,
    "exs-02-statique-12": macros_statique_12.macros_statique_douze,
    "exs-03-champ-magnetique-06": macros_champ_06.macros_champ_six,
    "exs-03-champ-magnetique-07": macros_champ_07.macros_champ_sept,
    "exs-03-champ-magnetique-08": macros_champ_08.macros_champ_huit,
    "exs-03-champ-magnetique-12": macros_champ_12.macros_champ_douze,
    "exs-03-circuits-magnetiques-06": _jeu_exercice_six,
    "exs-03-stationnaire-2-1": macros_seance_21_autres.macros_de_la_seance_deux_un,
    "exs-03-stationnaire-2-4": macros_seance_24.macros_de_la_seance_deux_quatre,
    "exs-04-variable-3-1": macros_seance_31.macros_de_la_seance_trois_un,
    "exs-04-variable-3-2": macros_seance_32.macros_de_la_seance_trois_deux,
    "lab-01-mesures-electriques": macros_labo_01.macros_du_labo_un,
    "lab-02-circuit-rc": macros_labo_02.macros_du_labo_deux,
    "lab-03-simulateur-de-circuit": macros_labo_03.macros_du_labo_trois,
    "exs-04-transitoire-05": macros_transitoire_05.macros_transitoire_cinq,
    "exs-03-thevenin-norton-01": _jeu_thevenin_norton,
    "exs-03-thevenin-08": _jeu_exercice_huit,
}
"""Le nom du dossier de travail designe le jeu qui l'alimente."""


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail, et de lui seul."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
