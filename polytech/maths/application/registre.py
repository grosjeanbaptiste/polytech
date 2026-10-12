"""Quel jeu de macros appartient a quel dossier de travail.

Meme convention que dans les autres matieres : le nom du dossier sous
`travaux/Mathematique/` designe le jeu, et chaque jeu prefixe ses macros
pour qu'aucun ne se melange a un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.maths.application import (
    exercices_des_fascicules,
    macros_examen_2021,
    macros_examen_2022,
    macros_examen_2025,
    macros_exs_01,
    macros_exs_02,
    macros_recueil_algebre_ch01,
    macros_recueil_algebre_ch02,
    macros_recueil_algebre_ch03,
    macros_recueil_algebre_ch04,
    macros_recueil_algebre_ch05,
    macros_recueil_algebre_ch06,
    macros_recueil_algebre_ch07,
    macros_recueil_algebre_ch08a,
    macros_recueil_algebre_ch08b,
    macros_recueil_algebre_ch08c,
    macros_recueil_algebre_ch09,
    macros_recueil_algebre_ch10_reflexions,
)

if TYPE_CHECKING:
    from collections.abc import Callable

JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "algebre-01-nombres-complexes": macros_recueil_algebre_ch01.macros_du_chapitre_un,
    "algebre-02-matrices": macros_recueil_algebre_ch02.macros_du_chapitre_deux,
    "algebre-04-polynomes": macros_recueil_algebre_ch04.macros_du_chapitre_quatre,
    "algebre-05-determinants": macros_recueil_algebre_ch05.macros_du_chapitre_cinq,
    "algebre-06-rang": macros_recueil_algebre_ch06.macros_du_chapitre_six,
    "algebre-09-espaces-metriques": (
        macros_recueil_algebre_ch09.macros_du_chapitre_neuf
    ),
    "exercices-des-fascicules": (
        exercices_des_fascicules.macros_des_exercices_des_fascicules
    ),
    "algebre-10-valeurs-propres": (
        macros_recueil_algebre_ch10_reflexions.macros_du_chapitre_dix
    ),
    "algebre-08c-examens": macros_recueil_algebre_ch08c.macros_de_la_partie_c,
    "algebre-08b-matrices-associees": (
        macros_recueil_algebre_ch08b.macros_de_la_partie_b
    ),
    "algebre-08a-applications-lineaires": (
        macros_recueil_algebre_ch08a.macros_de_la_partie_a
    ),
    "algebre-07-espaces-vectoriels": (
        macros_recueil_algebre_ch07.macros_du_chapitre_sept
    ),
    "algebre-03-systemes-lineaires": (
        macros_recueil_algebre_ch03.macros_du_chapitre_trois
    ),
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
