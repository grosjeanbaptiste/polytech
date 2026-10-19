"""Les travaux de fluides au-dela de l'activite 1, chacun avec son jeu de macros.

L'activite d'integration 1 garde sa commande d'origine, `GenererMacros`,
et rien n'en est change. Les suivants s'inscrivent ici, comme en thermique
et en electromagnetisme : le nom du dossier sous
`travaux/MecaniqueFluides/` designe le jeu.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.fluides.application import (
    activite_2_bateau,
    macros_activite_2,
    macros_activite_2_complements,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _activite_2() -> dict[str, str]:
    """Les resultats des trois exercices, puis ce qui les eprouve."""
    return {
        **macros_activite_2.macros(),
        **macros_activite_2_complements.macros_complements(),
        **activite_2_bateau.macros_bateau(),
    }


JEUX_DE_MACROS: Final[dict[str, Callable[[], dict[str, str]]]] = {
    "activite-integration-2": _activite_2,
}


def macros_du_travail(slug: str) -> dict[str, str]:
    """Les macros de ce dossier de travail ; leve s'il n'est pas inscrit."""
    if slug not in JEUX_DE_MACROS:
        msg = f"aucun jeu de macros pour le travail {slug!r}"
        raise KeyError(msg)
    return JEUX_DE_MACROS[slug]()
