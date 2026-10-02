"""Ports du domaine electromagnetique (architecture hexagonale).

Declares ici, dans le domaine ; les adaptateurs qui les implementent
vivent dans `electromagnetisme.infrastructure` et dependent de ce module
-- jamais l'inverse.

Le port est **duplique** depuis la thermique plutot que partage, comme
Thevenin l'est deja entre l'electromagnetisme et l'electronique : un
contexte borne qui importerait le domaine d'un autre cesserait d'etre
borne. Le prix est de dix lignes ; le gain est qu'aucune des matieres ne
peut casser l'autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping


class PublicationDeMacros(Protocol):
    """Port de sortie : publier un jeu de macros a destination du document."""

    def publier(self, macros: Mapping[str, str]) -> None:
        """Publie les macros ; leve si la publication echoue."""
        ...
