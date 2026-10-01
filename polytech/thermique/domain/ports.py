"""Ports du domaine thermique (architecture hexagonale).

Declares ici, dans le domaine ; les adaptateurs qui les implementent
vivent dans `thermique.infrastructure` et dependent de ce module --
jamais l'inverse.
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
