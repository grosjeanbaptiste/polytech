"""Ports du domaine (architecture hexagonale).

Les ports sont declares ici, dans le domaine ; les adaptateurs qui les
implementent vivent dans ``fluides.infrastructure`` et dependent de ce
module -- jamais l'inverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polytech.fluides.domain.cinematique.echantillonnage import Fleche


class PublicationDeMacros(Protocol):
    """Port de sortie : publier un jeu de macros a destination du document."""

    def publier(self, macros: Mapping[str, str]) -> None:
        """Publie les macros ; leve si la publication echoue."""
        ...


class PublicationDeTable(Protocol):
    """Port de sortie : publier une table de fleches pour le trace."""

    def publier(self, fleches: Sequence[Fleche]) -> None:
        """Publie la table ; leve si la publication echoue."""
        ...
