"""Commande : publier les macros d'un travail inscrit au registre.

Pendant de `GenererMacros`, qui reste celle de l'activite 1 : meme port,
mais le jeu est designe par le nom du dossier de travail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.application.registre import macros_du_travail

if TYPE_CHECKING:
    from polytech.fluides.domain.ports import PublicationDeMacros


@dataclass(frozen=True, slots=True)
class GenererMacrosDuTravail:
    """Calcule les macros d'un travail puis les publie via le port fourni."""

    publication: PublicationDeMacros
    travail: str

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(macros_du_travail(self.travail))
