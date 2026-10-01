"""Commande : publier les resultats du TP 1 sous forme de macros LaTeX.

Cote commande du CQRS : elle produit un effet -- l'ecriture du fichier --
et ne renvoie aucun resultat de lecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.thermique.application.macros_charbon import macros_charbon
from polytech.thermique.application.macros_donnees import (
    macros_donnees_charbon,
    macros_donnees_reservoir,
)
from polytech.thermique.application.macros_reservoir import macros_reservoir

if TYPE_CHECKING:
    from polytech.thermique.domain.ports import PublicationDeMacros


@dataclass(frozen=True, slots=True)
class GenererMacros:
    """Calcule les resultats du TP 1 puis les publie via le port fourni."""

    publication: PublicationDeMacros

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(self.macros_publiees())

    @staticmethod
    def macros_publiees() -> dict[str, str]:
        """Toutes les macros du TP 1, donnees et resultats confondus.

        Une seule fonction les rassemble : le document et les tests lisent
        donc exactement le meme jeu, et une macro oubliee d'un cote se voit
        de l'autre.
        """
        return {
            **macros_donnees_charbon(),
            **macros_charbon(),
            **macros_donnees_reservoir(),
            **macros_reservoir(),
        }
