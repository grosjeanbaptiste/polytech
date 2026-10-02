"""Commande : publier les macros d'un corrige sous forme de fragment LaTeX.

Cote commande du CQRS : elle produit un effet -- l'ecriture du fichier --
et ne renvoie aucun resultat de lecture.

Le jeu de macros est **passe en parametre** et non choisi ici : la
campagne en comptera vingt et un, et un seul jeu commun ferait porter a
chaque `resultats.tex` les macros des vingt autres exercices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.electromagnetisme.application.registre import macros_du_travail

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.ports import PublicationDeMacros


@dataclass(frozen=True, slots=True)
class GenererMacros:
    """Calcule les macros d'un travail puis les publie via le port fourni."""

    publication: PublicationDeMacros
    travail: str

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(macros_du_travail(self.travail))
