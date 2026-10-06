"""Commande : publier les macros d'un corrige sous forme de fragment LaTeX.

Cote commande du CQRS : elle produit un effet -- l'ecriture du fichier --
et ne renvoie aucun resultat de lecture. Le jeu est designe par le nom
du dossier de travail, comme dans les autres matieres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.maths.application.registre import macros_du_travail

if TYPE_CHECKING:
    from polytech.maths.domain.ports import PublicationDeMacros


@dataclass(frozen=True, slots=True)
class GenererMacros:
    """Calcule les macros d'un travail puis les publie via le port fourni."""

    publication: PublicationDeMacros
    travail: str

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(macros_du_travail(self.travail))
