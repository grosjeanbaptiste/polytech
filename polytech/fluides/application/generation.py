"""Commande : publier les resultats sous forme de macros LaTeX.

Cote commande du CQRS -- elle produit un effet (l'ecriture des macros) et ne
renvoie aucun resultat de lecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.application.activite import ResultatsDeLActivite
from polytech.fluides.application.macros_cinematique import macros_cinematique
from polytech.fluides.application.macros_corroboration import macros_corroboration
from polytech.fluides.application.macros_definitions import macros_definitions
from polytech.fluides.application.macros_derivations import macros_derivations
from polytech.fluides.application.macros_donnees import macros_donnees
from polytech.fluides.application.macros_hypotheses import macros_hypotheses
from polytech.fluides.application.macros_incertitude import macros_incertitude
from polytech.fluides.application.macros_numeriques import (
    macros_des_champs,
    macros_du_manometre,
)
from polytech.fluides.application.macros_statique import macros_statique
from polytech.fluides.application.macros_validite import macros_validite

if TYPE_CHECKING:
    from polytech.fluides.domain.ports import (
        PublicationDApercu,
        PublicationDeMacros,
    )


@dataclass(frozen=True, slots=True)
class GenererApercu:
    """Compose l'apercu des memes macros que publie `GenererMacros`."""

    publication: PublicationDApercu

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(GenererMacros.macros_publiees())


@dataclass(frozen=True, slots=True)
class GenererMacros:
    """Calcule les resultats puis les publie via le port fourni."""

    publication: PublicationDeMacros

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(self.macros_publiees())

    @staticmethod
    def macros_publiees() -> dict[str, str]:
        """Assemble les macros des differents modules et des resultats."""
        resultats = ResultatsDeLActivite().executer()
        return {
            **macros_definitions(),
            **macros_donnees(),
            **macros_derivations(),
            **macros_hypotheses(),
            **macros_incertitude(),
            **macros_statique(),
            **macros_validite(),
            **macros_cinematique(),
            **macros_corroboration(),
            **macros_du_manometre(resultats),
            **macros_des_champs(resultats),
        }
