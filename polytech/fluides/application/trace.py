"""Commande : publier le champ de vitesse echantillonne pour la figure.

Cote commande du CQRS, comme :mod:`fluides.application.generation` : elle
produit un effet (l'ecriture de la table) et ne renvoie rien.

La grille reprend le cadrage impose par l'enonce -- « le quadrant superieur
droit de x = 0 a 5 et de y = 0 a 6 » -- en evitant l'origine, ou la vitesse
est nulle et la direction indefinie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.echantillonnage import (
    GrilleDeTrace,
    fleches_du_champ,
)

if TYPE_CHECKING:
    from polytech.fluides.domain.ports import PublicationDeTable

GRILLE: GrilleDeTrace = GrilleDeTrace(
    abscisses=(0.5, 1.5, 2.5, 3.5, 4.5),
    ordonnees=(0.6, 1.8, 3.0, 4.2, 5.4),
    longueur=0.55,
)


@dataclass(frozen=True, slots=True)
class GenererTableDuChamp:
    """Echantillonne l'ecoulement C puis publie la table via le port fourni."""

    publication: PublicationDeTable

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(fleches_du_champ(enonce.CHAMP_C, GRILLE))
