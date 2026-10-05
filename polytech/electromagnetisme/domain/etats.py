"""Les trois etats entre lesquels le cours d'electromagnetisme se partage.

Le chapitre I des planches : dans les etats **statiques** les charges
sont au repos ; dans les etats **stationnaires** elles sont en mouvement
mais le courant est constant ; dans les etats **variables** le courant
est quelconque. Chaque partie du cours, et du syllabus d'electricite
generale, porte l'un de ces noms, et chaque planche d'applications en
donne quatre exemples.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Etat(Enum):
    """La partie du cours dont un systeme releve."""

    STATIQUE = "statique"
    STATIONNAIRE = "stationnaire"
    VARIABLE = "variable"


def etat_du_systeme(
    *, charges_en_mouvement: bool, courant_constant: bool = True
) -> Etat:
    """L'etat que le mouvement des charges et la constance du courant designent."""
    if not charges_en_mouvement:
        return Etat.STATIQUE
    return Etat.STATIONNAIRE if courant_constant else Etat.VARIABLE


APPLICATIONS: Final = {
    Etat.STATIQUE: (
        "Protection contre la foudre",
        "Separateurs electrostatiques",
        "Imprimantes LASER",
        "Pistolets electrostatiques",
    ),
    Etat.STATIONNAIRE: (
        "Moteurs a courant continu",
        "Levage magnetique",
        "Trains magnetiques",
        "Imagerie medicale (IRM)",
    ),
    Etat.VARIABLE: (
        "Alternateurs et moteurs a courant alternatif",
        "Antennes et fibres optiques",
        "Circuits et filtrage de signaux",
        "Transformateurs",
    ),
}
"""Planches 10, 12 et 14 : quatre applications par etat."""
