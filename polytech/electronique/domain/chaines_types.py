"""Les cinq architectures types d'une chaine d'instrumentation a `m` capteurs.

Chapitre 4, partie III, planches 45 a 50, de la plus chere a la plus
simple :

1. par capteur : amplificateur a faible bruit, filtres, amplificateur
   d'usage general, filtres, convertisseur ;
2. sans le second bloc de filtres, si le bruit du second amplificateur
   est sans consequence ;
3. sans amplificateur d'usage general, si le premier suffit a approcher
   la pleine echelle ;
4. les convertisseurs mutualises par des multiplexeurs, si les cadences
   sont faibles ;
5. l'amplificateur lui-meme apres le multiplexeur, si les signaux sont
   forts et le bruit negligeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from polytech.electronique.domain.multiplexage import (
    paires_multiplexeur_convertisseur,
)


class Architecture(Enum):
    """Les « Design 1 » a « Design 5 » du fascicule."""

    UN = 1
    DEUX = 2
    TROIS = 3
    QUATRE = 4
    CINQ = 5


@dataclass(frozen=True, slots=True)
class Composants:
    """Ce qu'une architecture demande, bloc par bloc."""

    faible_bruit: int
    filtres: int
    usage_general: int
    multiplexeurs: int
    convertisseurs: int


def composants(
    architecture: Architecture, *, capteurs: int, voies_par_multiplexeur: int = 1
) -> Composants:
    """Le nombre de blocs de chaque sorte pour `capteurs` capteurs."""
    if capteurs < 1:
        msg = f"au moins un capteur est attendu : {capteurs}"
        raise ValueError(msg)
    if architecture is Architecture.UN:
        return Composants(capteurs, 2 * capteurs, capteurs, 0, capteurs)
    if architecture is Architecture.DEUX:
        return Composants(capteurs, capteurs, capteurs, 0, capteurs)
    if architecture is Architecture.TROIS:
        return Composants(capteurs, capteurs, 0, 0, capteurs)
    paires = paires_multiplexeur_convertisseur(capteurs, voies_par_multiplexeur)
    if architecture is Architecture.QUATRE:
        return Composants(capteurs, capteurs, 0, paires, paires)
    return Composants(0, capteurs, paires, paires, paires)
