"""Macros de l'exercice 6 du chapitre IV : force sur une armature.

Les donnees et le resultat, plus **deux grandeurs de comparaison** qui
portent le raisonnement du document.

La premiere est la force a tension doublee. Elle montre la dependance en
`U^2` d'un coup d'oeil, et c'est elle qui explique pourquoi ces forces
sont negligeables a basse tension et redoutables aux dix kilovolts de
l'enonce.

La seconde est le poids d'un objet de meme force, en grammes. Deux
dixiemes de newton ne disent rien a personne ; vingt grammes disent
immediatement si la force compte dans un dimensionnement mecanique.
"""

from __future__ import annotations

from typing import Final

from polytech.electromagnetisme.application.formatage import nombre, scientifique
from polytech.electromagnetisme.domain.energie_et_forces import force_sur_une_armature

TENSION: Final = 10e3
"""`U`, en volts. Donnee de l'enonce."""

VARIATION_DE_CAPACITE: Final = 4e-12
"""`|Delta C|`, en farads : quatre picofarads."""

DEPLACEMENT: Final = 1e-3
"""`Delta x`, en metres : un millimetre."""

PESANTEUR: Final = 9.81
"""Pour convertir la force en une masse equivalente, en m/s^2."""


def macros_forces_six() -> dict[str, str]:
    """Les donnees, le resultat, et les deux grandeurs de comparaison."""
    force = force_sur_une_armature(TENSION, VARIATION_DE_CAPACITE, DEPLACEMENT)
    doublee = force_sur_une_armature(2 * TENSION, VARIATION_DE_CAPACITE, DEPLACEMENT)
    return {
        "ForceSixTension": scientifique(TENSION, 0),
        "ForceSixTensionEnKilovolts": nombre(TENSION / 1e3, 0),
        "ForceSixVariationCapacite": scientifique(VARIATION_DE_CAPACITE, 0),
        "ForceSixVariationEnPicofarads": nombre(VARIATION_DE_CAPACITE * 1e12, 0),
        "ForceSixDeplacement": scientifique(DEPLACEMENT, 0),
        "ForceSixDeplacementEnMillimetres": nombre(DEPLACEMENT * 1e3, 0),
        "ForceSixEnergieNumerateur": scientifique(
            TENSION**2 * VARIATION_DE_CAPACITE / 2,
            1,
        ),
        "ForceSix": nombre(force, 2),
        "ForceSixADoubleTension": nombre(doublee, 2),
        "ForceSixRapportDesTensions": nombre(doublee / force, 0),
        "ForceSixMasseEquivalente": nombre(force / PESANTEUR * 1e3, 1),
        "ForceSixPenteDeCapacite": scientifique(
            VARIATION_DE_CAPACITE / DEPLACEMENT,
            0,
        ),
    }
