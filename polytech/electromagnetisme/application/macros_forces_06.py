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

S'y ajoute l'ecart relatif entre la force moyenne -- ce que le quotient
de l'enonce donne -- et la force au depart, quand c'est l'ecartement qui
varie : sous forme litterale, puis chiffre pour les dix millimetres que
donne le recueil -- au cas ou le condensateur serait plan, ce qu'il n'est
pas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
    scientifique,
)
from polytech.electromagnetisme.domain.energie_et_forces import force_sur_une_armature
from polytech.electromagnetisme.domain.force_moyenne import (
    ecart_relatif_sur_l_ecartement,
)
from polytech.fluides.domain.grandeurs import PESANTEUR

TENSION: Final = 10e3
"""`U`, en volts. Donnee de l'enonce."""

VARIATION_DE_CAPACITE: Final = 4e-12
"""`|Delta C|`, en farads : quatre picofarads."""

DEPLACEMENT: Final = 1e-3
"""`Delta x`, en metres : un millimetre."""

"""Pour convertir la force en une masse equivalente, en m/s^2."""

ECARTEMENT: Final = sp.Symbol("d", positive=True)
"""La distance entre armatures, sous le nom que le recueil lui donne."""

DISTANCE_INITIALE: Final = sp.Rational(10, 1000)
"""`d = 10 mm`, donnee de l'enonce du recueil -- que le manuscrit n'emploie pas."""

PAS: Final = sp.Symbol(r"\Delta x", positive=True)
"""Le deplacement, sous le nom que l'enonce lui donne."""


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
        "ForceSixEcartRelatifSurLEcartement": expression_latex(
            ecart_relatif_sur_l_ecartement(ECARTEMENT, PAS),
        ),
        "ForceSixDistanceEnMillimetres": nombre(float(DISTANCE_INITIALE) * 1e3, 0),
        "ForceSixEcartSiPlan": nombre(
            float(
                ecart_relatif_sur_l_ecartement(
                    DISTANCE_INITIALE,
                    sp.Rational(DEPLACEMENT).limit_denominator(10**6),
                ),
            )
            * 100,
            1,
        ),
    }
