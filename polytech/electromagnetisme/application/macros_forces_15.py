"""Macros de l'exercice 15 : force d'arrachement de deux demi-tores.

Au-dela des cinq valeurs de l'enonce, trois publications portent le
raisonnement du document.

Le **rapport des deux champs** -- dix-huit mille -- explique d'un nombre
pourquoi l'air coute si cher. Le **partage des ampere-tours** en
pourcentages le chiffre : deux dixiemes de millimetre d'air contre un
metre de fer, et l'air prend les quatre cinquiemes. Le corrige officiel
ne releve ni l'un ni l'autre.

Et la **masse equivalente** rend la force lisible : huit cent quinze
newtons ne disent rien, quatre-vingt-trois kilogrammes disent tout.
"""

from __future__ import annotations

from typing import Final

from polytech.electromagnetisme.application.formatage import nombre, scientifique
from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    CircuitAEntrefers,
    champ_dans_l_entrefer,
    force_a_flux_constant_sur_un_entrefer,
)
from polytech.fluides.domain.grandeurs import PESANTEUR

INDUCTION: Final = 1.6
SECTION: Final = 4e-4
CHAMP_DANS_LE_FER: Final = 70.0
SPIRES: Final = 500
LONGUEUR_DU_FER: Final = 2 * 50e-2
EPAISSEUR: Final = 0.1e-3
ENTREFERS: Final = 2


def _circuit(epaisseur: float = EPAISSEUR) -> CircuitAEntrefers:
    """Le circuit de l'enonce, l'epaisseur d'entrefer restant variable."""
    return CircuitAEntrefers(
        champ_dans_le_fer=CHAMP_DANS_LE_FER,
        longueur_du_fer=LONGUEUR_DU_FER,
        induction=INDUCTION,
        epaisseur=epaisseur,
        entrefers=ENTREFERS,
    )


def macros_forces_quinze() -> dict[str, str]:
    """Les cinq valeurs de l'enonce, et les trois lectures du resultat."""
    circuit = _circuit()
    champ = champ_dans_l_entrefer(INDUCTION)
    flux = INDUCTION * SECTION
    courant = circuit.courant_d_excitation(SPIRES)
    force = force_a_flux_constant_sur_un_entrefer(INDUCTION, SECTION)
    return {
        "QuinzeInduction": nombre(INDUCTION, 1),
        "QuinzeSection": nombre(SECTION * 1e4, 0),
        "QuinzeChampDansLeFer": nombre(CHAMP_DANS_LE_FER, 0),
        "QuinzeSpires": nombre(SPIRES, 0),
        "QuinzeLongueurDuFer": nombre(LONGUEUR_DU_FER * 100, 0),
        "QuinzeEpaisseur": nombre(EPAISSEUR * 1e3, 1),
        "QuinzeEntrefers": nombre(ENTREFERS, 0),
        "QuinzeFlux": scientifique(flux, 2),
        "QuinzeFluxEnMilliwebers": nombre(flux * 1e3, 2),
        "QuinzeChampDansLEntrefer": nombre(champ, 0),
        "QuinzeRapportDesChamps": nombre(champ / CHAMP_DANS_LE_FER, 0),
        "QuinzeAmperesToursDuFer": nombre(circuit.amperes_tours_du_fer(), 1),
        "QuinzeAmperesToursDesEntrefers": nombre(
            circuit.amperes_tours_des_entrefers(),
            1,
        ),
        "QuinzeAmperesTours": nombre(circuit.amperes_tours(), 1),
        "QuinzePartDesEntrefers": nombre(circuit.part_des_entrefers() * 100, 1),
        "QuinzePartDuFer": nombre((1 - circuit.part_des_entrefers()) * 100, 1),
        # Publiee comme **somme** des deux parts, et non comme « 100 » :
        # la table de l'etape 4 verifie ainsi que les deux pourcentages se
        # completent, au lieu de l'affirmer. Un arrondi qui les ferait
        # tomber a 99,9 se verrait dans le document.
        "QuinzePartTotale": nombre(
            circuit.part_des_entrefers() * 100
            + (1 - circuit.part_des_entrefers()) * 100,
            1,
        ),
        "QuinzePartADeuxDixiemes": nombre(
            _circuit(2 * EPAISSEUR).part_des_entrefers() * 100,
            1,
        ),
        "QuinzeCourant": nombre(courant, 2),
        "QuinzeForce": nombre(force, 0),
        "QuinzeForceEnKilogrammes": nombre(force / PESANTEUR, 0),
        "QuinzeForceADoubleInduction": nombre(
            force_a_flux_constant_sur_un_entrefer(2 * INDUCTION, SECTION),
            0,
        ),
    }
