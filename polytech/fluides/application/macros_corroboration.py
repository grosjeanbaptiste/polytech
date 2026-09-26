"""Macros du recoupement numerique de la partie II.

La partie I se compare a la reponse publiee par le cours ; la partie II
n'avait aucun recoupement. Ces macros publient les ecarts entre la
resolution symbolique et une integration de Runge-Kutta, methode
independante.
"""

from __future__ import annotations

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.corroboration import (
    ecart_a_la_trajectoire_symbolique,
    ecart_aux_lignes_de_courant,
    variation_de_l_invariant,
)

DEPART = (1.5, 2.0)
"""Point de lacher arbitraire : l'ecart ne doit dependre d'aucun choix."""

DUREE = 0.6
PAS_EN_TEMPS = 1e-3
INSTANT_FIGE = 2.0
DEPART_LIGNE = (1.0, 0.5)
ARRIVEE_LIGNE = 3.0
PAS_EN_ESPACE = 1e-4


def _notation_scientifique(valeur: float) -> str:
    """Rend un ecart minuscule sous une forme lisible par siunitx."""
    return f"{valeur:.1e}"


def macros_corroboration() -> dict[str, str]:
    """Rend les ecarts entre les deux methodes, symbolique et numerique."""
    return {
        "EcartTrajectoireC": _notation_scientifique(
            ecart_a_la_trajectoire_symbolique(
                enonce.CHAMP_C,
                DEPART,
                DUREE,
                PAS_EN_TEMPS,
            ),
        ),
        "EcartInvariantC": _notation_scientifique(
            variation_de_l_invariant(enonce.CHAMP_C, DEPART, DUREE, PAS_EN_TEMPS),
        ),
        "EcartLignesCourantB": _notation_scientifique(
            ecart_aux_lignes_de_courant(
                enonce.CHAMP_B,
                instant=INSTANT_FIGE,
                depart=DEPART_LIGNE,
                arrivee=ARRIVEE_LIGNE,
                pas=PAS_EN_ESPACE,
            ),
        ),
        "PasRungeKutta": _notation_scientifique(PAS_EN_TEMPS),
        "InstantFige": f"{INSTANT_FIGE:.0f}",
    }
