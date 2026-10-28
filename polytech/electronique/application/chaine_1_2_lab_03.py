"""Laboratoire 3 d'electronique, exercice 1.2 : huit microphones.

Huit signaux de 1 mV de crete, de 0 a 20 kHz, issus de capteurs de 32 ohm.
Le bruit rapporte a l'entree ne doit pas depasser 2 uV efficaces -- un
budget de 4 uV^2 que chaque bloc entame --, et le pas du convertisseur,
rapporte a l'entree, 1 uV.
"""

from __future__ import annotations

from polytech.electronique.application.donnees_lab_03 import UnDeux as D
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import gain_maximal_par_etage


def gain() -> float:
    """1 mV de crete porte a la pleine echelle de 5 V."""
    return gain_de_chaine(D.PLEINE_ECHELLE, D.SIGNAL)


def cadence_par_voie() -> float:
    """Nyquist sur 20 kHz."""
    return nyquist(D.BANDE)


def cadence_totale() -> float:
    """Ce qu'un convertisseur unique doit tenir derriere le multiplexeur."""
    return cadence_du_multiplexeur(cadence_par_voie(), D.VOIES)


def bits_minimaux() -> int:
    """Le nombre de bits pour un pas de 1 uV sur les 2 mV du signal."""
    return bits_necessaires(2 * D.SIGNAL, D.PAS)


def convertisseurs_assez_fins() -> tuple[str, ...]:
    """Ceux qui ont assez de bits."""
    return tuple(
        n for n, (bits, _, _) in D.CONVERTISSEURS.items() if bits >= bits_minimaux()
    )


def cadence_disponible(nom: str) -> float:
    """La cadence cumulee de tout le stock de ce convertisseur."""
    _, cadence, stock = D.CONVERTISSEURS[nom]
    return cadence * stock


def convertisseurs_possibles() -> tuple[str, ...]:
    """Assez fins, et assez rapides pour les huit voies avec leur stock."""
    return tuple(
        nom
        for nom in convertisseurs_assez_fins()
        if cadence_disponible(nom) >= cadence_totale()
    )


def bruit_du_multiplexeur(nom: str) -> float:
    """Sa puissance de bruit rapportee a l'entree, tout le gain etant devant lui."""
    return D.MULTIPLEXEURS[nom][0] ** 2 * D.BANDE / gain() ** 2


def multiplexeur_retenu() -> str:
    """Le moins cher de ceux qui laissent du budget."""
    possibles = [n for n in D.MULTIPLEXEURS if bruit_du_multiplexeur(n) < D.BUDGET]
    return min(possibles, key=lambda nom: D.MULTIPLEXEURS[nom][1])


def bruit_d_un_bloc_en_tete(densite: float) -> float:
    """La puissance qu'un bloc de cette densite ajoute s'il est le premier."""
    return densite**2 * D.BANDE


def pire_courant_de_bruit() -> float:
    """Le plus grand bruit en courant des amplificateurs, en A/racine(Hz)."""
    return max(courant for _, courant, _, _ in D.AMPLIFICATEURS.values())


def pire_bruit_en_courant() -> float:
    """Le plus grand bruit en courant, converti en tension dans la source."""
    return puissance_de_bruit_en_courant(pire_courant_de_bruit(), D.SOURCE, D.BANDE)


def bruits_des_amplificateurs() -> dict[str, float]:
    """La puissance de bruit en tension de chacun, a sa propre entree."""
    return {nom: efficace**2 for nom, (efficace, _, _, _) in D.AMPLIFICATEURS.items()}


def premiers_amplificateurs_possibles() -> tuple[str, ...]:
    """Ceux qui, places en tete, n'epuisent pas le budget a eux seuls."""
    return tuple(n for n, p in bruits_des_amplificateurs().items() if p < D.BUDGET)


def budget_restant() -> float:
    """Ce qu'il reste apres l'AMP3 de tete."""
    return D.BUDGET - bruits_des_amplificateurs()["AMP3"]


def gain_maximal_du_premier() -> float:
    """Le gain que l'AMP3 tient sur 20 kHz."""
    return gain_maximal_par_etage(D.AMPLIFICATEURS["AMP3"][2], D.BANDE)
