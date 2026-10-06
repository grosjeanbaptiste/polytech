"""Laboratoire 3 d'electronique, exercice 1.1 : une chaine d'acquisition type.

Ampli A, passe-haut, ampli B, passe-bas, convertisseur. Le signal fait
1 mV de crete entre 100 Hz et 10 kHz, sur un decalage continu qui peut
atteindre 1 V ; tout est alimente sous 5 V, et la resolution visee a
l'entree est de 2 uV.
"""

from __future__ import annotations

from polytech.electronique.application.donnees_lab_03 import UnUn as D
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    gain_de_chaine,
    nyquist,
)
from polytech.electronique.domain.amplification import gain_maximal_par_etage
from polytech.electronique.domain.chaine_d_acquisition import (
    Etage,
    bruit_efficace,
    gain_total,
    puissances_en_sortie,
)


def gain_exact_du_premier_etage() -> float:
    """Le plus grand gain qui garde `decalage + signal` sous la pleine echelle."""
    return D.PLEINE_ECHELLE / (D.DECALAGE + D.SIGNAL)


def gains() -> tuple[float, float, float]:
    """Le gain total, celui de l'ampli A et celui de l'ampli B.

    Le corrige neglige le millivolt de signal devant le volt de decalage
    et prend 5 pour l'ampli A.
    """
    total = gain_de_chaine(D.PLEINE_ECHELLE, D.SIGNAL)
    premier = gain_de_chaine(D.PLEINE_ECHELLE, D.DECALAGE)
    return total, premier, total / premier


def sortie_maximale_du_premier_etage() -> float:
    """Ce que le gain arrondi a 5 demande a l'ampli A, en volts."""
    return gains()[1] * (D.DECALAGE + D.SIGNAL)


def gains_maximaux() -> dict[str, float]:
    """Le gain que chaque amplificateur tient sur la bande du signal."""
    return {
        nom: gain_maximal_par_etage(produit, D.BANDE_DU_SIGNAL)
        for nom, produit in D.PRODUITS_GAIN_BANDE.items()
    }


def amplificateurs_possibles(gain: float) -> tuple[str, ...]:
    """Ceux dont le produit gain-bande suffit a ce gain."""
    return tuple(nom for nom, maximal in gains_maximaux().items() if maximal >= gain)


def pas_a_l_entree(bits: int) -> float:
    """Le pas de quantification, rapporte a l'entree de la chaine."""
    return 2 * D.PLEINE_ECHELLE / float(2**bits - 1) / gains()[0]


def bits_minimaux() -> int:
    """Le nombre de bits qui tient la resolution a l'entree."""
    return bits_necessaires(2 * D.PLEINE_ECHELLE / gains()[0], D.RESOLUTION)


def cadence_minimale() -> float:
    """Nyquist, sur ce que le passe-bas laisse passer."""
    return nyquist(D.BANDE_DU_FILTRE)


def convertisseurs_possibles() -> tuple[str, ...]:
    """Ceux qui ont assez de bits et echantillonnent assez vite."""
    return tuple(
        nom
        for nom, (cadence, bits) in D.CONVERTISSEURS.items()
        if bits >= bits_minimaux() and cadence >= cadence_minimale()
    )


def chaine() -> tuple[Etage, ...]:
    """Les cinq blocs, avec leur gain et leur bruit en tension a l'entree."""
    _, premier, second = gains()
    return (
        Etage("ampli A", premier, 1e-9),
        Etage("passe-haut", 1.0, 10e-9),
        Etage("ampli B", second, 1e-9),
        Etage("passe-bas", 1.0, 100e-9),
        Etage("CAN", 1.0, 1000e-9),
    )


def puissances() -> dict[str, float]:
    """La puissance de bruit de chaque bloc a la sortie, en volts carres."""
    blocs = chaine()
    valeurs = puissances_en_sortie(blocs, D.BANDE_DU_FILTRE)
    return {bloc.nom: valeur for bloc, valeur in zip(blocs, valeurs, strict=True)}


def bruit_total() -> tuple[float, float, float]:
    """La puissance totale, la tension efficace et la densite efficace en sortie."""
    valeurs = list(puissances().values())
    efficace = bruit_efficace(valeurs)
    return sum(valeurs), efficace, efficace / D.BANDE_DU_FILTRE**0.5


def etage_dominant() -> tuple[str, float]:
    """Le bloc qui bruite le plus, et sa part de la puissance totale."""
    valeurs = puissances()
    nom = max(valeurs, key=lambda cle: valeurs[cle])
    return nom, valeurs[nom] / sum(valeurs.values())


def bruit_a_l_entree() -> float:
    """La tension efficace de bruit rapportee a l'entree de la chaine."""
    return bruit_total()[1] / gain_total(chaine())
