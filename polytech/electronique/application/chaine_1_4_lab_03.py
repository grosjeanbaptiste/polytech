"""Laboratoire 3 d'electronique, exercice 1.4 : deux cents capteurs de force.

10 mV de crete de 0 a 100 Hz, depuis des capteurs de 10 kohm. Ici la
resistance de source est grande : c'est le bruit *en courant* des
amplificateurs, converti en tension par ces 10 kohm, qui decide de tout.
"""

from __future__ import annotations

import math

from polytech.electronique.application.donnees_lab_03 import UnQuatre as D
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    gain_de_chaine,
    nyquist,
)
from polytech.electronique.domain.chaine_d_acquisition import bruit_efficace


def cadence_totale() -> float:
    """Deux cents capteurs a 200 Hz chacun."""
    return cadence_du_multiplexeur(nyquist(D.BANDE), D.CAPTEURS)


def cadence_maximale() -> float:
    """La cadence du plus rapide des convertisseurs, en hertz."""
    return max(cadence for cadence, _ in D.CONVERTISSEURS.values())


def nombre_de_convertisseurs() -> int:
    """Combien de convertisseurs, des plus rapides, pour tenir la cadence."""
    return math.ceil(cadence_totale() / cadence_maximale())


def voies_par_convertisseur() -> int:
    """Les capteurs que chaque convertisseur sert."""
    return math.ceil(D.CAPTEURS / nombre_de_convertisseurs())


def multiplexeur(voies: int | None = None) -> str:
    """Le plus petit multiplexeur qui porte ces voies -- celles d'un convertisseur."""
    a_porter = voies_par_convertisseur() if voies is None else voies
    assez = {nom: c for nom, c in D.MULTIPLEXEURS.items() if c >= a_porter}
    return min(assez, key=lambda nom: assez[nom])


def cadence_reelle_par_voie() -> float:
    """Ce que chaque voie recoit d'un convertisseur a sa cadence maximale."""
    return cadence_maximale() / voies_par_convertisseur()


def bande_d_arret() -> float:
    """La moitie de cette cadence : au-dela, le passe-bas doit avoir coupe."""
    return cadence_reelle_par_voie() / 2


def _densite(tension: float, courant: float) -> float:
    """Tension et courant de bruit, celui-ci converti par la source."""
    return math.hypot(tension, D.SOURCE * courant)


def densite_du_filtre_en_tete() -> float:
    """Le bruit du filtre s'il recevait directement le capteur."""
    return _densite(*D.FILTRE)


def bruit_du_filtre_en_tete() -> float:
    """Le meme, en volts efficaces sur la bande du signal."""
    return densite_du_filtre_en_tete() * math.sqrt(D.BANDE)


def densites_des_amplificateurs() -> dict[str, float]:
    """La densite totale de chacun, branche sur le capteur."""
    return {n: _densite(en, i) for n, (en, i, _) in D.AMPLIFICATEURS.items()}


def bruits_des_amplificateurs() -> dict[str, float]:
    """Les memes, en volts efficaces sur la bande du signal."""
    racine = math.sqrt(D.BANDE)
    return {n: d * racine for n, d in densites_des_amplificateurs().items()}


def amplificateurs_possibles() -> tuple[str, ...]:
    """Ceux dont le bruit seul tient sous le budget."""
    return tuple(n for n, b in bruits_des_amplificateurs().items() if b < D.BUDGET)


def gain() -> float:
    """10 mV de crete portes a 5 V."""
    return gain_de_chaine(D.PLEINE_ECHELLE, D.SIGNAL)


def produit_gain_bande_requis() -> float:
    """Un seul etage : ce gain sur la bande du signal."""
    return gain() * D.BANDE


def bits_minimaux() -> int:
    """Un pas de 0,2 mV sur les 20 mV d'excursion."""
    return bits_necessaires(2 * D.SIGNAL, D.PAS)


def convertisseurs_possibles() -> tuple[str, ...]:
    """Assez de bits, et la cadence maximale."""
    return tuple(
        nom
        for nom, (cadence, bits) in D.CONVERTISSEURS.items()
        if bits >= bits_minimaux() and cadence >= cadence_maximale()
    )


def bruits_a_l_entree(
    *, bande_apres_le_filtre: float | None = None
) -> dict[str, float]:
    """La tension efficace de chaque source de bruit, rapportee a l'entree.

    Le multiplexeur et le convertisseur suivent le passe-bas : rien ne
    borne leur bruit a la bande du signal. `bande_apres_le_filtre` dit
    sur quelle bande les compter ; par defaut, celle du signal, comme le
    corrige officiel.
    """
    large = math.sqrt(
        D.BANDE if bande_apres_le_filtre is None else bande_apres_le_filtre
    )
    etroite = math.sqrt(D.BANDE)
    ampli = amplificateurs_possibles()[0]
    return {
        "capteur": D.CAPTEUR * etroite,
        ampli: bruits_des_amplificateurs()[ampli],
        "passe-bas": D.FILTRE[0] * etroite / gain(),
        multiplexeur(): D.MULTIPLEXEUR * large / gain(),
        "CAN": D.CONVERTISSEUR * large / gain(),
    }


def bruit_total(*, bande_apres_le_filtre: float | None = None) -> float:
    """La tension efficace de bruit de la chaine entiere, a l'entree."""
    efficaces = bruits_a_l_entree(bande_apres_le_filtre=bande_apres_le_filtre)
    return bruit_efficace([e**2 for e in efficaces.values()])


def composants() -> dict[str, int]:
    """Le nombre de composants de chaque sorte."""
    return {
        amplificateurs_possibles()[0]: D.CAPTEURS,
        "passe-bas": D.CAPTEURS,
        multiplexeur(): nombre_de_convertisseurs(),
        "CAN": nombre_de_convertisseurs(),
    }
