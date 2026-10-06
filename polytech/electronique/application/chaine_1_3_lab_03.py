"""Laboratoire 3 d'electronique, exercice 1.3 : quarante-cinq signaux lents.

5 mV de crete entre 1 et 500 Hz, sur un decalage continu de 100 a 250 mV,
depuis des capteurs de 1 kohm. Le budget de bruit a l'entree est de
(5 uV)^2 ; chaque bloc pose en retire sa part, et le suivant herite du
reste -- c'est la methode du corrige officiel.
"""

from __future__ import annotations

import itertools
import math

from polytech.electronique.application.donnees_lab_03 import UnTrois as D
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.chaine_d_acquisition import (
    Etage,
    bruit_efficace,
    puissances_a_l_entree,
)


def bits_minimaux() -> int:
    """Un pas de 2 uV sur les 10 mV d'excursion du signal."""
    return bits_necessaires(2 * D.SIGNAL, D.PAS)


def convertisseurs_possibles() -> tuple[str, ...]:
    """Ceux qui ont assez de bits."""
    return tuple(
        n for n, (bits, _) in D.CONVERTISSEURS.items() if bits >= bits_minimaux()
    )


def cadence_par_voie() -> float:
    """Nyquist sur 500 Hz."""
    return nyquist(D.BANDE)


def cadence_totale() -> float:
    """Ce qu'un convertisseur unique aurait a tenir pour toutes les voies."""
    return cadence_du_multiplexeur(cadence_par_voie(), D.VOIES)


def multiplexeurs() -> tuple[int, ...]:
    """Le moins de multiplexeurs possible, puis le moins de canaux : 32 et 16."""
    tailles = sorted(D.CANAUX, reverse=True)
    nombre = math.ceil(D.VOIES / tailles[0])
    combinaisons = itertools.combinations_with_replacement(tailles, nombre)
    return min((c for c in combinaisons if sum(c) >= D.VOIES), key=sum)


def cadence_du_plus_grand_multiplexeur() -> float:
    """La cadence du convertisseur qui suit le multiplexeur le plus charge."""
    return cadence_du_multiplexeur(cadence_par_voie(), max(multiplexeurs()))


def amplificateurs_alimentables() -> tuple[str, ...]:
    """Ceux dont la plage d'alimentation contient les 5 V imposes."""
    return tuple(
        nom
        for nom, (_, _, (bas, haut), _) in D.AMPLIFICATEURS.items()
        if bas <= D.PLEINE_ECHELLE <= haut
    )


def premiers_amplificateurs_possibles() -> tuple[str, ...]:
    """Alimentables, et dont le bruit seul n'epuise pas le budget."""
    return tuple(
        nom
        for nom in amplificateurs_alimentables()
        if D.AMPLIFICATEURS[nom][0] ** 2 < D.BUDGET
    )


def amplificateur_retenu() -> str:
    """Le premier, dans l'ordre de priorite de l'enonce."""
    possibles = premiers_amplificateurs_possibles()
    return next(nom for nom in D.PRIORITE if nom in possibles)


def pire_bruit_en_courant() -> float:
    """Le plus grand bruit en courant, converti en tension dans le capteur."""
    pire = max(courant for _, courant, _, _ in D.AMPLIFICATEURS.values())
    return puissance_de_bruit_en_courant(pire, D.SOURCE, D.BANDE)


def bruit_efficace_en_tete(densite: float) -> float:
    """La tension efficace qu'un bloc de cette densite ajouterait en tete."""
    return densite * math.sqrt(D.BANDE)


def gains() -> tuple[float, float]:
    """Avant le passe-haut, limite par le decalage ; apres, ce qui manque."""
    premier = gain_de_chaine(D.PLEINE_ECHELLE, D.DECALAGE)
    return premier, gain_de_chaine(D.PLEINE_ECHELLE, D.SIGNAL) / premier


def produit_gain_bande_requis() -> float:
    """Ce que le premier etage demande a son amplificateur."""
    return gains()[0] * D.BANDE


def _chaine(multiplexeurs_en_cascade: int) -> tuple[Etage, ...]:
    premier, second = gains()
    ampli = D.AMPLIFICATEURS[amplificateur_retenu()][0] / math.sqrt(D.BANDE)
    nom = amplificateur_retenu()
    blocs = [
        Etage(nom, premier, ampli),
        Etage("passe-haut", 1.0, D.PASSE_HAUT),
        Etage(f"{nom} bis", second, ampli),
        Etage("passe-bas", 1.0, D.PASSE_BAS),
    ]
    blocs += [Etage("MUX", 1.0, D.MULTIPLEXEUR)] * multiplexeurs_en_cascade
    return tuple(blocs)


def puissances() -> dict[str, float]:
    """La puissance de bruit de chaque bloc, rapportee a l'entree."""
    blocs = _chaine(1)
    valeurs = puissances_a_l_entree(blocs, D.BANDE)
    return {bloc.nom: valeur for bloc, valeur in zip(blocs, valeurs, strict=True)}


def budgets() -> list[float]:
    """Le budget avant chaque bloc, puis ce qui reste apres le dernier."""
    restes = [D.BUDGET]
    for puissance in puissances().values():
        restes.append(restes[-1] - puissance)
    return restes


def bruit_du_passe_bas_en_second() -> float:
    """Ce qu'il couterait juste apres le premier amplificateur."""
    return D.PASSE_BAS**2 * D.BANDE / gains()[0] ** 2


def bruit_total(*, multiplexeurs_en_cascade: int = 1) -> float:
    """La tension efficace de bruit de la chaine, a l'entree."""
    blocs = _chaine(multiplexeurs_en_cascade)
    return bruit_efficace(puissances_a_l_entree(blocs, D.BANDE))
