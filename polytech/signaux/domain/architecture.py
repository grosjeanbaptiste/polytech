"""L'architecture d'une boucle de controle, avant tout calcul de stabilite.

Separe de `stabilite`, qui porte les **criteres**, parce que ce module
porte la **structure** : les deux grandes questions, la distinction
asservissement / regulation, les deux chaines, et la demarche de
dimensionnement. Le fichier unique depassait la limite de taille du depot.

L'ordre des questions n'est pas indifferent : parler des performances
d'une boucle instable n'a aucun sens, et c'est pourquoi le fascicule
traite la stabilite au chapitre 7 et les performances au 8.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polytech.signaux.domain.transmittance import Transmittance


class NatureDeLaBoucle(Enum):
    """Les deux situations que le fascicule distingue."""

    ASSERVISSEMENT = "asservissement : la consigne varie, la sortie doit suivre"
    REGULATION = "regulation : la consigne est fixe, les perturbations se rejettent"


def questions_de_la_boucle() -> tuple[str, str]:
    """Les « deux grandes questions » du fascicule, dans son ordre.

    L'ordre n'est pas indifferent : parler des performances d'une boucle
    instable n'a aucun sens, et c'est pourquoi la stabilite occupe le
    chapitre 7 et les performances le 8.
    """
    return (
        "la boucle fermee est-elle stable ?",
        "quelles performances la boucle fermee atteint-elle ?",
    )


def nature_de_la_boucle(*, consigne_variable: bool) -> NatureDeLaBoucle:
    """Asservissement si la consigne varie, regulation si elle est fixe.

    La distinction decide des **specifications** : un asservissement se
    juge sur sa capacite a suivre, une regulation sur sa capacite a
    rejeter. Les deux ne demandent pas le meme correcteur.
    """
    if consigne_variable:
        return NatureDeLaBoucle.ASSERVISSEMENT
    return NatureDeLaBoucle.REGULATION


def chaine_directe(blocs: tuple[Transmittance, ...]) -> Transmittance:
    """Le produit des blocs de la chaine directe, du correcteur au procede."""
    if not blocs:
        msg = "chaine directe vide : la boucle n'a rien a corriger"
        raise ValueError(msg)
    produit = blocs[0]
    for bloc in blocs[1:]:
        produit = produit.en_serie_avec(bloc)
    return produit


def chaine_de_reaction(capteur: Transmittance) -> Transmittance:
    """La chaine de retour, qui est le capteur et rien d'autre.

    Le fascicule y consacre une planche -- « comment choisir le
    capteur ? » -- parce que son gain entre dans le gain de boucle, donc
    dans la stabilite : un capteur plus sensible rapproche de l'instabilite.
    """
    return capteur


def coherence_des_unites_du_gain_de_boucle(
    unite_directe: str,
    unite_reaction: str,
) -> bool:
    """Vrai si le produit des deux chaines est sans dimension.

    « Attention aux grandeurs physiques / unites », avertit la planche. Le
    gain de boucle se compare a `-1` : s'il portait une unite, la
    comparaison n'aurait aucun sens. La verification est elementaire --
    les deux unites doivent etre inverses l'une de l'autre -- et c'est
    pourquoi on l'oublie.
    """
    numerateur, _, denominateur = unite_directe.partition("/")
    return unite_reaction == f"{denominateur}/{numerateur}"


def etapes_du_dimensionnement() -> tuple[str, ...]:
    """Les quatre etapes que la planche enumere sous « dimensionner une boucle ».

    L'ordre compte : le capteur vient **en premier**, parce que son gain
    entre dans le gain de boucle et donc dans la stabilite. Le choisir
    apres le correcteur obligerait a tout reprendre.
    """
    return (
        "choisir le capteur, dont le gain entre dans le gain de boucle",
        "etablir le modele du procede et le valider",
        "verifier la stabilite en boucle fermee",
        "regler le correcteur sur les performances visees",
    )
