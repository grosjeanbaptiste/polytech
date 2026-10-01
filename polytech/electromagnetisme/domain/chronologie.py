"""La chronologie de l'electromagnetisme, telle que le chapitre I la donne.

Ce fascicule annonce lui-meme ses trois parties -- historique,
applications, structure du cours -- et aucune n'est un calcul. La
tentation etait de l'ecarter comme « presentation », ce qui a deja ete
une erreur pour `TH-01` : ce chapitre-la enseignait la loi de Newton sous
le meme intitule.

L'examen du fascicule donne autre chose. La chronologie **est** son
contenu, et elle dit ce que les chapitres suivants ne disent pas : dans
quel ordre les lois sont venues, donc laquelle generalise laquelle.
Maxwell unifie en 1873 ce que Coulomb, Oersted, Ampere et Faraday avaient
etabli separement ; Hertz confirme en 1887 par l'experience ce que
Maxwell avait predit par le calcul ; et la pile de Volta, en 1796, est ce
qui a rendu les quatre lois du courant observables -- « il devient
possible de mettre des charges en mouvement », dit le fascicule.

Cet ordre est un argument, pas un ornement, et une date fausse le
romprait. Le precedent est `EI-0`, ou le plan du cours s'est revele
testable de la meme facon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Jalon:
    """Une etape de la chronologie : qui, quand, et ce qu'il a etabli."""

    annee: int
    auteur: str
    apport: str


JALONS: Final = (
    Jalon(
        -625,
        "Thales de Milet",
        "l'ambre frottee attire la paille, et la "
        "pierre de Magnesie attire le fer : les deux mots du nom",
    ),
    Jalon(
        1785,
        "Charles-Augustin Coulomb",
        "la loi de la force electrostatique entre deux charges",
    ),
    Jalon(
        1796,
        "Alessandro Volta",
        "la premiere pile : mettre des charges en mouvement, donc produire un courant",
    ),
    Jalon(
        1819,
        "Hans Christian Oersted",
        "un fil parcouru par un courant "
        "devie l'aiguille d'une boussole : electricite et magnetisme sont "
        "lies",
    ),
    Jalon(
        1820,
        "Jean-Baptiste Biot et Felix Savart",
        "la loi donnant la force exercee sur l'aiguille, en grandeur et en direction",
    ),
    Jalon(
        1826,
        "Andre Marie Ampere",
        "l'action mutuelle entre deux fils parcourus par un courant",
    ),
    Jalon(
        1827,
        "Georg Simon Ohm",
        "la difference de potentiel aux bornes d'un "
        "conducteur est proportionnelle au courant",
    ),
    Jalon(
        1830,
        "Michael Faraday",
        "l'induction : un courant circule dans un "
        "conducteur soumis a un champ magnetique variable",
    ),
    Jalon(
        1873,
        "James Clerk Maxwell",
        "une theorie unifiant electricite et "
        "magnetisme, dont les equations font apparaitre des ondes se "
        "propageant a la vitesse de la lumiere",
    ),
    Jalon(
        1887,
        "Henry Hertz",
        "la preuve experimentale de l'existence des "
        "ondes electromagnetiques que Maxwell avait predites",
    ),
)
"""Les dix jalons du fascicule, dans son ordre -- qui est chronologique."""

ANNEE_DE_L_UNIFICATION: Final = 1873
"""L'annee de Maxwell, qui partage la chronologie en deux."""


def jalon(auteur: str) -> Jalon:
    """Le jalon dont l'auteur porte ce nom, par correspondance partielle."""
    for candidat in JALONS:
        if auteur.casefold() in candidat.auteur.casefold():
            return candidat
    msg = f"aucun jalon pour {auteur!r} dans la chronologie du chapitre I"
    raise KeyError(msg)


def premiere_unification() -> Jalon:
    """Le jalon de Maxwell : celui qui reunit ce qui precedait."""
    return jalon("Maxwell")


def apres(tardif: Jalon, precoce: Jalon) -> bool:
    """Vrai si le premier jalon vient apres le second."""
    return tardif.annee > precoce.annee
