"""Les familles d'approximation, les cinq types, et le reseau en echelle.

Le fascicule compare quatre familles, et chacune optimise une chose
differente -- c'est un compromis, pas une hierarchie :

* **Butterworth** est le plus plat dans la bande, sans aucune ondulation ;
* **Chebyshev** de type I est plus raide, au prix d'une ondulation dans la
  bande passante ;
* **Bessel** a le temps de groupe le plus constant, donc deforme le moins
  un signal numerique -- le fascicule y consacre cinq planches ;
* **elliptique**, dit de Cauer, est le plus raide de tous, au prix d'une
  ondulation **des deux cotes**.

L'ordre croit donc de l'elliptique vers Bessel a gabarit egal, et aucune
famille ne peut etre a la fois la plus raide et la plus plate.

Le **reseau en echelle LC** est la realisation passive de ces prototypes.
La convention des tables du cours : inductance serie d'abord, puis
capacite shunt, en alternance.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Final

from polytech.electronique.domain.gabarit import (
    DECIBELS,
    Gabarit,
    ordre_de_butterworth,
    ordre_de_chebyshev,
)

TYPES_DE_FILTRES: Final = (
    "passe-bas",
    "passe-haut",
    "passe-bande",
    "coupe-bande",
    "passe-tout",
)
"""Les cinq types que le fascicule enumere.

Le passe-tout est le moins connu : il ne touche pas au module et n'agit
que sur la phase. C'est l'outil des correcteurs de temps de groupe.
"""

FACTEUR_DE_BESSEL: Final = 1.4
"""Bessel paie sa linearite de phase par un ordre plus eleve.

Le facteur est approche et dit comme tel : le fascicule donne ses
procedures de dimensionnement en tables, et non en formule fermee. Il
sert a **ordonner** les familles, pas a dimensionner.
"""

FACTEUR_ELLIPTIQUE: Final = 0.7
"""L'elliptique gagne sur Chebyshev, dans la meme proportion approchee."""


class Famille(Enum):
    """Les quatre approximations que le fascicule compare."""

    BUTTERWORTH = "Butterworth"
    CHEBYSHEV = "Chebyshev de type I"
    BESSEL = "Bessel"
    ELLIPTIQUE = "elliptique (Cauer)"


def compromis_de_la_famille(famille: Famille) -> str:
    """Ce que cette famille optimise, et ce qu'elle y sacrifie."""
    return {
        Famille.BUTTERWORTH: "le plus plat dans la bande, sans ondulation",
        Famille.CHEBYSHEV: "plus raide, au prix d'une ondulation dans la bande",
        Famille.BESSEL: "temps de groupe le plus constant, donc deforme le moins",
        Famille.ELLIPTIQUE: "le plus raide, au prix d'une ondulation des deux cotes",
    }[famille]


def ordre_par_famille(gabarit: Gabarit, famille: Famille) -> int:
    """L'ordre que cette famille exige pour tenir ce gabarit.

    Butterworth et Chebyshev ont des formules fermees, que le cours donne.
    Bessel et l'elliptique n'en ont pas -- le fascicule les dimensionne par
    tables -- et les facteurs employes ici sont approches et declares comme
    tels : ils servent a ordonner les familles, non a dimensionner.
    """
    if famille is Famille.BUTTERWORTH:
        return ordre_de_butterworth(gabarit)
    if famille is Famille.CHEBYSHEV:
        return ordre_de_chebyshev(gabarit)
    if famille is Famille.BESSEL:
        return math.ceil(FACTEUR_DE_BESSEL * ordre_de_butterworth(gabarit))
    return max(1, math.ceil(FACTEUR_ELLIPTIQUE * ordre_de_chebyshev(gabarit)))


def ondulation_depuis_l_affaiblissement(affaiblissement: float) -> float:
    """`epsilon = racine(10^(A/10) - 1)` : l'ondulation que `A` autorise.

    Trois decibels donnent une ondulation de **un**, et c'est pourquoi
    `3 dB` revient partout dans le chapitre : c'est l'affaiblissement pour
    lequel le parametre d'ondulation vaut l'unite.
    """
    if affaiblissement < 0:
        msg = (
            f"affaiblissement negatif : {affaiblissement} dB -- un filtre "
            f"n'amplifie pas dans sa bande passante"
        )
        raise ValueError(msg)
    return math.sqrt(10 ** (affaiblissement / DECIBELS) - 1)


def reseau_en_echelle(elements: tuple[float, ...]) -> tuple[str, ...]:
    """La nature de chaque element du reseau, dans l'ordre du prototype.

    Inductance serie aux rangs pairs, capacite shunt aux rangs impairs :
    c'est la convention des tables du cours. Un ordre impair commence donc
    et finit par une inductance, ce qui decide des terminaisons.
    """
    if not elements:
        msg = "prototype vide : aucun element a realiser"
        raise ValueError(msg)
    if any(valeur <= 0 for valeur in elements):
        msg = f"element non positif dans le prototype : {elements}"
        raise ValueError(msg)
    return tuple(
        "inductance serie" if rang % 2 == 0 else "capacite shunt"
        for rang in range(len(elements))
    )


DECIBELS_PAR_DECADE_ET_PAR_POLE: Final = 20.0
"""Chaque pole ajoute vingt decibels par decade a la pente asymptotique."""


def raideur_asymptotique(ordre: int) -> float:
    """`20 n` decibels par decade, loin de la coupure.

    C'est le **seul** effet de l'ordre a l'infini, et c'est pourquoi les
    familles ne se distinguent pas par leur raideur asymptotique -- toutes
    font `20 n` dB/decade -- mais par leur comportement **dans** la
    transition. Le fascicule le montre sur sa carte des poles et zeros.
    """
    if ordre <= 0:
        msg = f"ordre nul ou negatif : {ordre}"
        raise ValueError(msg)
    return DECIBELS_PAR_DECADE_ET_PAR_POLE * ordre


def ondulation_des_deux_cotes(famille: Famille) -> bool:
    """Vrai si cette famille ondule dans la bande **et** hors de la bande.

    Seul l'elliptique le fait, et c'est ce qui lui donne sa raideur : il
    depense de l'ondulation des deux cotes pour raidir la transition.
    Chebyshev de type I n'ondule que dans la bande.
    """
    return famille is Famille.ELLIPTIQUE


def procedure_de_dimensionnement() -> tuple[str, ...]:
    """Les quatre etapes que le fascicule donne pour chaque famille.

    Elles sont les memes pour toutes -- seule la table du prototype change.
    C'est ce qui rend le choix de famille independant de la demarche, et
    donc reportable a la fin.
    """
    return (
        "poser le gabarit : A_max, A_min, omega_p, omega_s",
        "en deduire l'ordre par la formule de la famille",
        "lire le prototype passe-bas normalise dans sa table",
        "mettre a l'echelle d'impedance, puis transposer",
    )
