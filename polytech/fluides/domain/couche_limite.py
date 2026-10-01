"""La couche limite laminaire, et la solution de Blasius qui la decrit.

Separee de `aerodynamique`, qui porte la portance et la trainee d'un
profil. Les deux responsabilites sont distinctes -- l'une decrit ce qui
se passe **le long d'une paroi**, l'autre ce qu'une aile subit dans son
ensemble -- et le fichier unique depassait la limite de taille du depot.

Le fil du chapitre est que les trois epaisseurs sont **proportionnelles
entre elles**, avec des rapports que ni la vitesse ni la position ne
changent : la solution de Blasius est auto-semblable. On en calcule donc
une, et les deux autres s'en deduisent -- mais seulement tant que
l'ecoulement est laminaire, ce que chaque fonction **verifie** plutot que
de le supposer. C'est la garde qui porte le sens du chapitre : passe
`Re = 5 . 10^5`, ces formules ne disent plus rien du reel.
"""

from __future__ import annotations

import math
from typing import Final

CONSTANTE_DE_BLASIUS: Final = 5.0
"""`delta = 5,0 x / racine(Re_x)` : la constante de la solution de Blasius.

Elle vient de la resolution **numerique** de son equation differentielle,
et non d'une forme fermee -- c'est pourquoi le fascicule la donne telle
quelle.
"""

RAPPORT_DE_DEPLACEMENT: Final = 0.344
"""`delta* / delta` pour le profil de Blasius, independant de x et de U."""

RAPPORT_DE_QUANTITE_DE_MOUVEMENT: Final = 0.133
"""`theta / delta` pour le meme profil, et de la meme facon universel."""

COEFFICIENT_DE_FROTTEMENT_DE_BLASIUS: Final = 0.664
"""`C_f = 0,664 / racine(Re_x)` en laminaire."""

REYNOLDS_DE_TRANSITION: Final = 5e5
"""Au-dela, la couche limite devient turbulente et Blasius cesse de valoir."""


def reynolds_local(vitesse: float, distance: float, viscosite: float) -> float:
    """`Re_x = U x / nu` : il mesure la distance parcourue le long de la paroi.

    Et non une dimension du corps -- c'est ce qui le distingue du Reynolds
    global, et ce qui permet de dire que la couche limite transitionne **a
    une abscisse** et non d'un bloc.
    """
    if viscosite <= 0:
        msg = f"viscosite cinematique nulle ou negative : {viscosite} m^2/s"
        raise ValueError(msg)
    if distance < 0:
        msg = f"distance au bord d'attaque negative : {distance} m"
        raise ValueError(msg)
    return vitesse * distance / viscosite


def est_laminaire(reynolds: float) -> bool:
    """Vrai sous le Reynolds de transition : Blasius y vaut, au-dela non."""
    return reynolds < REYNOLDS_DE_TRANSITION


def _exiger_le_laminaire(reynolds: float) -> None:
    if not est_laminaire(reynolds):
        msg = (
            f"la solution de Blasius ne vaut qu'en laminaire : Re_x = "
            f"{reynolds:.3g} depasse {REYNOLDS_DE_TRANSITION:.0e}"
        )
        raise ValueError(msg)


def epaisseur_de_couche_limite(
    vitesse: float,
    distance: float,
    viscosite: float,
) -> float:
    """`delta_99 = 5,0 x / racine(Re_x)`, en metres.

    C'est l'epaisseur ou la vitesse atteint 99 pour cent de l'ecoulement
    libre. Elle croit comme la **racine** de la distance : la couche limite
    s'epaissit de plus en plus lentement.
    """
    reynolds = reynolds_local(vitesse, distance, viscosite)
    _exiger_le_laminaire(reynolds)
    if reynolds == 0:
        return 0.0
    return CONSTANTE_DE_BLASIUS * distance / math.sqrt(reynolds)


def epaisseur_de_deplacement(
    vitesse: float,
    distance: float,
    viscosite: float,
) -> float:
    """`delta* = 0,344 delta` : de combien les lignes de courant sont repoussees.

    Ce n'est **pas** l'epaisseur de la couche limite. C'est le deficit de
    debit converti en une epaisseur equivalente, et elle vaut un tiers de
    la premiere. Les confondre fausse un calcul de section utile d'un
    facteur trois.
    """
    delta = epaisseur_de_couche_limite(vitesse, distance, viscosite)
    return RAPPORT_DE_DEPLACEMENT * delta


def epaisseur_de_quantite_de_mouvement(
    vitesse: float,
    distance: float,
    viscosite: float,
) -> float:
    """`theta = 0,133 delta` : le deficit de quantite de mouvement.

    La plus petite des trois, et celle qui donne directement la trainee de
    friction. Confondre les trois change un resultat d'un facteur sept.
    """
    delta = epaisseur_de_couche_limite(vitesse, distance, viscosite)
    return RAPPORT_DE_QUANTITE_DE_MOUVEMENT * delta


def coefficient_de_frottement_local(reynolds: float) -> float:
    """`C_f = 0,664 / racine(Re_x)` : il **decroit** le long de la paroi.

    Contre-intuitif, et c'est le point du fascicule : le frottement total
    croit avec la surface mouillee, mais le coefficient local decroit,
    parce que la couche limite epaissit et adoucit le gradient a la paroi.
    """
    _exiger_le_laminaire(reynolds)
    if reynolds <= 0:
        msg = f"Reynolds local nul ou negatif : {reynolds}"
        raise ValueError(msg)
    return COEFFICIENT_DE_FROTTEMENT_DE_BLASIUS / math.sqrt(reynolds)
