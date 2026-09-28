"""La formule de Friis : le bruit d'une cascade est domine par son entree.

Le second etage voit son bruit divise par le gain en puissance du premier,
soit le carre de son gain en tension. C'est pourquoi le choix du premier
amplificateur commande tout le budget -- et pourquoi, a l'inverse, la
contrainte sur le second est si lache.

Un facteur de bruit vaut au moins 1 : un quadripole ne peut pas rendre le
signal moins bruite qu'il ne l'a recu. Cette borne est ce qui fait de la
question 5.2 un probleme et non une simple application.
"""

from __future__ import annotations

FACTEUR_MINIMAL = 1.0
"""Un quadripole n'enleve pas de bruit ; son facteur ne descend pas sous 1."""


def _valider(gain_du_premier: float) -> None:
    """Refuse un gain qui rendrait la formule de Friis vide de sens."""
    if gain_du_premier <= 0:
        msg = f"gain du premier etage non positif : {gain_du_premier}"
        raise ValueError(msg)


def facteur_en_cascade(
    premier: float,
    second: float,
    gain_du_premier: float,
) -> float:
    """`F = F1 + (F2 - 1) / A1^2`, les gains etant des gains en tension."""
    _valider(gain_du_premier)
    return premier + (second - FACTEUR_MINIMAL) / gain_du_premier**2


def facteur_du_premier_etage(
    global_: float,
    second: float,
    gain_du_premier: float,
) -> float:
    """La meme formule, resolue pour le premier etage."""
    _valider(gain_du_premier)
    return global_ - (second - FACTEUR_MINIMAL) / gain_du_premier**2


def facteur_maximal_du_second_etage(
    global_: float,
    gain_du_premier: float,
) -> float:
    """Le plus grand `F2` qui laisse `F1` physiquement possible.

    `F1 >= 1` donne directement `F2 <= 1 + (F - 1) A1^2`.
    """
    _valider(gain_du_premier)
    if global_ < FACTEUR_MINIMAL:
        msg = f"facteur global sous la borne physique : {global_}"
        raise ValueError(msg)
    return FACTEUR_MINIMAL + (global_ - FACTEUR_MINIMAL) * gain_du_premier**2
