"""La loi de Newton de la viscosite, et les trois lois qui s'en ecartent.

Coeur du chapitre I : la planche « Relation pour la viscosite » definit
la viscosite par `tau = mu du/dy`, et la suivante met en regard quatre
comportements qu'elle nomme -- newtonien, Bingham, pseudoplastique,
dilatant.

La distinction n'est pas academique, et chacune se reconnait a un geste
du quotidien :

* **Bingham** -- `tau - tau_0 = mu du/dy` : sous son seuil le fluide ne
  coule pas du tout. Le dentifrice tient dans sa brosse, la peinture ne
  goutte pas du pot ;
* **pseudoplastique** -- `tau = k (du/dy)^n` avec `n < 1` : il se
  fluidifie sous le cisaillement. Une peinture s'etale au pinceau puis se
  fige sur le mur ;
* **dilatant** -- la meme loi avec `n > 1` : il s'epaissit sous le
  cisaillement. C'est le sable mouille de la planche, ou l'on s'enfonce
  lentement mais qui durcit sous un coup.

La loi de puissance redonne **exactement** Newton a `n = 1`, ce qu'un
test verifie : c'est ce qui en fait une generalisation et non une autre
loi.

L'aller-retour contrainte-viscosite est le principe du viscosimetre : on
impose un gradient connu, on mesure la contrainte par le couple sur un
cylindre tournant, on en tire `mu`. Son exactitude est donc la condition
de la mesure.
"""

from __future__ import annotations

from enum import Enum

SEUIL_NEWTONIEN = 1.0
"""L'exposant qui separe le rheofluidifiant du rheoepaississant."""


class Comportement(Enum):
    """Les trois comportements que la loi de puissance distingue."""

    PSEUDOPLASTIQUE = "pseudoplastique"
    NEWTONIEN = "newtonien"
    DILATANT = "dilatant"

    @classmethod
    def depuis_l_exposant(cls, exposant: float) -> Comportement:
        """Nomme le comportement que cet exposant decrit."""
        if exposant < SEUIL_NEWTONIEN:
            return cls.PSEUDOPLASTIQUE
        if exposant > SEUIL_NEWTONIEN:
            return cls.DILATANT
        return cls.NEWTONIEN


def _exiger_le_gradient(gradient: float) -> None:
    """Le gradient apparait en denominateur des deux mesures."""
    if gradient == 0:
        msg = (
            "gradient de vitesse nul : au repos, un fluide non newtonien "
            "n'a pas de viscosite definie -- c'est ce qui le distingue"
        )
        raise ValueError(msg)


def contrainte_de_newton(viscosite: float, gradient: float) -> float:
    """`tau = mu du/dy`, en pascals : la definition de la viscosite."""
    return viscosite * gradient


def viscosite_depuis_la_contrainte(contrainte: float, gradient: float) -> float:
    """`mu = tau / (du/dy)` : ce que mesure un viscosimetre."""
    _exiger_le_gradient(gradient)
    return contrainte / gradient


def contrainte_de_bingham(seuil: float, viscosite: float, gradient: float) -> float:
    """`tau = tau_0 + mu du/dy`, en pascals.

    A gradient nul elle vaut le **seuil** et non zero : le fluide
    s'oppose, il ne cede pas. C'est ce qui le fait tenir en place.
    """
    if seuil < 0:
        msg = f"seuil de Bingham negatif : {seuil} Pa"
        raise ValueError(msg)
    return seuil + viscosite * gradient


def contrainte_en_loi_de_puissance(
    constante: float,
    exposant: float,
    gradient: float,
) -> float:
    """`tau = k (du/dy)^n`, qui redonne Newton a `n = 1`."""
    if exposant <= 0:
        msg = f"exposant negatif ou nul : {exposant}"
        raise ValueError(msg)
    if gradient < 0:
        msg = f"gradient de vitesse negatif : {gradient} s^-1"
        raise ValueError(msg)
    return float(constante * gradient**exposant)


def viscosite_apparente(constante: float, exposant: float, gradient: float) -> float:
    """`tau / (du/dy)` : la viscosite que l'on mesurerait a ce cisaillement.

    Elle decroit avec le gradient sous l'unite, croit au-dessus -- et
    c'est tout ce que « rheofluidifiant » et « rheoepaississant » veulent
    dire.
    """
    _exiger_le_gradient(gradient)
    contrainte = contrainte_en_loi_de_puissance(constante, exposant, gradient)
    return contrainte / gradient
