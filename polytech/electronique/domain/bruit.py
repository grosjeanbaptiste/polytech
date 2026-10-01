"""Les **sources** de bruit : thermique, grenaille, scintillement, quantique.

La metrologie du bruit -- facteur, figure, temperature equivalente, plage
dynamique, choix d'une technologie -- vit dans `facteur_de_bruit`. La
separation n'est pas de confort : ce sont deux responsabilites, et le
fichier unique depassait la limite de taille du depot.

Le fascicule enumere cinq familles, dont **deux seulement** sont blanches.
La distinction decide de l'integration : un bruit blanc s'integre comme la
largeur de bande, un bruit rose comme son logarithme. Elargir la bande vers
le haut coute donc cher en blanc, et allonger la mesure coute cher en rose.

La densite spectrale du bruit thermique se donne sous deux conventions. La
version *unilaterale*, celle du cours, ne compte que les frequences
positives et vaut donc le double de la bilaterale : c'est la distinction
que le mot « one-sided » de l'enonce recouvre.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import sympy as sp

FACTEUR_UNILATERAL: Final = 4
"""Deux pour la convention unilaterale, deux pour le theoreme de Nyquist."""

CHARGE_ELEMENTAIRE: Final = 1.602176634e-19
"""`q`, en coulombs. Le bruit de grenaille la fait apparaitre explicitement."""

FACTEUR_DE_GRENAILLE: Final = 2
"""`2 q I` : un pour la convention unilaterale, un pour le theoreme de Schottky."""

CONSTANTE_DE_PLANCK: Final = 6.62607015e-34
"""`h`, en J s. Le bruit quantique la fait apparaitre."""

FAMILLES_DE_BRUIT: Final = (
    "thermique",
    "grenaille",
    "scintillement",
    "popcorn",
    "quantique",
)
"""Les cinq familles que le fascicule enumere."""

FAMILLES_BLANCHES: Final = frozenset({"thermique", "grenaille"})
"""Celles dont la densite spectrale ne depend pas de la frequence."""


def densite_spectrale_unilaterale(
    resistance: sp.Expr,
    temperature: sp.Expr,
    constante_de_boltzmann: sp.Expr,
) -> sp.Expr:
    """Rend `4 k_B R T`, en volts carres par hertz."""
    return FACTEUR_UNILATERAL * constante_de_boltzmann * resistance * temperature


def densite_spectrale_de_grenaille(
    courant: float,
    charge: float = CHARGE_ELEMENTAIRE,
) -> float:
    """`2 q I`, en amperes carres par hertz.

    Le bruit de grenaille n'apparait que lorsqu'un courant franchit une
    **barriere de potentiel** -- la planche y insiste. Un courant dans une
    resistance ohmique n'en produit pas : il ne franchit rien.
    """
    if courant < 0:
        msg = f"courant negatif : {courant} A -- le sens ne change pas le bruit"
        raise ValueError(msg)
    return FACTEUR_DE_GRENAILLE * charge * courant


def densite_spectrale_en_un_sur_f(
    coefficient: float,
    frequence: float,
    exposant: float = 1.0,
) -> float:
    """`K / f^alpha` : le bruit de scintillement, dit aussi rose ou d'exces.

    Il diverge en zero, et ce n'est pas un defaut du modele : c'est la
    raison pour laquelle on ne l'integre jamais depuis zero mais depuis une
    frequence basse que la duree de mesure fixe.
    """
    if frequence <= 0:
        msg = (
            f"frequence nulle ou negative : {frequence} Hz -- la densite en "
            f"1/f y diverge, et c'est pourquoi on integre depuis f_basse"
        )
        raise ValueError(msg)
    return coefficient / float(frequence**exposant)


def puissance_integree_blanc_et_rose(
    densite_blanche: float,
    coefficient_rose: float,
    basse: float,
    haute: float,
) -> float:
    """`w (f2 - f1) + K ln(f2 / f1)` : la DSP integree sur la bande."""
    if basse <= 0 or haute <= basse:
        msg = f"bande invalide : {basse} Hz a {haute} Hz"
        raise ValueError(msg)
    blanc = densite_blanche * (haute - basse)
    rose = coefficient_rose * math.log(haute / basse)
    return blanc + rose


def tension_efficace_ramenee_a_l_entree(
    densite_en_tension: float,
    densite_en_courant: float,
    resistance_de_source: float,
) -> float:
    """`racine(v_n^2 + (i_n R_s)^2)` : les deux sources du modele reunies.

    Elles s'ajoutent en **puissance** et non en amplitude, parce qu'elles
    sont decorrelees. Et le poids du bruit en courant croit avec la
    resistance de source -- c'est ce qui decide, sur une datasheet, lequel
    des deux domine.
    """
    if resistance_de_source < 0:
        msg = f"resistance de source negative : {resistance_de_source} ohm"
        raise ValueError(msg)
    return math.hypot(densite_en_tension, densite_en_courant * resistance_de_source)


def est_blanc(famille: str) -> bool:
    """Vrai si cette famille a une densite spectrale plate."""
    if famille not in FAMILLES_DE_BRUIT:
        msg = f"famille de bruit inconnue : {famille!r} -- {FAMILLES_DE_BRUIT}"
        raise ValueError(msg)
    return famille in FAMILLES_BLANCHES


def bruit_de_grenaille_quantique(frequence: float) -> float:
    """`2 h f`, en W/Hz : le bruit quantique, proportionnel a la frequence.

    Negligeable en electronique, dominant en optique. Le fascicule le
    mentionne sans l'employer -- a un gigahertz il est quinze ordres de
    grandeur sous le bruit thermique a temperature ambiante.
    """
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    return FACTEUR_DE_GRENAILLE * CONSTANTE_DE_PLANCK * frequence
