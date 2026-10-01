"""L'effet Seebeck -- section 6.8 du syllabus d'electricite generale.

« Dans une chaine metallique dont les extremites seules sont a la meme
temperature, il apparait en general une difference de potentiel entre ces
dernieres », et elle resulte de deux causes que le syllabus distingue :
les forces electromotrices de contact successives, « qui n'obeissent plus
a la loi des chaines », et celles dues aux gradients de temperature au
sein des conducteurs.

La precision sur la loi des chaines porte tout le sens : une chaine
**isotherme** ne donne aucun effet, les forces de contact s'y annulant.
C'est l'ecart de temperature qui rompt cette annulation.

Le syllabus chiffre, et c'est ce qui rend le chapitre verifiable :
« environ 2,2 µV » par degre pour le cuivre seul, et « quelque 40 a
50 µV par degre Celsius » pour une chaine cuivre-constantan-cuivre. Le
facteur vingt entre les deux est la raison d'etre du thermocouple -- un
metal seul donne un signal a la limite du mesurable.

Le dernier chiffre explique ce que le cours d'electronique traite de son
cote : a cent degres d'ecart un thermocouple ne delivre que quatre a cinq
millivolts, et une telle tension se perd dans le bruit d'une chaine
d'acquisition non amplifiee.

Le modele est **lineaire**, comme le « par degre Celsius » du syllabus
l'indique. La courbe reelle d'un thermocouple ne l'est pas tout a fait,
mais le syllabus ne donne aucune correction et ce module n'en invente
pas.
"""

from __future__ import annotations

from typing import Final

COEFFICIENT_DU_CUIVRE: Final = 2.2e-6
"""« Environ 2,2 µV » par degre Celsius, pour le cuivre seul."""

PLAGE_DU_COUPLE_CUIVRE_CONSTANTAN: Final = (40e-6, 50e-6)
"""« Quelque 40 a 50 µV par degre Celsius », pour la chaine du syllabus."""

CAUSES_DE_L_EFFET_SEEBECK: Final = (
    (
        "les forces electromotrices de contact successives, qui n'obeissent "
        "plus a la loi des chaines"
    ),
    (
        "les forces electromotrices dues aux gradients de temperature au "
        "sein des conducteurs"
    ),
)
"""Les deux causes, dans l'ordre ou le syllabus les numerote."""


def tension_de_seebeck(coefficient: float, ecart: float) -> float:
    """`V = S Delta T`, en volts.

    Le signe du coefficient depend du sens de la jonction et non du
    materiau seul ; le syllabus ne donnant que des grandeurs, ce module
    refuse un coefficient negatif plutot que d'y ajouter une convention
    qu'il ne pose pas.
    """
    if coefficient < 0:
        msg = f"coefficient de Seebeck negatif : {coefficient} V/K"
        raise ValueError(msg)
    return coefficient * ecart


def ecart_depuis_la_tension(coefficient: float, tension: float) -> float:
    """`Delta T = V / S`, en kelvins : le sens ou un thermocouple sert.

    On mesure une tension, on en deduit un ecart de temperature.
    L'exactitude de l'aller-retour est donc la condition de la mesure.
    """
    if coefficient <= 0:
        msg = f"coefficient de Seebeck nul ou negatif : {coefficient} V/K"
        raise ValueError(msg)
    return tension / coefficient
