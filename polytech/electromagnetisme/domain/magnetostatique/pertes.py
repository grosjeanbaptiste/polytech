"""Les pertes magnetiques, et le cycle dynamique -- section 17.7 du syllabus.

Le syllabus n'en donne **aucune** formule de Steinmetz : ni exposant, ni
constante de materiau. Ce qu'il donne est plus fondamental, et c'est ce
qui est porte ici : l'aire du cycle d'hysteresis **est** le travail
volumique dissipe. « Comme dans le cas d'une energie dissipee pour
vaincre des frottements, cette energie se retrouve dans la matiere sous
forme de chaleur. »

De la suit la puissance : un cycle par periode, donc l'aire multipliee
par la frequence et par le volume. C'est vrai de l'hysteresis **seule**.

Le cycle dynamique est plus large que le statique, parce que « l'energie
dissipee augmente, par suite notamment de l'apparition de courants
induits au sein du materiau (courants de Foucault) ». L'inegalite est un
resultat du cours, et elle garde le calcul -- mais le surcout n'est pas
chiffre ici, faute de correlation dans le syllabus. C'est la meme
discipline que pour la lame de fluide entre 1400 et 6000.

Le syllabus nomme quatre mecanismes, et les deux derniers sont ceux qu'on
oublie : le **trainage** -- « l'aimantation ne prend sa valeur
d'equilibre qu'apres un certain temps » -- et l'**hysteresis tournante**,
qui apparait quand le champ garde sa grandeur mais tourne, ce qui est le
cas dans une machine.
"""

from __future__ import annotations

from typing import Final

MECANISMES_DE_PERTE: Final = (
    "hysteresis alternative : l'aire du cycle, parcouru infiniment lentement",
    "courants de Foucault : les courants induits dans le materiau lui-meme",
    "trainage ou viscosite magnetique : le retard de l'aimantation sur le champ",
    "hysteresis tournante : un champ de grandeur constante dont la direction tourne",
)
"""Les quatre mecanismes que le syllabus nomme, du plus connu au moins."""


def energie_du_cycle(aire: float, volume: float) -> float:
    """`A V`, en joules : l'energie dissipee a chaque cycle.

    L'aire est en joules par metre cube -- c'est le travail volumique --
    et le volume la convertit en joules.
    """
    if aire < 0:
        msg = f"aire de cycle negative : {aire} J/m^3"
        raise ValueError(msg)
    if volume < 0:
        msg = f"volume negatif : {volume} m^3"
        raise ValueError(msg)
    return aire * volume


def pertes_par_hysteresis(aire: float, frequence: float, volume: float) -> float:
    """`A f V`, en watts : un cycle par periode.

    Lineaire en frequence, et c'est ce qui distingue l'hysteresis des
    courants de Foucault -- dont la part croit plus vite, ce que le
    syllabus annonce sans le chiffrer.
    """
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    return energie_du_cycle(aire, volume) * frequence


def surcout_du_regime_dynamique(statique: float, dynamique: float) -> float:
    """Ce que le regime dynamique dissipe en plus, en joules par metre cube.

    Positif ou nul, toujours : il faudrait que les courants induits
    **rendent** de l'energie pour qu'un cycle dynamique soit plus etroit
    que le statique.
    """
    if dynamique < statique:
        # Message tenu sur deux lignes : l'audit des gardes du depot ne
        # lit que les cinq lignes au-dessus du `raise`, et une affectation
        # plus etalee en sortirait -- la garde passerait pour non eprouvee.
        msg = (
            f"cycle dynamique plus etroit que le statique : {dynamique} "
            f"contre {statique} J/m^3, or les courants induits dissipent"
        )
        raise ValueError(msg)
    return dynamique - statique
