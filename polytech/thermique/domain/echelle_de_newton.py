"""La « scala graduum caloris » de 1701, et ce que Newton en a fait.

Separee de `modes`, qui porte les trois modes de transfert et leur
superposition. Ce module-ci porte une **echelle de temperature**, c'est
une autre responsabilite -- et le fichier unique depassait la limite de
taille du depot.

Ce que la planche souligne n'est pas l'echelle, qui n'a que deux points
fixes et n'a pas survecu, mais ce que Newton en a fait : il l'a
**extrapolee** au-dela du domaine de son thermometre a l'huile de lin,
qui plafonnait vers deux cents degres Celsius, en se servant d'une loi de
refroidissement. C'est ainsi qu'il a mesure des temperatures de fusion de
metaux que l'instrument ne pouvait pas atteindre -- et c'est la raison
pour laquelle cette loi ouvre le cours plutot que de l'illustrer.
"""

from __future__ import annotations

from typing import Final

CORPS_HUMAIN_EN_NEWTON: Final = 12.0
"""Le second point fixe de l'echelle de Newton : le corps humain.

Le premier est la glace fondante, a zero. Deux points suffisent a graduer
une echelle lineaire, et c'est tout ce que Newton avait -- son thermometre
a l'huile de lin plafonnait vers 200 degres Celsius.
"""

CORPS_HUMAIN_EN_CELSIUS: Final = 37.0
"""Le meme point, dans l'echelle d'aujourd'hui."""


def degres_newton_depuis_celsius(celsius: float) -> float:
    """La « scala graduum caloris » de 1701, depuis l'echelle actuelle.

    Le coup de genie que la planche souligne n'est pas l'echelle elle-meme
    mais ce que Newton en a fait : il l'a **extrapolee** au-dela du domaine
    de son thermometre, en se servant d'une loi de refroidissement. C'est
    ainsi qu'il a mesure des temperatures de fusion de metaux inaccessibles
    a l'instrument.
    """
    if celsius < 0:
        msg = (
            f"{celsius} degres Celsius est sous le zero de l'echelle de Newton, "
            f"qui part de la glace fondante et n'a pas ete graduee en dessous"
        )
        raise ValueError(msg)
    return celsius * CORPS_HUMAIN_EN_NEWTON / CORPS_HUMAIN_EN_CELSIUS


def celsius_depuis_degres_newton(newton: float) -> float:
    """L'inverse exact de la conversion precedente."""
    if newton < 0:
        msg = f"{newton} degres Newton est sous le zero de son echelle"
        raise ValueError(msg)
    return newton * CORPS_HUMAIN_EN_CELSIUS / CORPS_HUMAIN_EN_NEWTON
