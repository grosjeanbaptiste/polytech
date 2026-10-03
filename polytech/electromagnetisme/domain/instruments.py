"""La precision d'un appareil de mesure, et sa propagation.

Un multimetre annonce sa precision en deux parts :
`+-(% de la lecture + % de la gamme)`. La premiere est proportionnelle a
ce qu'on lit ; la seconde est fixe pour un calibre, et c'est pourquoi on
mesure sur le plus petit calibre qui contienne la valeur -- au plus haut
d'une gamme, la part fixe pese le moins.

Une grandeur tiree d'un quotient, comme `R = U / I`, cumule dans le pire
des cas les incertitudes **relatives** de ses deux termes.
"""

from __future__ import annotations


def precision_d_une_lecture(
    lecture: float,
    gamme: float,
    pourcent_de_la_lecture: float,
    pourcent_de_la_gamme: float,
) -> float:
    """L'incertitude absolue, dans l'unite de la lecture ; leve hors de la gamme."""
    if abs(lecture) > gamme:
        msg = f"lecture {lecture} hors de la gamme {gamme} : changer de calibre"
        raise ValueError(msg)
    return (abs(lecture) * pourcent_de_la_lecture + gamme * pourcent_de_la_gamme) / 100


def precision_d_un_quotient(
    dividende: float,
    incertitude_du_dividende: float,
    diviseur: float,
    incertitude_du_diviseur: float,
) -> tuple[float, float]:
    """Le quotient et son incertitude absolue, au pire des cas."""
    if diviseur == 0:
        msg = "diviseur nul : le quotient n'existe pas"
        raise ValueError(msg)
    quotient = dividende / diviseur
    relative = abs(incertitude_du_dividende / dividende) + abs(
        incertitude_du_diviseur / diviseur
    )
    return quotient, abs(quotient) * relative
