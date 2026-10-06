"""Le potentiometre : une resistance a trois pattes, et un curseur.

Presentation des laboratoires, planche 19. Entre les pattes extremes A et
C se lit le calibre ; le curseur B le partage en `x` entre A et B et le
reste entre B et C. Avec deux pattes seulement, c'est une resistance
variable de zero au calibre ; avec les trois, un diviseur de tension dont
le rapport est la position du curseur.
"""

from __future__ import annotations


def resistances_du_potentiometre(calibre: float, curseur: float) -> tuple[float, float]:
    """`(x, calibre - x)` pour un curseur a la fraction `curseur` de sa course."""
    if calibre <= 0:
        msg = f"calibre nul ou negatif : {calibre} ohm"
        raise ValueError(msg)
    if not 0 <= curseur <= 1:
        msg = f"curseur hors de [0, 1] : {curseur}"
        raise ValueError(msg)
    gauche = curseur * calibre
    return gauche, calibre - gauche


def rapport_du_diviseur(calibre: float, curseur: float) -> float:
    """`R_AB / R_AC` : la fraction de la tension que le curseur preleve, a vide."""
    gauche, droite = resistances_du_potentiometre(calibre, curseur)
    return gauche / (gauche + droite)
