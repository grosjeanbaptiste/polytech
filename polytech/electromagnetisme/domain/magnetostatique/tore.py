"""Le champ d'un tore bobine : le circuit magnetique parfait.

Les planches du chapitre III.5 (21 et 22), et le paragraphe 10.13 du
syllabus. Un anneau porte un courant superficiel uniformement reparti :
par symetrie, le champ est tangent aux cercles centres sur l'axe et de
meme grandeur le long de chacun. Le theoreme d'Ampere donne alors
`2 pi r H = I` sur un cercle interieur a l'anneau, `I` etant le courant
total qui traverse le disque, et `2 pi r H = 0` sur un cercle exterieur :
le champ est nul au-dehors, et le flux reste concentre dans l'anneau.

A l'interieur, le champ depend de `r` ; il varie peu si le rayon central
est grand devant la section.
"""

from __future__ import annotations

import math


def champ_du_tore_parfait(
    courant_total: float, rayon: float, *, interieur: float, exterieur: float
) -> float:
    """`H = I / (2 pi r)` dans l'anneau, zero ailleurs, en amperes par metre."""
    if not 0 < interieur < exterieur:
        msg = f"rayons invalides : interieur {interieur} m, exterieur {exterieur} m"
        raise ValueError(msg)
    if not interieur <= rayon <= exterieur:
        return 0.0
    return courant_total / (2 * math.pi * rayon)
