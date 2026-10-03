"""Le solenoide a noyau, et l'auto-inductance d'une boucle carree.

**Le solenoide.** Ampere sur un contour qui suit le noyau donne
`H = N I / l` : le champ ne depend que de la densite lineique de spires.
Le flux dans une section vaut `mu H A`, et chacune des `N` spires le
capte : le flux total est proportionnel a `N^2`, d'ou `L = mu N^2 A / l`.
Le noyau sature quand `H` atteint le champ de saturation : le courant
maximal s'en deduit.

**La boucle carree.** Meme sans noyau, un circuit a une inductance. Pour
une boucle carree de cote `a` en fil de rayon `rho`, la formule de Rosa
(1908) l'estime a `mu0/(4 pi) 8a (ln(a/rho) + rho/a - 0,524)`. Elle ne
pese qu'au-dela de la frequence ou `omega L` rattrape la resistance du
circuit.
"""

from __future__ import annotations

import math
from typing import Final

MU0_SUR_4PI: Final = 1e-7
"""`mu0 / (4 pi)` en H/m, la valeur que l'enonce donne."""


def inductance_du_solenoide(
    permeabilite: float, spires: int, aire: float, longueur: float
) -> float:
    """`L = mu N^2 A / l`."""
    return permeabilite * spires**2 * aire / longueur


def courant_de_saturation(
    champ_de_saturation: float, longueur: float, spires: int
) -> float:
    """Le courant pour lequel `N I / l` atteint le champ de saturation."""
    return champ_de_saturation * longueur / spires


def permeabilite_lineaire(induction: float, champ: float) -> float:
    """`mu = B / H` sur la partie lineaire ; leve sur un champ nul."""
    if champ == 0:
        msg = "champ nul : la pente B/H n'est pas definie"
        raise ValueError(msg)
    return induction / champ


def inductance_de_rosa(cote: float, rayon_du_fil: float) -> float:
    """La formule de Rosa ; leve si le fil depasse la boucle."""
    if not 0 < rayon_du_fil < cote:
        msg = f"rayon du fil {rayon_du_fil} incompatible avec le cote {cote}"
        raise ValueError(msg)
    logarithme = math.log(cote / rayon_du_fil)
    return MU0_SUR_4PI * 8 * cote * (logarithme + rayon_du_fil / cote - 0.524)


def frequence_de_croisement(resistance: float, inductance: float) -> float:
    """La frequence ou `|j omega L| = R` : `f = R / (2 pi L)`."""
    return resistance / (2 * math.pi * inductance)
