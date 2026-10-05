"""L'ailette circulaire, par sa solution exacte.

Syllabus de thermique, tome 1, premiere partie, section 2.3.3. Un disque
de rayons `r_1` et `r_2` et d'epaisseur `b`, serti sur un tube. En
coordonnees cylindriques, `theta'' + theta' / r - m^2 theta = 0` avec
`m = racine(2 K / (lambda b))`, dont les solutions sont les fonctions de
Bessel modifiees d'ordre zero. Bout adiabatique, l'efficacite vaut

`Omega = 2 r_1 / (m (r_2^2 - r_1^2))
         (I_1(m r_2) K_1(m r_1) - K_1(m r_2) I_1(m r_1))
       / (I_1(m r_2) K_0(m r_1) + K_1(m r_2) I_0(m r_1))`,

fonction de `m (r_2 - r_1)` et de `r_2 / r_1` ; elle tend vers celle de
l'ailette droite quand ce rapport tend vers un. Une ailette carree de
cote `a` se traite comme le disque de meme aire.
"""

from __future__ import annotations

import math

from scipy.special import ive, kve


def rayon_equivalent_d_un_carre(cote: float) -> float:
    """`r = racine(a^2 / pi)` : le disque de meme aire."""
    return math.sqrt(cote**2 / math.pi)


def efficacite_circulaire(parametre: float, pied: float, exterieur: float) -> float:
    """`Omega`, par les fonctions de Bessel modifiees.

    Elles sont prises sous leur forme reduite, `I e^(-x)` et `K e^(x)` :
    le rapport ne change pas, et les grands arguments ne debordent plus.
    """
    if exterieur <= pied:
        msg = f"rayon exterieur non superieur au pied : {exterieur} <= {pied}"
        raise ValueError(msg)
    au_pied, au_bout = parametre * pied, parametre * exterieur
    retour = math.exp(2 * (au_pied - au_bout))
    haut = ive(1, au_bout) * kve(1, au_pied) - retour * kve(1, au_bout) * ive(
        1, au_pied
    )
    bas = ive(1, au_bout) * kve(0, au_pied) + retour * kve(1, au_bout) * ive(0, au_pied)
    prefacteur = 2 * pied / (parametre * (exterieur**2 - pied**2))
    return float(prefacteur * haut / bas)
