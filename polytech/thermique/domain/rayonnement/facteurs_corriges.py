"""Les facteurs d'angle corriges : la geometrie et les emissivites ensemble.

Syllabus de thermique, tome 1, deuxieme partie, sections 3.4 et 3.6.2.
Le facteur d'angle corrige rapporte l'echange total a l'aire de la
source : `Phi = F S_1 sigma (T_1^4 - T_2^4)`, avec `S_1S_2 = F S_1`.

Pour une source, un recepteur et une paroi adiabatique, le schema
equivalent met en serie les deux resistances de surface et l'echange
entre corps noirs, que la paroi relaie :

    F_12,R = F_12 + F_1R F_R2 / (1 - F_RR).
"""

from __future__ import annotations


def _exiger_une_emissivite(emissivite: float) -> None:
    if not 0 < emissivite <= 1:
        msg = f"emissivite hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)


def facteur_corrige_a_trois_zones(
    emissivites: tuple[float, float], *, rapport_des_aires: float, geometrique: float
) -> float:
    """`1 / (rho_1/eps_1 + (rho_2/eps_2)(S_1/S_2) + 1/F_12,R)`."""
    premiere, seconde = emissivites
    _exiger_une_emissivite(premiere)
    _exiger_une_emissivite(seconde)
    source = (1 - premiere) / premiere
    recepteur = (1 - seconde) / seconde * rapport_des_aires
    return 1 / (source + recepteur + 1 / geometrique)
