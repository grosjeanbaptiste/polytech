"""L'ecran de tubes adosse a une paroi adiabatique.

Syllabus de thermique, tome 1, deuxieme partie, sections 3.8 et 3.9. Ce
qui passe entre les tubes n'est plus perdu : la paroi `R` le renvoie, et
la rangee en arrete de nouveau la fraction `F_PT`. Pour des tubes noirs,

    F_PT,R = F_PT + (1 - F_PT) F_PT = F_PT (2 - F_PT).

Des tubes gris ajoutent leur resistance de surface, rapportee au metre
carre de plan : le facteur corrige (3.19). C'est aussi l'emissivite du
plan gris fictif qui remplace, vu du reste du four, l'ensemble des
tubes et de leur paroi.

Avec deux rangees en quinconce, chacune arrete la fraction `F_PT` de ce
qui lui arrive, a l'aller comme au retour.
"""

from __future__ import annotations

import math

from polytech.thermique.domain.rayonnement.ecran_de_tubes import (
    facteur_du_plan_vers_la_rangee,
)


def facteur_vers_l_ecran_adosse(diametre: float, pas: float) -> float:
    """`F_PT,R`, pour des tubes noirs."""
    direct = facteur_du_plan_vers_la_rangee(diametre, pas)
    return direct * (2 - direct)


def facteur_corrige_de_l_ecran(
    diametre: float, pas: float, *, emissivite: float
) -> float:
    """`1 / ((rho_T / eps_T)(l / (pi d)) + 1 / F_PT,R)`, le plan etant noir."""
    if not 0 < emissivite <= 1:
        msg = f"emissivite hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)
    surface = (1 - emissivite) / emissivite * pas / (math.pi * diametre)
    return 1 / (surface + 1 / facteur_vers_l_ecran_adosse(diametre, pas))


def facteurs_vers_deux_rangees_adossees(
    diametre: float, pas: float
) -> tuple[float, float]:
    """L'echange total vers la premiere rangee, puis vers la seconde."""
    direct = facteur_du_plan_vers_la_rangee(diametre, pas)
    passe = 1 - direct
    premiere = direct + passe**3 * direct
    seconde = passe * direct + passe**2 * direct
    return premiere, seconde
