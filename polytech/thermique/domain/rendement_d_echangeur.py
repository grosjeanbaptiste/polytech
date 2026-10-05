"""Le rendement d'un echangeur, pour en calculer les sorties.

Syllabus de thermique, tome 2, deuxieme partie, section 2.4.2. Le
rendement d'un fluide est son ecart de temperature reel, rapporte a
l'ecart maximal que la thermodynamique permet -- celui des deux entrees,
atteint dans un contrecourant de surface infinie.

Eliminer les temperatures laisse deux nombres :

    N_1 = K_gl S / (q_m,1 c_1),        R_1 = q_m,1 c_1 / (q_m,2 c_2),

et `eta_1 = F(N_1, R_1)`, par (2.15) a contrecourant et (2.16) a
cocourant. Un fluide isotherme -- condenseur, bouilleur -- se comporte
comme un debit infini : `R_1 = 0`, et `eta_1 = 1 - exp(-N_1)` dans les
deux sens. Le rendement de l'autre fluide est `eta_2 = R_1 eta_1`.

La forme sert a calculer un echangeur **existant** : ses sorties sont
affines en ses entrees, `t_1S = t_1E - eta_1 (t_1E - t_2E)`, la ou la
moyenne logarithmique demanderait une resolution numerique.
"""

from __future__ import annotations

import math

from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens


def rendement(sens: Sens, unites: float, rapport: float) -> float:
    """`eta_1` par (2.15) ou (2.16), pour tout `R_1` positif ou nul."""
    if unites <= 0:
        msg = f"nombre d'unites nul ou negatif : {unites}"
        raise ValueError(msg)
    if rapport < 0:
        msg = f"rapport des debits de capacite negatif : {rapport}"
        raise ValueError(msg)
    if sens is Sens.COCOURANT:
        return -math.expm1(-unites * (1 + rapport)) / (1 + rapport)
    if math.isclose(rapport, 1.0):
        return unites / (1 + unites)
    exponentielle = math.exp(-unites * (1 - rapport))
    return (1 - exponentielle) / (1 - rapport * exponentielle)


def efficacite(sens: Sens, conductance: float, capacites: tuple[float, float]) -> float:
    """Remarque 3 : le plus eleve des deux rendements, `N` et `Z` sur `(q c)_min`."""
    minimale, maximale = min(capacites), max(capacites)
    return rendement(sens, conductance / minimale, minimale / maximale)


def _exiger_un_rendement(valeur: float) -> None:
    if not 0 <= valeur <= 1:
        msg = f"rendement hors de [0, 1] : {valeur}"
        raise ValueError(msg)


def sortie_du_fluide_chaud(chaude: float, froide: float, rendement_1: float) -> float:
    """`t_1S = t_1E - eta_1 (t_1E - t_2E)` : la caracteristique de fonctionnement."""
    _exiger_un_rendement(rendement_1)
    return chaude - rendement_1 * (chaude - froide)


def sortie_du_fluide_froid(chaude: float, froide: float, rendement_2: float) -> float:
    """`t_2S = t_2E + eta_2 (t_1E - t_2E)`, avec `eta_2 = R_1 eta_1`."""
    _exiger_un_rendement(rendement_2)
    return froide + rendement_2 * (chaude - froide)
