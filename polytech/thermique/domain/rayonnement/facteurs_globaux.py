"""Les facteurs globaux d'absorption et d'emission d'un corps selectif.

Syllabus de thermique, tome 1, deuxieme partie, sections 1.2.3, 1.3 et
1.5. Le facteur d'absorption spectral `alpha_lambda` caracterise le
materiau. Le facteur **global**, lui, se pondere par le spectre recu :

    alpha = integrale(alpha_lambda f_lambda) / integrale(f_lambda),

et depend donc de la source. Quand celle-ci est un corps noir ou gris a
la temperature `T_i`, le poids est la loi de Planck a `T_i`. Le facteur
d'emission global est la meme moyenne, prise a la temperature du corps
lui-meme : c'est une propriete du materiau.

Pour un facteur spectral en marches d'escalier, la moyenne se reduit a
une somme ponderee par la fraction `k(lambda T)` que le corps noir emet
dans chaque bande.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from scipy.integrate import quad

from polytech.thermique.domain.rayonnement.corps_noir import fraction_emise_sous

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEBUT_DU_SPECTRE = 1e-8
FIN_DU_SPECTRE = 1e-3
"""De 0,01 a 1000 micrometres : le rayonnement thermique y est tout entier."""


@dataclass(frozen=True, slots=True)
class Bande:
    """Une marche du facteur spectral : sa limite haute, en metres, et sa valeur."""

    limite: float
    facteur: float


def _integrale(fonction: Callable[[float], float], coupures: Sequence[float]) -> float:
    bornes = (DEBUT_DU_SPECTRE, *coupures, FIN_DU_SPECTRE)
    morceaux = (
        quad(fonction, bas, haut, limit=200)[0] for bas, haut in pairwise(bornes)
    )
    return float(sum(morceaux))


def facteur_global(
    spectral: Callable[[float], float],
    spectre: Callable[[float], float],
    coupures: Sequence[float] = (),
) -> float:
    """`alpha`, moyenne de `spectral` ponderee par le `spectre` incident.

    Les `coupures` sont les longueurs d'onde ou `spectral` saute.
    """

    def pondere(longueur: float) -> float:
        return spectral(longueur) * spectre(longueur)

    return _integrale(pondere, coupures) / _integrale(spectre, coupures)


def _fraction_sous(limite: float, temperature: float) -> float:
    if math.isinf(limite):
        return 1.0
    return fraction_emise_sous(limite, temperature)


def facteur_global_par_bandes(bandes: Sequence[Bande], temperature: float) -> float:
    """Le facteur global face a un corps noir, ou gris, a cette temperature."""
    if not math.isinf(bandes[-1].limite):
        msg = f"derniere bande bornee : {bandes[-1].limite} m"
        raise ValueError(msg)
    somme, deja = 0.0, 0.0
    precedente = 0.0
    for bande in bandes:
        if bande.limite <= precedente:
            msg = f"bandes non croissantes : {bande.limite} m"
            raise ValueError(msg)
        sous = _fraction_sous(bande.limite, temperature)
        somme += bande.facteur * (sous - deja)
        deja, precedente = sous, bande.limite
    return somme
