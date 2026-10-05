"""Le rayonnement d'un volume de gaz vers les surfaces qui l'entourent.

Syllabus de thermique, tome 1, deuxieme partie, sections 4.2 a 4.7. Un
gaz n'emet que dans ses bandes d'absorption. Son facteur d'emission
global, pour un volume hemispherique de rayon `l`, est la moyenne de
`1 - exp(-b p l)` ponderee par le rayonnement du corps noir a la
temperature du gaz ; son facteur d'absorption est la meme moyenne, prise
a la temperature de la surface qui emet. Les fumees melent `CO2` et
`H2O`, dont les bandes se recouvrent en partie.

Un volume quelconque se ramene a l'hemisphere par son rayon equivalent :
`1 - exp(-a L_eq)` est la moyenne de `1 - exp(-a l)` sur tous les
rayons. Pour un gaz mince il vaut `L_o = 4 V / S` ; en moyenne sur les
opacites, `L_m = 0,88 L_o`.

Le gaz n'etant pas gris, son echange avec une enveloppe noire s'ecrit
`sigma S (eps_g T_g^4 - alpha_g T_S^4)`. Une enveloppe grise, proche du
noir, y ajoute le facteur `(eps_S + 1) / 2`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.facteurs_globaux import facteur_global
from polytech.thermique.domain.rayonnement.lambert import luminance_de_planck
from polytech.thermique.domain.rayonnement_des_gaz import (
    longueur_moyenne_de_rayonnement,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

FACTEURS_DE_FORME = (0.85, 0.95)
"""Page 166 : « pour toutes les formes etudiees, 0,85 < x < 0,95 »."""


def facteur_d_emission_hemispherique(
    spectral: Callable[[float], float],
    opacite: float,
    temperature: float,
    coupures: Sequence[float] = (),
) -> float:
    """`eps_g`, pour le coefficient `b(lambda)` et l'opacite `p l`."""

    def absorbe(longueur: float) -> float:
        return -math.expm1(-spectral(longueur) * opacite)

    def noir(longueur: float) -> float:
        return luminance_de_planck(longueur, temperature)

    return facteur_global(absorbe, noir, coupures)


def facteur_d_emission_du_melange(
    dioxyde: float, vapeur: float, *, recouvrement: float = 0.0
) -> float:
    """`eps_CO2 + eps_H2O - delta`."""
    if recouvrement < 0:
        msg = f"recouvrement negatif : {recouvrement}"
        raise ValueError(msg)
    melange = dioxyde + vapeur - recouvrement
    if not 0 <= melange <= 1:
        msg = f"facteur d'emission hors de [0, 1] : {melange}"
        raise ValueError(msg)
    return melange


def absorption_d_une_sphere(coefficient: float, rayon: float) -> float:
    """La moyenne de `1 - exp(-a l)` sur les cordes d'une sphere, vue de sa surface."""
    epaisseur = 2 * coefficient * rayon
    reste = -math.expm1(-epaisseur) - epaisseur * math.exp(-epaisseur)
    return 1 - 2 * reste / epaisseur**2


def rayon_equivalent(absorption: float, coefficient: float) -> float:
    """`L_eq`, tel que `1 - exp(-a L_eq)` soit l'absorption moyenne."""
    if not 0 <= absorption < 1:
        msg = f"absorption hors de [0, 1[ : {absorption}"
        raise ValueError(msg)
    return -math.log1p(-absorption) / coefficient


def rayon_moyen(volume: float, surface: float, *, facteur: float = 0.88) -> float:
    """`L_m = x L_o`, soit `3,52 V / S` avec `x = 0,88`."""
    bas, haut = FACTEURS_DE_FORME
    if not bas < facteur < haut:
        msg = f"facteur de forme hors de ]{bas}, {haut}[ : {facteur}"
        raise ValueError(msg)
    return facteur * longueur_moyenne_de_rayonnement(volume, surface)


def flux_entre_le_gaz_et_son_enveloppe(
    surface: float,
    *,
    gaz: tuple[float, float],
    paroi: tuple[float, float],
    emissivite: float = 1.0,
) -> float:
    """Le flux net du gaz `(eps_g, T_g)` vers sa paroi `(alpha_g, T_S)`."""
    if not 0 < emissivite <= 1:
        msg = f"emissivite hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)
    emis = gaz[0] * emittance(gaz[1])
    absorbe = paroi[0] * emittance(paroi[1])
    return (emissivite + 1) / 2 * surface * (emis - absorbe)


def aire_d_echange_avec_le_gaz(
    recepteur: float,
    paroi: float,
    *,
    gaz: float,
    vers_le_recepteur: float,
    emissivite: float,
) -> float:
    """`GS_1|R`, pour un gaz, un recepteur gris et une paroi adiabatique."""
    relais = 0.0
    if vers_le_recepteur > 0:
        transmis = vers_le_recepteur * (1 - gaz)
        relais = (paroi / recepteur) / (1 / gaz + 1 / transmis)
    noires = recepteur * (gaz + relais)
    surface = (1 - emissivite) / (emissivite * recepteur)
    return 1 / (1 / noires + surface)
