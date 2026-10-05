"""La convection forcee le long d'une plaque plane.

Syllabus de thermique, tome 2, premiere partie, section 3.2. La couche
limite thermique suit la couche limite dynamique : leurs epaisseurs sont
dans le rapport `Pr^-1/3`, confondues pour un gaz, la thermique bien
plus mince pour un liquide visqueux.

En laminaire, `Nu_x = 0,332 Re_x^0,5 Pr^1/3` : le coefficient local
decroit comme l'inverse de l'epaisseur, et sa moyenne sur la plaque vaut
le double de sa valeur au bord de fuite.

Au-dela de la transition, `delta / x = 0,376 Re_x^-0,2`, et la
sous-couche visqueuse en occupe la fraction `194 Re_x^-0,7`. Le Nusselt
moyen d'une plaque dont le debut est laminaire se raccorde aux deux
lois : `Pr^1/3 [0,664 Re_c^0,5 + 0,037 (Re_L^0,8 - Re_c^0,8)]`.
"""

from __future__ import annotations

from polytech.thermique.domain.convection.couches_limites import (
    REYNOLDS_CRITIQUE_SUR_PLAQUE,
)

_TIERS = 1.0 / 3.0
_CRITIQUE = REYNOLDS_CRITIQUE_SUR_PLAQUE[1]


def rapport_des_couches_limites(prandtl: float) -> float:
    """`delta_t / delta = Pr^-1/3`."""
    return float(prandtl**-_TIERS)


def nusselt_local_laminaire(reynolds: float, prandtl: float) -> float:
    """`Nu_x = 0,332 Re_x^0,5 Pr^1/3`."""
    if reynolds >= _CRITIQUE:
        msg = f"couche limite turbulente : Re_x = {reynolds:.3g}"
        raise ValueError(msg)
    return float(0.332 * reynolds**0.5 * prandtl**_TIERS)


def epaisseur_turbulente(distance: float, reynolds: float) -> float:
    """`delta = 0,376 x Re_x^-0,2`, en metres."""
    return float(0.376 * distance * reynolds**-0.2)


def rapport_de_la_sous_couche(reynolds: float) -> float:
    """`delta_l / delta = 194 Re_x^-0,7`."""
    return float(194.0 * reynolds**-0.7)


def nusselt_moyen_mixte(
    reynolds: float,
    prandtl: float,
    *,
    critique: float = _CRITIQUE,
) -> float:
    """Le Nusselt moyen d'une plaque laminaire jusqu'a `critique`, turbulente apres."""
    laminaire = 0.664 * min(reynolds, critique) ** 0.5
    turbulent = 0.037 * (max(reynolds, critique) ** 0.8 - critique**0.8)
    return float((laminaire + turbulent) * prandtl**_TIERS)
