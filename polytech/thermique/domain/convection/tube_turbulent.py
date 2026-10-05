"""La convection forcee dans un tube, en ecoulement turbulent.

Syllabus de thermique, tome 2, premiere partie, section 3.3.2. La couche
limite turbulente se referme vite : l'effet d'entree se neglige, et le
coefficient moyen est celui de l'ecoulement etabli. Il ne reste a la
paroi qu'une sous-couche visqueuse, `delta_l / d = 63,5 Re_d^-0,8`, qui
porte l'essentiel de la resistance.

Pour les fluides peu visqueux, Colburn : `Nu = 0,0234 Re^0,8 Pr^1/3`.
Pour les fluides visqueux, Sieder et Tate y ajoutent le rapport des
viscosites. Dans la zone de transition, une relation fait intervenir
l'elancement `d / L`.

Ces relations se rassemblent sur un seul graphique par le facteur
`j_H = Nu Pr^-1/3 (mu_m / mu_p)^-0,14 / Re`. Une section non circulaire
s'y ramene par son diametre hydraulique, `4 A / p`.
"""

from __future__ import annotations

from polytech.thermique.domain.convection.couches_limites import (
    REYNOLDS_CRITIQUE_EN_TUBE,
)

_TIERS = 1.0 / 3.0


def sous_couche_en_tube(diametre: float, reynolds: float) -> float:
    """`delta_l = 63,5 d Re_d^-0,8`, en metres."""
    return float(63.5 * diametre * reynolds**-0.8)


def _exiger_le_turbulent(reynolds: float) -> None:
    if reynolds <= REYNOLDS_CRITIQUE_EN_TUBE:
        msg = f"ecoulement non turbulent : Re_d = {reynolds:.3g}"
        raise ValueError(msg)


def nusselt_de_colburn(reynolds: float, prandtl: float) -> float:
    """`Nu = 0,0234 Re^0,8 Pr^1/3`."""
    _exiger_le_turbulent(reynolds)
    return float(0.0234 * reynolds**0.8 * prandtl**_TIERS)


def nusselt_turbulent_de_sieder_et_tate(
    reynolds: float, prandtl: float, *, viscosites: float = 1.0
) -> float:
    """`Nu = 0,027 Re^0,8 Pr^1/3 (mu_m / mu_p)^0,14`."""
    _exiger_le_turbulent(reynolds)
    return float(0.027 * reynolds**0.8 * prandtl**_TIERS * viscosites**0.14)


def nusselt_de_transition(
    reynolds: float, prandtl: float, *, elancement: float, viscosites: float = 1.0
) -> float:
    """`0,116 (Re^2/3 - 125) Pr^1/3 [1 + (d/L)^2/3] (mu_m / mu_p)^0,14`."""
    _exiger_le_turbulent(reynolds)
    entree = 1 + elancement ** (2 / 3)
    coeur = 0.116 * (reynolds ** (2 / 3) - 125) * prandtl**_TIERS
    return float(coeur * entree * viscosites**0.14)


def facteur_j(
    nusselt: float, reynolds: float, prandtl: float, *, viscosites: float = 1.0
) -> float:
    """`j_H = Nu Pr^-1/3 (mu_m / mu_p)^-0,14 / Re`."""
    return float(nusselt * prandtl**-_TIERS * viscosites**-0.14 / reynolds)


def diametre_hydraulique_de_la_section(aire: float, perimetre: float) -> float:
    """`d_hyd = 4 A / p`, `p` etant le perimetre mouille."""
    if perimetre <= 0:
        msg = f"perimetre nul ou negatif : {perimetre} m"
        raise ValueError(msg)
    return 4 * aire / perimetre
