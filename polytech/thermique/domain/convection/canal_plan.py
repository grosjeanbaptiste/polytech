"""La convection forcee entre deux plaques paralleles, en turbulent developpe.

`Nu = 0,0158 Re^0,8`, la correlation que donne l'enonce du TP 3, sur le
**diametre hydraulique** `4 A / P` : pour deux plaques larges devant leur
ecartement, il vaut deux fois l'ecartement. Le Reynolds se forme sur le
debit massique, `Re = q D_h / (A mu)`, ce qui evite de passer par la
vitesse.

La correlation est celle du turbulent ; sous un Reynolds de 2300,
l'ecoulement est laminaire et elle ne s'applique pas -- la fonction le
dit au lieu de rendre un coefficient faux.
"""

from __future__ import annotations

from typing import Final

TRANSITION: Final = 2300.0
"""Le Reynolds sous lequel un ecoulement en conduite reste laminaire."""


def diametre_hydraulique(ecartement: float) -> float:
    """`4 A / P` pour deux plaques larges : deux fois l'ecartement."""
    return 2 * ecartement


def reynolds_du_canal(
    debit: float, largeur: float, ecartement: float, viscosite: float
) -> float:
    """`Re = q D_h / (A mu)`, avec `A = largeur x ecartement`."""
    section = largeur * ecartement
    return debit * diametre_hydraulique(ecartement) / (section * viscosite)


def coefficient_du_canal(
    debit: float,
    largeur: float,
    ecartement: float,
    viscosite: float,
    conductivite: float,
) -> float:
    """`K = Nu lambda / D_h` ; leve si l'ecoulement est laminaire."""
    reynolds = reynolds_du_canal(debit, largeur, ecartement, viscosite)
    if reynolds < TRANSITION:
        msg = (
            f"Re = {reynolds:.0f} : laminaire, la correlation du turbulent ne vaut pas"
        )
        raise ValueError(msg)
    nusselt = float(0.0158 * reynolds**0.8)
    return nusselt * conductivite / diametre_hydraulique(ecartement)
