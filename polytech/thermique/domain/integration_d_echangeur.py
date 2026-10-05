"""L'echangeur integre numeriquement, quand rien n'y est constant.

Syllabus de thermique, tome 2, deuxieme partie, section 2.6. Si le
coefficient global et les chaleurs massiques varient le long de
l'echangeur -- des fumees chaudes, ou le rayonnement domine --, la
moyenne logarithmique ne vaut plus. Le bilan lie `t_2` a `t_1` ; il
reste une seule equation,

    dh_1 / ds = -phi / q_m,1,                                     (2.18)

qui s'integre en `S = q_m,1 integrale(dh_1 / phi)` entre la sortie et
l'entree du fluide chaud. Pour un echangeur donne, l'inconnue est la
borne de l'integrale : la temperature de sortie.

Tracer les deux profils fait apparaitre le **point de pincement**, ou
l'ecart est minimal (figure 2.16). Cet ecart doit rester positif.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.integrate import quad

from polytech.maths.domain.fausse_position import racine_par_fausse_position

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TOLERANCE = 1e-9
"""En m^2, sur la surface retrouvee."""


def _unite(_: float) -> float:
    return 1.0


def surface_par_integration(
    debit: float,
    densite_de_flux: Callable[[float], float],
    *,
    entree: float,
    sortie: float,
    pente: Callable[[float], float] = _unite,
) -> float:
    """`S = q_m,1 integrale(c_1 dt_1 / phi)` ; `pente` est `dh_1 / dt_1`.

    Sans `pente`, `debit` est le debit de capacite `q_m,1 c_1`.
    """

    def integrande(chaud: float) -> float:
        return pente(chaud) / densite_de_flux(chaud)

    return debit * float(quad(integrande, sortie, entree, limit=200)[0])


def temperature_de_sortie_par_integration(
    debit: float,
    densite_de_flux: Callable[[float], float],
    *,
    entree: float,
    surface: float,
    plancher: float,
) -> float:
    """`t_1S` d'un echangeur de `surface` donnee, cherchee au-dessus du `plancher`."""

    def exces(sortie: float) -> float:
        offerte = surface_par_integration(
            debit, densite_de_flux, entree=entree, sortie=sortie
        )
        return offerte - surface

    return racine_par_fausse_position(exces, plancher, entree, tolerance=TOLERANCE)


def pincement(chaud: Sequence[float], froid: Sequence[float]) -> tuple[int, float]:
    """Le rang et la valeur du plus petit ecart entre les deux profils."""
    if len(chaud) != len(froid):
        msg = f"profils de longueurs differentes : {len(chaud)} et {len(froid)}"
        raise ValueError(msg)
    ecarts = [haut - bas for haut, bas in zip(chaud, froid, strict=True)]
    rang = min(range(len(ecarts)), key=ecarts.__getitem__)
    if ecarts[rang] <= 0:
        msg = f"les profils se croisent au rang {rang} : ecart de {ecarts[rang]}"
        raise ValueError(msg)
    return rang, ecarts[rang]
