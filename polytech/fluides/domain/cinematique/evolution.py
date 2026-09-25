"""Evolution materielle d'une particule (cours III.4).

Le volume et les ecarts materiels s'obtiennent en *integrant dans le temps*
les taux instantanes -- divergence pour le volume, taux de deformation
lineique pour chaque direction.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T

_AXES = (0, 1)


def loi_de_volume(divergence: sp.Expr) -> sp.Expr:
    """Rapport ``V(t)/V0``, integre depuis la divergence."""
    return sp.simplify(sp.exp(sp.integrate(divergence, (T, 0, T))))


def ecart_materiel(tenseur: sp.Matrix, axe: int) -> sp.Expr:
    """Rapport ``delta(t)/delta_0`` le long d'un axe principal."""
    if axe not in _AXES:
        msg = f"axe hors des deux directions du plan : {axe}"
        raise ValueError(msg)
    taux = tenseur[axe, axe]
    return sp.simplify(sp.exp(sp.integrate(taux, (T, 0, T))))
