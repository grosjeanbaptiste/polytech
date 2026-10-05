"""La portance et la trainee, sommes des efforts de surface.

Planches du chapitre IX. Un fluide n'agit sur un corps que par sa
surface : la pression pousse suivant la normale, le frottement parietal
tire suivant la tangente. `theta` etant l'angle de la normale exterieure
avec l'ecoulement, un element d'aire `dA` recoit

`dF_x = -p cos(theta) dA + tau_w sin(theta) dA`,
`dF_y = -p sin(theta) dA - tau_w cos(theta) dA`,

et la trainee `D` et la portance `L` en sont les sommes. La planche ecrit
`+ tau_w cos(theta)` dans la portance ; sa propre figure, ou le
frottement est tangent a la paroi et dirige vers l'aval, impose le signe
moins.

`tau_w` y est algebrique : positif quand il tire dans le sens de la
normale tournee d'un quart de tour vers la droite, c'est-a-dire vers
l'aval sur le dessus du corps -- la ou la figure le dessine. Sur le
dessous, un frottement vers l'aval est donc negatif.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Facette:
    """Un element de paroi : pression, frottement algebrique, angle, aire."""

    pression: float
    frottement: float
    angle: float
    aire: float

    def __post_init__(self) -> None:
        """Refuse une facette sans etendue."""
        if self.aire <= 0:
            msg = f"aire nulle ou negative : {self.aire} m^2"
            raise ValueError(msg)

    def force(self) -> tuple[float, float]:
        """`(dF_x, dF_y)`, en newtons."""
        cosinus, sinus = math.cos(self.angle), math.sin(self.angle)
        selon_x = -self.pression * cosinus + self.frottement * sinus
        selon_y = -self.pression * sinus - self.frottement * cosinus
        return (selon_x * self.aire, selon_y * self.aire)


def resultante(facettes: Sequence[Facette]) -> tuple[float, float]:
    """`(D, L)` : la trainee suivant l'ecoulement, la portance en travers."""
    if not facettes:
        msg = "aucune facette : la resultante n'est pas definie"
        raise ValueError(msg)
    forces = [facette.force() for facette in facettes]
    return (sum(f[0] for f in forces), sum(f[1] for f in forces))
