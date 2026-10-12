"""Une force est un vecteur localise, et son moment en depend.

« Une meme force n'aura pas le meme effet selon qu'elle est appliquee en
A, B ou C » : une force se donne donc avec son point d'application. Son
**moment** par rapport a un pole `O` traduit son effet en rotation :

    M_O(F/A) = OA ^ F.

Il ne change pas quand la force glisse sur sa ligne d'action, et sa
grandeur est celle de la force multipliee par le bras de levier.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
section 1.6.1, pages 13 et 14, relations (1.8) a (1.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.produit_vectoriel import produit_vectoriel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur

DIMENSION = 3


def de_l_espace(vecteur: Sequence[object]) -> Vecteur:
    """Le vecteur, apres verification qu'il a trois composantes."""
    if len(vecteur) != DIMENSION:
        msg = f"trois composantes attendues, et non {len(vecteur)}"
        raise ValueError(msg)
    x, y, z = (sp.sympify(composante) for composante in vecteur)
    return x, y, z


def difference(arrivee: Sequence[object], depart: Sequence[object]) -> Vecteur:
    """Le vecteur qui mene de `depart` a `arrivee`."""
    x, y, z = (
        a - d for a, d in zip(de_l_espace(arrivee), de_l_espace(depart), strict=True)
    )
    return x, y, z


@dataclass(frozen=True, slots=True)
class Force:
    """Une force `F/A` : son point d'application et son vecteur."""

    point: Vecteur
    vecteur: Vecteur

    def __post_init__(self) -> None:
        """Ramene les deux triplets a des expressions, et refuse tout autre format."""
        object.__setattr__(self, "point", de_l_espace(self.point))
        object.__setattr__(self, "vecteur", de_l_espace(self.vecteur))


def moment(force: Force, pole: Sequence[object]) -> Vecteur:
    """`M_O(F/A) = OA ^ F`, relation (1.8)."""
    return produit_vectoriel(difference(force.point, pole), force.vecteur)
