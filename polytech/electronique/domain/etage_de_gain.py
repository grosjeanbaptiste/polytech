"""Une cellule active autour d'un etage de gain `mu`, resolue par ses noeuds.

Les cellules de Sallen-Key du chapitre 3 (partie II, planches 22 a 43) ont
toutes la meme forme : un reseau de resistances et de capacites, et un
amplificateur non inverseur de gain `mu = 1 + R_b / R_a` dont l'entree ne
tire aucun courant. Les planches en donnent les fonctions de transfert
« obtenues avec Matlab » ; ce module les fait sortir de l'analyse nodale,
ce qui permet a un test de les confronter a ce qui est imprime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electronique.domain.netliste import (
    Element,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ENTREE: Final = 1
"""Le noeud que la source attaque, par convention de ce module."""

_COURANT_D_ENTREE: Final = sp.Symbol("I_e")
_COURANT_DE_SORTIE: Final = sp.Symbol("I_s")


def transfert_autour_d_un_gain(
    reseau: Sequence[Element],
    noeuds: int,
    *,
    commande: int,
    sortie: int,
    gain: sp.Expr,
) -> sp.Expr:
    """`E_2 / E_1` : la sortie vaut `gain` fois la tension du noeud `commande`."""
    elements: list[Element] = [
        *reseau,
        SourceDeTension(ENTREE, 0, sp.Integer(1), _COURANT_D_ENTREE),
        SourceDeTensionCommandee(sortie, 0, gain, (commande, 0), _COURANT_DE_SORTIE),
    ]
    return sp.cancel(resoudre(elements, noeuds)[sortie - 1])
