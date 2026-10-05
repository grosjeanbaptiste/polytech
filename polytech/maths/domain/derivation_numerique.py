"""La derivation numerique par differences finies.

Sur une grille de pas `h`, la derivee d'une fonction au noeud `k`
s'approche par la difference **centree**
`(f_(k+1) - f_(k-1)) / (2 h)`, dont l'erreur est en `h^2` : les termes
d'ordre impair des deux developpements de Taylor se retranchent, et le
premier qui reste porte `h^2 f''' / 6`. Elle est donc exacte sur tout
polynome de degre deux, et diviser le pas par deux divise l'erreur par
quatre.

Les differences **decentrees**, `(f_(k+1) - f_k) / h` a droite et
`(f_k - f_(k-1)) / h` a gauche, ne sont que d'ordre un. A l'inverse,
prendre deux noeuds de chaque cote porte la derivee seconde a l'ordre
quatre :

`(-f_(k-2) + 16 f_(k-1) - 30 f_k + 16 f_(k+1) - f_(k+2)) / (12 h^2)`.

Ce sont les briques des methodes numeriques de la mecanique des fluides
et de la thermique, qui les emploient l'une et l'autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

CINQ = 5


def _exiger_le_pas(pas: float) -> None:
    if pas <= 0:
        msg = f"pas nul ou negatif : {pas}"
        raise ValueError(msg)


def derivee_centree(avant: float, apres: float, pas: float) -> float:
    """`(f_(k+1) - f_(k-1)) / (2 h)`, `avant` etant `f_(k-1)` et `apres` `f_(k+1)`."""
    _exiger_le_pas(pas)
    return (apres - avant) / (2 * pas)


def derivee_a_droite(centre: float, apres: float, pas: float) -> float:
    """`(f_(k+1) - f_k) / h`, d'ordre un."""
    _exiger_le_pas(pas)
    return (apres - centre) / pas


def derivee_a_gauche(avant: float, centre: float, pas: float) -> float:
    """`(f_k - f_(k-1)) / h`, d'ordre un."""
    _exiger_le_pas(pas)
    return (centre - avant) / pas


def derivee_seconde_d_ordre_quatre(valeurs: Sequence[float], pas: float) -> float:
    """La derivee seconde au noeud central de cinq noeuds consecutifs."""
    _exiger_le_pas(pas)
    if len(valeurs) != CINQ:
        msg = f"cinq valeurs attendues, et non {len(valeurs)}"
        raise ValueError(msg)
    poids = (-1, 16, -30, 16, -1)
    somme = sum(p * v for p, v in zip(poids, valeurs, strict=True))
    return somme / (12 * pas**2)
