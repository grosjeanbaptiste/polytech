"""La derivation numerique par differences finies.

Sur une grille de pas `h`, la derivee d'une fonction au noeud `k`
s'approche par la difference **centree**
`(f_(k+1) - f_(k-1)) / (2 h)`, dont l'erreur est en `h^2` : les termes
d'ordre impair des deux developpements de Taylor se retranchent, et le
premier qui reste porte `h^2 f''' / 6`. Elle est donc exacte sur tout
polynome de degre deux, et diviser le pas par deux divise l'erreur par
quatre.

C'est la brique des methodes numeriques de la mecanique des fluides et de
la thermique, qui l'emploient l'une et l'autre.
"""

from __future__ import annotations


def derivee_centree(avant: float, apres: float, pas: float) -> float:
    """`(f_(k+1) - f_(k-1)) / (2 h)`, `avant` etant `f_(k-1)` et `apres` `f_(k+1)`."""
    if pas <= 0:
        msg = f"pas nul ou negatif : {pas}"
        raise ValueError(msg)
    return (apres - avant) / (2 * pas)
