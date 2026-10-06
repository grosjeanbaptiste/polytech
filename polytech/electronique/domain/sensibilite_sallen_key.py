"""La sensibilite du facteur de qualite d'un Sallen-Key a ses composants.

`S_x = (x / Q) dQ/dx` : une erreur relative de 1 % sur le composant `x`
deplace `Q` de `S_x` %. Le cours (chapitre 3, partie II, planches 26 et
27) nomme une methode « a sensibilite moderee » et une « a sensibilite
minimale » ; ce module chiffre ce que ces noms recouvrent.

Deux faits que le calcul rend visibles. Dans la methode 4, les
sensibilites aux capacites valent plus et moins un demi *quel que soit*
`Q`, et celles aux resistances sont nulles. Et sa sensibilite au gain,
`2 Q^2`, n'est jamais sollicitee : le gain d'un suiveur est fixe par un
fil, non par un rapport de resistances.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas

COMPOSANTS: Final = ("r1", "r2", "c1", "c2", "gain")
"""Les cinq grandeurs dont `Q` depend, dans l'ordre du constructeur."""

_SYMBOLES: Final = sp.symbols("r_1 r_2 c_1 c_2 mu", positive=True)
_QUALITE: Final = SallenKeyPasseBas(*_SYMBOLES).qualite()


def sensibilites_de_la_qualite(cellule: SallenKeyPasseBas) -> dict[str, sp.Expr]:
    """`S_x` pour chacun des cinq composants, aux valeurs de la cellule."""
    valeurs = dict(
        zip(_SYMBOLES, (getattr(cellule, nom) for nom in COMPOSANTS), strict=True)
    )
    return {
        nom: sp.simplify(
            (symbole / _QUALITE * sp.diff(_QUALITE, symbole)).subs(valeurs)
        )
        for nom, symbole in zip(COMPOSANTS, _SYMBOLES, strict=True)
    }
