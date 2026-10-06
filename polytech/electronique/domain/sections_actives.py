"""Les sections d'un filtre actif en cascade (chapitre 3, partie II).

Planches 9 a 17. Un etage a AOP a une impedance d'entree tres grande et
une impedance de sortie tres petite : les sections se suivent sans se
charger, et la fonction de transfert globale est le *produit* des leurs.
Un filtre d'ordre eleve se decoupe donc en sections du second ordre --
plus une seule du premier ordre quand l'ordre est impair.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.butterworth import poles_de_butterworth
from polytech.electronique.domain.netliste import Admittance, SourceDeTension

_COURANT: Final = sp.Symbol("I_e")
_ENTREE, _MOINS, _SORTIE = 1, 2, 3


def biquads_de_butterworth(ordre: int) -> tuple[float, ...]:
    """Les `b_1` des cellules `s^2 + b_1 s + 1` du prototype, croissants.

    Planche 11 : les poles vont par paires conjuguees, et une paire
    `p, p*` de module un donne `s^2 - 2 Re(p) s + 1`. Donc `Q = 1 / b_1`.
    """
    if ordre % 2:
        msg = f"ordre pair attendu, une section du premier ordre reste : {ordre}"
        raise ValueError(msg)
    hauts = (pole for pole in poles_de_butterworth(ordre) if pole.imag > 0)
    return tuple(sorted(-2 * pole.real for pole in hauts))


def section_inverseuse(entree: sp.Expr, retour: sp.Expr) -> sp.Expr:
    """`E_2 / E_1 = -Y_1 / Y_2` : `Y_1` en entree, `Y_2` en contre-reaction.

    Planche 15. Diviser les deux admittances par un meme facteur ne
    change rien : c'est la mise a l'echelle des impedances.
    """
    elements = (
        SourceDeTension(_ENTREE, 0, sp.Integer(1), _COURANT),
        Admittance(_ENTREE, _MOINS, entree),
        Admittance(_MOINS, _SORTIE, retour),
        *aop(0, _MOINS, _SORTIE),
    )
    return sortie_ideale(elements, _SORTIE, _SORTIE)


def section_non_inverseuse(vers_la_masse: sp.Expr, retour: sp.Expr) -> sp.Expr:
    """`E_2 / E_1 = 1 + Y_1 / Y_2` : `Y_1` vers la masse, `Y_2` en retour.

    Planche 17. Pour realiser `(a_1 s + a_0) / (s + b_0)`, il faut
    `Y_1 / Y_2 = ((a_1 - 1) s + (a_0 - b_0)) / (s + b_0)`, donc `a_1 > 1`
    et `a_0 > b_0` pour que les composants soient positifs.
    """
    elements = (
        SourceDeTension(_ENTREE, 0, sp.Integer(1), _COURANT),
        Admittance(_MOINS, 0, vers_la_masse),
        Admittance(_MOINS, _SORTIE, retour),
        *aop(_ENTREE, _MOINS, _SORTIE),
    )
    return sortie_ideale(elements, _SORTIE, _SORTIE)
