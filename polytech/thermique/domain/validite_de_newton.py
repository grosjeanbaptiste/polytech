"""Ou la loi de refroidissement de Newton vaut, et ou elle cesse de valoir.

Planche « Remarques - validite de la loi de Newton » de l'introduction.
La loi `phi = K (t_p - t_a)` suppose un coefficient d'echange constant,
c'est-a-dire independant des temperatures. Le cours en tire deux
remarques : la linearite ne tient a peu pres que pour la convection
forcee, et la loi ne vaut donc que dans un domaine restreint de
temperatures -- qu'il ne chiffre pas.
"""

from __future__ import annotations

from typing import Final

from polytech.thermique.domain.modes import ModeDeConvection
from polytech.thermique.domain.modes_fondamentaux import ModeFondamental

ECART_MAXIMAL_DE_NEWTON: Final = 200.0
"""Un domaine de validite **que le cours ne donne pas**, en kelvins.

La planche « validite de la loi de Newton » dit seulement « un domaine
restreint de temperatures ». Cette valeur est une convention de la
bibliotheque, conservee pour qui l'importait ; plus rien ne la lit, et
`loi_de_newton_valide` exige desormais son domaine.
"""


def loi_de_newton_valide(ecart: float, *, domaine: float) -> bool:
    """Vrai si l'ecart reste dans le `domaine` ou le coefficient est constant.

    C'est le sujet de la planche « remarques - validite de la loi de
    Newton » : « un domaine restreint de temperatures », que le cours ne
    chiffre pas et qui se donne donc. Son module seul compte : refroidir
    et chauffer sont symetriques de ce point de vue.
    """
    if domaine <= 0:
        msg = f"domaine nul ou negatif : {domaine} K"
        raise ValueError(msg)
    return abs(ecart) <= domaine


def echange_lineaire(mode: ModeDeConvection | ModeFondamental) -> bool:
    """Vrai si le flux est a peu pres proportionnel a l'ecart de temperature.

    Meme planche : c'est « a peu pres le cas pour la convection forcee »,
    pas pour le rayonnement ni pour la convection naturelle.
    """
    if mode is ModeDeConvection.FORCEE:
        return True
    if mode in (ModeDeConvection.NATURELLE, ModeFondamental.RAYONNEMENT):
        return False
    msg = f"le cours ne se prononce pas sur la linearite de : {mode.name}"
    raise ValueError(msg)
