"""Les trois modes de transfert, et ce que leur superposition change.

Chapitre d'introduction du cours, et il n'est pas administratif : il porte
la loi de refroidissement de **Newton** (1701), les trois modes de
convection, et le « cas reel » ou plusieurs modes agissent ensemble.

La loi de Newton est **lineaire** en ecart de temperature, la ou le
rayonnement est en puissance quatrieme. Le chapitre les met en regard pour
cette raison : a faible ecart le rayonnement se linearise et se range dans
un coefficient d'echange, a grand ecart il ne s'y range plus.

Et la superposition est le piege d'entree. Trois parois **en serie**
additionnent leurs resistances ; trois modes sur **une meme** paroi
additionnent leurs coefficients. Ce sont deux assemblages inverses, et
c'est pourquoi le chapitre parle de « superposition des modes » et non
d'addition de resistances.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

RAPPORT_DE_DOMINANCE: Final = 10.0
"""Au-dela de ce rapport, un mode de convection donne son nom au regime.

Le seuil est choisi et dit : entre un dixieme et dix, le chapitre parle de
convection **mixte**, et ses correlations ne sont ni celles de la
convection naturelle ni celles de la forcee.
"""


class ModeDeConvection(Enum):
    """Les trois regimes que le chapitre distingue."""

    NATURELLE = "naturelle : le mouvement nait de l'ecart de densite"
    FORCEE = "forcee : le mouvement est impose de l'exterieur"
    MIXTE = "mixte : les deux contribuent, et aucune correlation simple ne vaut"


def flux_de_newton(echange: float, aire: float, ecart: float) -> float:
    """`phi = K S (t_p - t_f)`, en watts : la loi de 1701.

    Le signe de l'ecart est conserve : il dit le sens du transfert, et le
    perdre ferait disparaitre une direction d'un bilan.
    """
    if echange < 0:
        msg = f"coefficient d'echange negatif : {echange} W/(m^2 K)"
        raise ValueError(msg)
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    return echange * aire * ecart


def mode_de_convection(force: float, naturel: float) -> ModeDeConvection:
    """Le regime que ces deux coefficients designent."""
    if force <= 0 or naturel <= 0:
        msg = (
            f"coefficient de convection nul ou negatif : force {force}, "
            f"naturel {naturel} W/(m^2 K)"
        )
        raise ValueError(msg)
    if force >= RAPPORT_DE_DOMINANCE * naturel:
        return ModeDeConvection.FORCEE
    if naturel >= RAPPORT_DE_DOMINANCE * force:
        return ModeDeConvection.NATURELLE
    return ModeDeConvection.MIXTE


def coefficient_superpose(coefficients: tuple[float, ...]) -> float:
    """`K = somme K_i` : les modes agissant en parallele s'additionnent.

    C'est l'inverse de l'assemblage en serie, ou ce sont les resistances
    qui s'additionnent. Le chapitre previent d'entree, et la confusion
    donne un coefficient global plus petit que le plus petit des modes --
    ce qui est absurde.
    """
    if not coefficients:
        msg = "aucun mode de transfert : il n'y a rien a superposer"
        raise ValueError(msg)
    if any(coefficient < 0 for coefficient in coefficients):
        msg = f"coefficient negatif dans la superposition : {coefficients}"
        raise ValueError(msg)
    return sum(coefficients)


def mode_dominant(coefficients: dict[str, float]) -> str:
    """Le mode au plus grand coefficient ; leve si deux sont a egalite.

    C'est la question que le chapitre pose sur chacune de ses applications
    -- l'aerotherme, le refroidisseur de processeur, le cas reel. Designer
    un mode au hasard en cas d'egalite serait arbitraire, et masquerait
    justement le cas interessant.
    """
    if not coefficients:
        msg = "aucun mode fourni : il n'y a pas de dominant"
        raise ValueError(msg)
    ordonnes = sorted(coefficients.items(), key=lambda couple: -couple[1])
    attendu = 2
    if len(ordonnes) >= attendu and ordonnes[0][1] == ordonnes[1][1]:
        msg = f"aucun mode ne domine : egalite a {ordonnes[0][1]} W/(m^2 K)"
        raise ValueError(msg)
    return ordonnes[0][0]


MODES_FONDAMENTAUX: Final = ("conduction", "convection", "rayonnement")
"""Les trois modes que le chapitre nomme fondamentaux.

Trois, et non quatre : le changement de phase n'est pas un mode distinct
mais de la convection avec un terme latent. Le chapitre le pose ainsi, et
les porter comme donnee evite de les recompter a chaque bilan.
"""

ECART_MAXIMAL_DE_NEWTON: Final = 200.0
"""Au-dela, la loi de Newton cesse de valoir, en kelvins.

Le seuil est choisi et dit. La loi suppose un coefficient d'echange
**constant** ; a grand ecart les proprietes du fluide varient et le
rayonnement, en puissance quatrieme, finit par dominer. Deux cents kelvins
est l'ordre de grandeur que le chapitre retient dans ses remarques.
"""


def resistance_depuis_l_echange(echange: float, aire: float) -> float:
    """`R = 1 / (K S)`, en K/W : ce qui rend l'analogie electrique possible.

    Et c'est pourquoi tout le reste du cours parle en resistances plutot
    qu'en coefficients : les resistances s'additionnent en serie, ce que
    les coefficients ne font pas.
    """
    if echange <= 0:
        msg = f"echange nul ou negatif : {echange} W/(m^2 K)"
        raise ValueError(msg)
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    return 1.0 / (echange * aire)


def loi_de_newton_valide(ecart: float) -> bool:
    """Vrai si l'ecart reste dans le domaine ou le coefficient est constant.

    C'est le sujet de la planche « remarques - validite de la loi de
    Newton ». Son module seul compte : refroidir et chauffer sont
    symetriques de ce point de vue.
    """
    return abs(ecart) <= ECART_MAXIMAL_DE_NEWTON


def gain_de_la_convection(conduction: float, convection: float) -> float:
    """Le rapport des deux coefficients, qui doit depasser l'unite.

    « Conductibilite pure / convection » : a epaisseur egale, un fluide en
    mouvement transporte un ordre de grandeur de plus qu'un fluide au
    repos. C'est tout l'interet de le mettre en mouvement, et c'est le
    chiffre qui le justifie.
    """
    if conduction <= 0 or convection <= 0:
        msg = f"coefficient non positif : {conduction} ou {convection} W/(m^2 K)"
        raise ValueError(msg)
    rapport = convection / conduction
    if rapport < 1:
        msg = (
            f"gain de convection sous l'unite : {rapport:.4g} -- un fluide en "
            f"mouvement ne transporte pas moins qu'au repos"
        )
        raise ValueError(msg)
    return rapport


PUISSANCE_METABOLIQUE: Final = 100.0
"""La puissance qu'un corps humain au repos dissipe, en watts.

Ordre de grandeur de la litterature, que le chapitre emploie dans son
application de physiologie thermique. Aucun mode seul n'y suffirait a
temperature ambiante : c'est leur superposition qui l'evacue.
"""


def puissance_metabolique_au_repos() -> float:
    """Les cent watts d'un corps humain au repos, en watts.

    Application de physiologie thermique du chapitre. Le nombre sert
    d'ordre de grandeur : il dit pourquoi une piece se rechauffe quand on
    y entre, et pourquoi une salle comble se ventile.
    """
    return PUISSANCE_METABOLIQUE
