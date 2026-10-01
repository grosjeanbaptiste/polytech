"""Les deux proprietes fondamentales du champ magnetique.

Symetriquement a l'electrostatique, les planches les nomment par le flux
et par la circulation -- mais les roles sont **echanges**, et c'est tout
l'interet du parallele que les planches tracent explicitement :

* le flux de `B` est **toujours nul** (`div B = 0`) : il n'y a pas de
  charge magnetique ;
* la circulation de `H` **ne l'est pas** : elle vaut le courant enlace,
  c'est le theoreme d'Ampere.

C'est l'inverse exact de l'electrostatique, ou la divergence vaut la
charge et la circulation est nulle. Confondre les deux est l'erreur que ce
parallele sert a prevenir.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)

if TYPE_CHECKING:
    import sympy as sp


def est_un_champ_magnetique(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> bool:
    """Vrai si `div B = 0` : condition necessaire, et jamais facultative."""
    return bool(divergence_cartesienne(champ) == 0)


def induction_du_fil_infini(courant: float, distance: float) -> float:
    """`B = mu_0 I / (2 pi r)` : Biot et Savart integre sur un fil droit."""
    if distance <= 0:
        msg = f"distance au fil nulle ou negative : {distance} m"
        raise ValueError(msg)
    return PERMEABILITE_DU_VIDE * courant / (2 * math.pi * distance)


def courant_enlace(champ: float, longueur: float) -> float:
    """`N I = H l` : le theoreme d'Ampere, lu dans le sens du probleme.

    C'est exactement le geste de l'exercice 6 sur les circuits
    magnetiques : on lit `H` sur la courbe du materiau, on multiplie par
    la longueur du contour, et l'on obtient les amperes-tours.
    """
    return champ * longueur


def flux_magnetique(induction: float, section: float) -> float:
    """`Phi = B S`, pour une induction uniforme et normale a la section."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return induction * section


def champ_depuis_l_induction(induction: float, relative: float = 1.0) -> float:
    """`H = B / (mu_r mu_0)` ; leve pour une permeabilite relative sous un."""
    if relative < 1:
        msg = (
            f"permeabilite relative sous l'unite : {relative} -- un materiau "
            f"ferromagnetique la porte a plusieurs milliers"
        )
        raise ValueError(msg)
    return induction / (relative * PERMEABILITE_DU_VIDE)


def _produit_vectoriel(
    premier: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Le produit vectoriel, ecrit une fois pour les deux forces du chapitre."""
    a1, a2, a3 = premier
    b1, b2, b3 = second
    return (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)


def force_de_lorentz(
    charge: float,
    vitesse: tuple[float, float, float],
    induction: tuple[float, float, float],
) -> tuple[float, float, float]:
    """`q v x B`, en newtons.

    Elle est **orthogonale a la vitesse**, donc elle ne travaille jamais :
    un champ magnetique courbe une trajectoire sans changer l'energie du
    porteur. C'est la propriete qui distingue le magnetisme de
    l'electrostatique, et elle se verifie par un produit scalaire nul.
    """
    un, deux, trois = _produit_vectoriel(vitesse, induction)
    return charge * un, charge * deux, charge * trois


def force_sur_un_element_de_courant(
    courant: float,
    element: tuple[float, float, float],
    induction: tuple[float, float, float],
) -> tuple[float, float, float]:
    """`I dl x B` : la force de Lorentz sous sa forme integrable.

    C'est elle que le fascicule emploie, parce qu'un circuit se decoupe en
    elements de courant alors qu'il ne se decoupe pas en porteurs.
    """
    un, deux, trois = _produit_vectoriel(element, induction)
    return courant * un, courant * deux, courant * trois


def intensite_depuis_la_densite(densite: float, section: float) -> float:
    """`I = J S` pour une densite uniforme et normale a la section."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return densite * section


def continuite_a_une_interface() -> tuple[str, str]:
    """Les deux composantes qui se conservent en traversant une interface.

    Le point contre-intuitif que le fascicule met en regard de
    l'electrostatique : ici c'est la composante **normale** de `B` et la
    **tangentielle** de `H` ; la-bas c'etait la normale de `D` et la
    tangentielle de `E`. Les deux couples ne se correspondent pas, et les
    confondre fait traverser une interface de travers.
    """
    return "composante normale de B", "composante tangentielle de H"


def potentiel_vecteur_du_fil_infini(courant: float, distance: float) -> float:
    """`-(mu_0 I / 2 pi) ln r` : l'expression macroscopique du potentiel vecteur.

    Le fascicule la donne sous « expression macroscopique du potentiel
    vecteur : resultat ». Elle decroit **logarithmiquement**, et non comme
    `1/r` -- c'est son rotationnel qui decroit ainsi, et c'est lui qui est
    l'induction.
    """
    if distance <= 0:
        msg = f"distance nulle ou negative : {distance} m"
        raise ValueError(msg)
    return -PERMEABILITE_DU_VIDE * courant * math.log(distance) / (2 * math.pi)
