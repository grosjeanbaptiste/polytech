"""L'induction : loi de Faraday, et les deux origines de la force electromotrice.

Les planches distinguent le champ **electromoteur** -- `v x B`, qui
apparait parce que le conducteur bouge -- du champ electrique **induit
dynamique**, qui apparait parce que `B` varie dans le temps. Les deux
produisent une force electromotrice, et la loi de Faraday les reunit :
`e = -dPhi/dt`, quelle que soit la raison pour laquelle `Phi` varie.

Le signe moins est la loi de Lenz, et il n'est pas decoratif : il dit que
l'effet s'oppose a la cause. L'omettre donne une machine qui s'emballe au
lieu de freiner, et c'est une faute que seul le signe revele.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import force_de_lorentz
from polytech.electromagnetisme.domain.symboles import TEMPS


def force_electromotrice(flux: sp.Expr) -> sp.Expr:
    """`e = -dPhi/dt` : la loi de Faraday, signe de Lenz compris."""
    return sp.simplify(-sp.diff(flux, TEMPS))


def fem_de_deplacement(induction: float, longueur: float, vitesse: float) -> float:
    """`e = B l v` pour une barre qui coupe les lignes de champ.

    C'est le champ electromoteur `v x B` integre le long de la barre, dans
    le cas ou les trois vecteurs sont orthogonaux -- le seul que le cours
    traite en formule fermee.
    """
    if longueur <= 0:
        msg = f"longueur de barre nulle ou negative : {longueur} m"
        raise ValueError(msg)
    return induction * longueur * vitesse


def champ_induit_dynamique(
    vitesse: tuple[float, float, float], induction: tuple[float, float, float]
) -> tuple[float, float, float]:
    """`E_d = v x B` : le champ qu'un conducteur entraine a la vitesse `v` voit.

    C'est la force de Lorentz par unite de charge, pour la seule vitesse
    d'entrainement : le terme de vitesse relative, lui, donne la force de
    Laplace sur le conducteur.
    """
    return force_de_lorentz(1.0, vitesse, induction)


def est_constante(expression: sp.Expr) -> bool:
    """Vrai si l'expression ne depend pas du temps."""
    return TEMPS not in expression.free_symbols


def flux_a_entrefers_dominants(
    spires: object, courant: object, section: object, air: object
) -> sp.Expr:
    """`Phi = mu_0 N I S / l` : le flux quand l'air consomme tous les amperes-tours.

    `air` est la longueur **totale** d'air que le flux traverse -- deux
    entrefers de `l_e` en font `2 l_e`. Le theoreme d'Ampere se reduit a
    `N I = H_e l`, et `B = mu_0 H_e` donne le flux. Le fer n'intervient
    plus : c'est l'hypothese, et elle cesse de valoir quand l'air devient
    assez long pour que la dispersion compte.
    """
    longueur = sp.sympify(air)
    if longueur == 0:
        msg = "aucun entrefer : le fer consommerait tous les amperes-tours"
        raise ValueError(msg)
    numerateur = PERMEABILITE * sp.sympify(spires) * sp.sympify(courant)
    return numerateur * sp.sympify(section) / longueur


def fem_d_un_entrefer_mobile(
    spires: object, courant: object, section: object, entrefer: object, vitesse: object
) -> sp.Expr:
    """`e(t) = mu_0 v N^2 I S / (2 (l_e + v t)^2)` : un demi-tore qui s'eloigne.

    Interrogation cotee 2022-2023 (Charleroi), question 2 : deux
    demi-tores de `N` spires chacun, separes par deux entrefers de `l_e`
    a `t = 0` ; celui de droite s'ecarte a la vitesse `v`. Le flux est
    celui des entrefers dominants, sur `2 (l_e + v t)` d'air, et
    l'enroulement de droite voit `e = -N dPhi/dt`.

    Le mouvement est uniforme, mais le flux varie comme l'**inverse** de
    l'entrefer : la force electromotrice n'est pas constante, elle
    decroit comme `1 / (l_e + v t)^2`.
    """
    tours, aire, depart = sp.sympify(spires), sp.sympify(section), sp.sympify(entrefer)
    if tours.is_positive is False:
        msg = f"nombre de spires nul ou negatif : {spires}"
        raise ValueError(msg)
    if aire.is_positive is False:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    if depart.is_positive is False:
        msg = f"entrefer initial nul ou negatif : {entrefer} m"
        raise ValueError(msg)
    air = 2 * (depart + sp.sympify(vitesse) * TEMPS)
    flux = flux_a_entrefers_dominants(tours, courant, aire, air)
    return force_electromotrice(tours * flux)
