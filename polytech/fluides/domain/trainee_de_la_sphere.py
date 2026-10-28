"""La courbe `C_D(Re)` d'une sphere lisse et la crise de trainee.

Fascicule 8, planches « Trainee d'une sphere » et « Promoteur de
turbulence » : le coefficient suit `24 / Re` (Stokes), reste presque
constant (~0,4) quand le sillage est large, puis **chute brusquement**
quand la couche limite devient turbulente -- elle resiste mieux au
gradient de pression adverse, decolle plus tard, le sillage se resserre.
Les alveoles d'une balle de golf jouent le role du fil de declenchement :
la crise arrive a un Reynolds plus bas, celui du jeu.

Le cours ne donne pas la courbe en formule : elle vient de la correlation
de Morrison (2013, *An improved correlation for drag on a smooth sphere*,
`Re <= 10^6`), lue dans son implementation `fluids.drag.Morrison`, via KAG :
https://fluids.readthedocs.io/_modules/fluids/drag.html (verifie : les
quatre termes, `24/Re`, `2,6 (Re/5)/(1+(Re/5)^1,52)`,
`0,411 (Re/263000)^-7,94 / (1+(Re/263000)^-8)` et `Re^0,8 / 461000`, et la
borne `Re <= 1E6`). Le PDF de Morrison (pages.mtu.edu/~fmorriso) n'est pas
lisible par KAG (flux compresses) : la source est donc secondaire.

Reynolds critique de la sphere lisse : « de 0,5 a 0,2 vers un Reynolds de
300000 » (https://en.wikipedia.org/wiki/Drag_crisis, verifie par KAG).
Balle de golf : crise « vers 5.10^4 dans l'experience »
(https://link.springer.com/article/10.1007/s10494-017-9859-1, verifie par
KAG) ; la valeur depend de la geometrie des alveoles. La forme de `C_D`
apres la crise pour la balle alveolee n'est pas donnee par ces sources :
elle n'est pas modelisee (seul `est_apres_la_crise_de_trainee` l'est).
"""

from __future__ import annotations

from typing import Final

REYNOLDS_CRITIQUE_SPHERE_LISSE: Final = 3.0e5
"""La crise de la sphere lisse : `C_D` tombe sous 0,2 vers `Re = 3.10^5`."""

REYNOLDS_CRITIQUE_BALLE_DE_GOLF: Final = 5.0e4
"""Crise de la balle alveolee, ~5.10^4 (Springer ci-dessus, hors cours)."""

REYNOLDS_MAXIMAL_DE_LA_CORRELATION: Final = 1.0e6
"""Au-dela, la correlation de Morrison n'est plus etablie."""


def coefficient_de_trainee_de_la_sphere(reynolds: float) -> float:
    """`C_D` d'une sphere lisse, correlation de Morrison, `0 < Re <= 10^6`."""
    if not 0 < reynolds <= REYNOLDS_MAXIMAL_DE_LA_CORRELATION:
        msg = f"Re hors de ]0 ; 10^6] : Re = {reynolds}"
        raise ValueError(msg)
    r = reynolds
    return float(
        24 / r
        + 2.6 * (r / 5) / (1 + (r / 5) ** 1.52)
        + 0.411 * (r / 263_000) ** -7.94 / (1 + (r / 263_000) ** -8.0)
        + r**0.8 / 461_000
    )


def est_apres_la_crise_de_trainee(reynolds: float, reynolds_critique: float) -> bool:
    """La couche limite est turbulente : `Re >= Re_critique`."""
    if reynolds <= 0 or reynolds_critique <= 0:
        msg = f"Re non positif : Re = {reynolds}, Re critique = {reynolds_critique}"
        raise ValueError(msg)
    return reynolds >= reynolds_critique
