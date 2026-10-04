"""La bande passante d'une chaine d'etages, qui retrecit a chaque etage.

`amplification.nombre_d_etages` suit la regle du cours : chaque etage
tient la bande visee, donc il en faut `ceil(log(G) / log(GBW / B))`.
C'est la bande de *chaque* etage. Celle de la chaine est plus etroite :
`n` etages du premier ordre, tous coupes a `f`, perdent ensemble
`3 dB` des `f sqrt(2^(1/n) - 1)`. Ce module tient compte de ce
retrecissement.
"""

from __future__ import annotations

import math
from typing import Final

_ETAGES_AU_PLUS: Final = 50
"""Au-dela, le gain par etage tend vers un et la bande ne gagne plus rien."""


def bande_de_la_cascade(etages: int, bande_par_etage: float) -> float:
    """La frequence a `-3 dB` de `etages` passe-bas identiques du premier ordre."""
    if etages < 1 or bande_par_etage <= 0:
        msg = f"cascade sans etage ou sans bande : {etages}, {bande_par_etage}"
        raise ValueError(msg)
    return bande_par_etage * math.sqrt(2 ** (1 / etages) - 1)


def nombre_d_etages_en_cascade(
    gain_vise: float,
    produit_gain_bande: float,
    bande_visee: float,
) -> int:
    """Le plus petit nombre d'etages identiques dont la *chaine* tient la bande.

    Chaque etage prend la racine `n`-ieme du gain, donc la bande
    `GBW / G^(1/n)` ; la chaine retrecit ensuite cette bande.
    """
    for etages in range(1, _ETAGES_AU_PLUS + 1):
        par_etage = abs(gain_vise) ** (1 / etages)
        if bande_de_la_cascade(etages, produit_gain_bande / par_etage) >= bande_visee:
            return etages
    msg = (
        f"aucune cascade de {_ETAGES_AU_PLUS} etages au plus ne tient "
        f"{bande_visee} Hz avec un gain {gain_vise} et GBW {produit_gain_bande}"
    )
    raise ValueError(msg)
