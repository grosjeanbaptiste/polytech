"""Les elements parasites d'un circuit reel.

Le syllabus d'electricite generale (15.13). Une resistance a toujours une
inductance propre residuelle, une bobine une resistance residuelle, et
les conducteurs de liaison ajoutent les leurs.

L'exemple traite : a la fermeture d'un circuit RC, le courant ne saute
pas de zero a `E / R`. Avec l'inductance parasite `l`, sa transformee est
`C E / (p^2 T^2 + 2 xi p T + 1)` (15.13.1), ou `T^2 = l C`, `tau = R C`
et `xi = tau / (2 T)` : c'est la decharge (15.12.5) pour `Q_0 = -C E`.
Le cas reel a `T` petit et `xi` grand : le courant monte en un temps tres
court, puis se confond avec `(E / R) exp(-t / tau)`. Le syllabus imprime
ce dernier avec l'exposant `-2 xi t / T` (15.13.3) ; `tau` valant
`2 xi T`, c'est `-t / (2 xi T)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.decharge_rlc import (
    Regime,
    courant_de_decharge,
    regime_de_la_decharge,
)

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def courant_rc_avec_inductance_parasite(
    fem: Grandeur,
    resistance: Grandeur,
    amortissement: Grandeur,
    periode: Grandeur,
) -> sp.Expr:
    """`i(t)` de (15.13.2), pour `xi = R C / (2 T)` superieur a un."""
    if regime_de_la_decharge(amortissement) is not Regime.APERIODIQUE:
        msg = f"amortissement {amortissement} : (15.13.2) suppose xi superieur a un"
        raise ValueError(msg)
    capacite = 2 * amortissement * periode / resistance
    return courant_de_decharge(amortissement, periode, -capacite * fem)
