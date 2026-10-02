"""Une spire tournant dans un champ uniforme -- seance 2.1, exercice 5.

Une spire rectangulaire de cotes `l` et `L` tourne a la vitesse angulaire
`omega`. A `t = 0` sa normale est alignee sur le champ : le flux y est
maximal.

**Deux routes.** `fem_par_le_flux` derive le flux -- c'est celle du
manuscrit. `fem_par_les_cotes_actifs` integre le champ electromoteur
`v ^ B` sur les deux cotes paralleles a l'axe de rotation, et n'emploie
aucun flux.

Pour la seconde, on prend l'axe parallele aux cotes de longueur `L`,
passant par le milieu des cotes de longueur `l`. Chaque cote actif
tourne alors au rayon `l/2`. Le manuscrit ne dit pas lequel des deux
cotes porte l'axe ; le resultat n'en depend pas, la f.e.m. ne
contenant que le produit `l L`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

COTES_ACTIFS = 2
"""Les deux cotes paralleles a l'axe ; les deux autres ne coupent rien."""


def flux_a_travers_la_spire(
    induction: sp.Expr,
    petit_cote: sp.Expr,
    grand_cote: sp.Expr,
) -> sp.Expr:
    """`phi = B l L cos(omega t)` : le flux d'un champ uniforme."""
    return induction * petit_cote * grand_cote * sp.cos(PULSATION * TEMPS)


def fem_par_le_flux(
    induction: sp.Expr,
    petit_cote: sp.Expr,
    grand_cote: sp.Expr,
) -> sp.Expr:
    """`e = -d(phi)/dt` : la route du manuscrit."""
    return force_electromotrice(
        flux_a_travers_la_spire(induction, petit_cote, grand_cote),
    )


def fem_par_les_cotes_actifs(
    induction: sp.Expr,
    petit_cote: sp.Expr,
    grand_cote: sp.Expr,
) -> sp.Expr:
    """`e = 2 B L v sin(omega t)`, avec `v = omega l / 2`.

    La vitesse d'un cote actif est tangente a son cercle. Seule sa
    composante **perpendiculaire** au champ produit un champ
    electromoteur le long du cote, et elle vaut `v sin(omega t)` : nulle
    quand la spire est face au champ, maximale un quart de tour plus tard.
    """
    vitesse = PULSATION * petit_cote / 2
    par_cote = induction * grand_cote * vitesse * sp.sin(PULSATION * TEMPS)
    return COTES_ACTIFS * par_cote
